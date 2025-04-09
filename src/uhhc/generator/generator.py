import geopandas as gpd
import pyosrm
import numpy as np
import random
import click
from ..models.instance_models import Instance, Patient, Caregiver, TerminalPoint, RequiredService, Synchronization, Service, TimeWindow
from statistics import mode
from typing import Any
from numpy.typing import NDArray
import math

def generate_spatial(points : int, density : gpd.GeoDataFrame, router : pyosrm.PyOSRM) -> tuple[list[tuple[float, float]], NDArray[np.int64]] | tuple[None, None]:
    centroid = density.centroid.values[0]
    click.echo(f"Generating {points} locations in a geographic area whose centroid is {centroid}")
    # Calculate the total population and area to set the bias
    total_population = density['OBS_VALUE_T'].sum()
    total_area = density.geometry.unary_union.area / 10**6
    density['bias']= density['OBS_VALUE_T'] / total_population
    click.echo(f"Total population: {int(total_population)}, total area: {total_area:.1f} km2")
    # Sample patients and departing_points according to the bias
    points_selected = density.sample(points, weights='bias', replace=True, random_state=42)    
    # Compute distances between the points, using the OSRM router (convert into geographical coordinates)
    locations = list(map(lambda p: (p.x, p.y), points_selected.centroid.to_crs(epsg=4326)))
    result = router.table(locations).json(to_timedelta=True)
    if result['code'] == 'Ok':
        click.echo(f"Distance matrix computed successfully")
        distances = (np.round(np.array(result['durations'], dtype=np.timedelta64) / np.timedelta64(1, 'm'), 0)).astype(np.int64)  # Convert timedelta to int in minutes
        return locations, distances
    else:
        click.echo(f"Error computing distance matrix: {result['code']}")
        return None, None

def generate_temporal(distances : NDArray[np.int64], 
                      departing_locations : list[tuple[float, float]],
                      departing_indexes : list[int],
                      patient_locations : list[tuple[float, float]],
                      patient_indexes : list[int],
                      arrival_locations : list[tuple[float, float]],
                      arrival_indexes : list[int],
                      arrival_point_type : str,
                      single : float, independent : float, simultaneous : float, sequential : float, 
                      horizon : int,
                      incompatibility : float,                      
                      service_classes : tuple[int, int],
                      optional_rate : float,
                      double_time_window_rate : float,
                      preferred_caregiver_rate : float,
                      config : dict[str, Any]
                    ) -> tuple[list[TerminalPoint], list[TerminalPoint], list[Caregiver], list[Patient], list[Service]]:
    assert len(departing_locations) + len(patient_locations) + len(arrival_locations) == distances.shape[0] == distances.shape[1], "The number of locations and the distance matrix must match"
    assert len(departing_locations) == len(departing_indexes), "The number of departing locations do not match with their indexes"
    assert len(patient_locations) == len(patient_indexes), "The number of patients locations do not match with their indexes"
    assert len(arrival_locations) == len(arrival_indexes), "The number of arrival locations do not match with their indexes"
    assert math.isclose(single + independent + simultaneous + sequential, 1.0), "The sum of the probabilities must be 1"
    assert 0.0 <= incompatibility <= 1.0, "The incompatibility must be between 0 and 1"
    assert arrival_point_type in ('same_as_departure', 'shuffle_departures', 'different_than_departure'), f'Wrong arrival point type: {arrival_point_type}'
    patients = len(patient_locations)

    service_types : dict[str, list[str]] = { f"t{i}": [] for i in range(len(service_classes)) }
    # Generate services
    click.echo(f"Generating {len(service_classes)} service classes")
    j = 0
    for i, n in enumerate(service_classes):
        for _ in range(n):
            service_types[f"t{i}"].append(f"s{j}")
            j += 1
    # Generating (and randomly shuffling) the type of single/multiple services for all patients
    double_service : list[str | None] = []
    double_service += ['independent'] * int(patients * independent) + ['simultaneous'] * int(patients * simultaneous) + ['sequential'] * int(patients * sequential)
    double_service += [None] * (patients - len(double_service))
    random.shuffle(double_service)
    # Defining durations
    # TODO: currently, time windows, durations and lengths are somewhat fixed
    config_patient = config.get('patient')
    assert config_patient is not None and type(config_patient) is dict, "No patient info in config file"
    time_window_lengths = config_patient.get('time_windows')
    assert time_window_lengths is not None and len(time_window_lengths) > 0, "No time windows sizes specified in config file"
    durations = config_patient.get('treatment_durations')
    assert durations is not None and len(durations) > 0, "No treatment durations specified in config file"    
    duration_biases = config_patient.get('treatment_duration_biases')
    assert duration_biases is not None and len(duration_biases) == len(durations), "No duration biases specified in config file or wrong size (i.e., should be the same as treatment durations)"
    sequential_distances = config_patient.get('sequential_distances')
    assert sequential_distances is not None and len(sequential_distances) > 0, "No distances for sequential double services is specified in config file"
    time_window_granularity = config_patient.get('time_window_granularity')
    if time_window_granularity is None:
        click.secho("No time window granularity provided in config file, assuming 30", fg='yellow')
        time_window_granularity = 30
    # Generate patients
    click.echo(f"Generating {patients} patients")
    used_services : set[str] = set()
    total_services = 0
    optional_patients : list[bool | None] = []
    if optional_rate > 0.0:
        total_optional_patients = int(optional_rate * patients)
        optional_patients += [True] * total_optional_patients + [False] * (patients - total_optional_patients)
    else:
        optional_patients += [None] * patients
    random.shuffle(optional_patients)
    double_time_window_patients = [True] * int(double_time_window_rate * patients) + [False] * (patients - int(double_time_window_rate * patients))
    random.shuffle(double_time_window_patients)
    preferred_caregiver_patients = [True] * int(preferred_caregiver_rate * patients) + [False] * (patients - int(preferred_caregiver_rate * patients))
    random.shuffle(preferred_caregiver_patients)

    generated_patients = []
    for i in range(patients):        
        time_windows : list[TimeWindow] = []         
        types = random.sample([f"t{i}" for i in range(len(service_classes))], k=2)
        assert len(set(types)) == 2, f"Types {types} not valid ({[f't{i}' for i in range(len(service_classes))]})" 
        if double_service[i] is not None:
            service_1, service_2 = random.choice(service_types[types[0]]), random.choice(service_types[types[1]])
            used_services.update([service_1, service_2])
            total_services += 2
        else:
            service = random.choice(sum((st for st in service_types.values()), start=[]))
            used_services.add(service)
            duration = random.choices(durations, weights=duration_biases)[0]
            required_services = [RequiredService(service=service, duration=duration)]
            total_services += 1            
        synchronization = None
        # Construct the service characteristics in the difference cases for double services
        if double_service[i] == 'independent':
            synchronization = Synchronization(type='independent')
            duration_1, duration_2 = random.choices(durations, weights=duration_biases, k=2)
            required_services = [RequiredService(service=service_1, duration=duration_1), RequiredService(service=service_2, duration=duration_2)]
        elif double_service[i] == 'simultaneous':                
            synchronization = Synchronization(type='simultaneous')  
            duration = random.choices(durations, weights=duration_biases)[0]
            required_services = [RequiredService(service=service_1, duration=duration), RequiredService(service=service_2, duration=duration)]            
        elif double_service[i] == 'sequential':
            min_distance, max_distance = sorted(random.choices(sequential_distances, k=2))
            synchronization = Synchronization(type='sequential', distance=Synchronization.SynchronizationDistance(min=min_distance, max=max_distance))
            duration_1, duration_2 = random.choices(durations, weights=duration_biases, k=2)
            required_services = [RequiredService(service=service_1, duration=duration_1), RequiredService(service=service_2, duration=duration_2)]
            
        target_time_windows = 2 if double_time_window_patients[i] else 1
        while len(time_windows) < target_time_windows:
            for _ in range(target_time_windows):
                length = random.choice(time_window_lengths)
                ref_time = random.randint(length, horizon - length) // time_window_granularity * time_window_granularity
                time_window = TimeWindow(start=int(ref_time - length), end=int(ref_time + length))
                if double_service[i] == 'sequential':
                    time_window_length = time_window.end - time_window.start
                    # TODO: currently margins have been somewhat fixed
                    margin = random.choice([15, 30, 45, 60])
                    # Adjust the time window length
                    assert synchronization is not None and synchronization.distance is not None
                    if time_window_length < synchronization.distance.max + margin:
                        to_add = synchronization.distance.max + margin - time_window_length
                        if time_window.end + to_add <= horizon or time_window.start - to_add < 0:
                            time_window.end += to_add
                        else:
                            time_window.start -= to_add
                time_windows.append(time_window)

            time_windows.sort(key=lambda tw: tw.start)
            for j in range(len(time_windows) - 1):
                if TimeWindow.overlap(time_windows[j], time_windows[j + 1]):
                    time_windows.clear()
                    break   

        p = Patient(id=f'p{i}', 
                    time_windows=time_windows, 
                    required_services=required_services, 
                    synchronization=synchronization, 
                    location=patient_locations[i], 
                    distance_matrix_index=patient_indexes[i],
                    optional=optional_patients[i])            
        p._compatible_caregivers = [set() for _ in range(len(p.required_services))]
        generated_patients.append(p)

    # Generate departing points
    departing_points = len(departing_locations)
    arrival_points = len(arrival_indexes)
    click.echo(f"Generating {departing_points} departing points and {arrival_points} arrival points")
    generated_departing_points = []
    generated_arrival_points = []
    for d in range(departing_points):
        t = TerminalPoint(id=f'd{d}', distance_matrix_index=departing_indexes[d], location=departing_locations[d])
        generated_departing_points.append(t)            
        if arrival_point_type != 'different_than_departure':
            generated_arrival_points.append(t)
    if arrival_point_type == 'different_than_departure':
        for d in range(arrival_points):
            t = TerminalPoint(id=f'd{d + len(generated_departing_points)}', distance_matrix_index=arrival_indexes[d], location=arrival_locations[d])
            generated_arrival_points.append(t)            
    
    assert len(generated_departing_points) == departing_points
    if arrival_point_type != 'different_than_departure':
        assert len(generated_arrival_points) == departing_points
    else:
        assert len(generated_arrival_points) == arrival_points

    # Generate caregivers
    config_caregiver = config.get('caregiver')
    assert config_caregiver is not None and type(config_caregiver) is dict, "No caregiver info in config file"
    shift_durations = config_caregiver.get('shift_durations')
    assert shift_durations is not None and len(shift_durations) > 0, "No shift duration information provided in the config file"
    assert max(shift_durations) <= horizon, f"Max shift duration {max(shift_durations)} is over the horizon {horizon}"
    shift_granularity = config_caregiver.get('shift_granularity')
    if shift_granularity is None:
        click.secho("No shift granularity provided in config file, assuming 30", fg='yellow')
        shift_granularity = 30
    generated_caregivers : list[Caregiver] = []    
    lunch_break_threshold = config_caregiver.get('lunch_break', {}).get('threshold')
    if config_caregiver.get('lunch_break') is None:
        click.secho("Lunch break specificaion for caregivers is not present in the file", fg='yellow')
    # TODO: currently the total caregivers range is somewhat fixed
    target_caregivers = random.randint(total_services // 9, total_services // 7)
    click.echo(f"Generating {target_caregivers} caregivers")
    # Cover all used services by at least one caregiver
    while used_services or len(generated_caregivers) < target_caregivers:
        abilities = []
        # First try to cover all skills required by patients
        if used_services:           
            skill = random.choice(list(used_services))
            for st in service_types.keys():
                if skill in service_types[st]:
                    skill_type = st
                    break
        else:
            # Else, choose a random skill
            skill_type = random.choice([f"t{i}" for i in range(len(service_classes))])
            skill = random.choice(service_types[skill_type])
        abilities.append(skill)
        if skill in used_services:
            used_services.remove(skill)
        n_skills = random.randint(1, len(service_types[skill_type]))
        while len(abilities) < n_skills:
            skill = random.choice(service_types[skill_type])
            if skill in used_services:
                used_services.remove(skill)
            if skill not in abilities:
                abilities.append(skill)                            
        # check services
        for s in abilities:
            assert s in service_types[skill_type], f"Service {s} not in skill type {skill_type}"
        departing_point = random.choice(generated_departing_points).id
        arrival_point = departing_point
        if arrival_point_type == 'shuffle_departures':
            arrival_point = random.choice(generated_departing_points).id
        elif arrival_point_type == 'different_than_departure':
            arrival_point = random.choice(generated_arrival_points).id
        caregiver_shift_duration = random.choice(shift_durations)
        caregiver_shift_start = random.randint(0, horizon - caregiver_shift_duration + shift_granularity) // shift_granularity * shift_granularity

        lunch_break = None
        if lunch_break_threshold is not None:
            if caregiver_shift_duration >= lunch_break_threshold:
                lunch_break = True
            else:
                lunch_break = False
        generated_caregivers.append(Caregiver(id=f'c{len(generated_caregivers)+1}', abilities=abilities, departing_point=departing_point, arrival_point=arrival_point, working_shift=TimeWindow(start=caregiver_shift_start, end=caregiver_shift_start + caregiver_shift_duration), lunch_break=lunch_break))
        # add the caregiver to the inner field _compatible_caregivers for suitable patients
        total_matchings = 0
        for p in generated_patients:
            for i, rs in enumerate(p.required_services):
                for skill in abilities:
                    if skill == rs.service:
                        assert p._compatible_caregivers is not None
                        p._compatible_caregivers[i].add(generated_caregivers[-1].id)
                        total_matchings += 1

    click.echo("Setting incompatible matchings")
    target_incompatible_matchings = int(incompatibility * total_matchings)
    potential_patients = generated_patients.copy()
    random.shuffle(potential_patients)
    while target_incompatible_matchings > 0 and potential_patients:
        p = potential_patients[0]
        assert p._compatible_caregivers is not None
        indices = list(range(len(p._compatible_caregivers)))
        random.shuffle(indices)
        for i in indices:
            if len(p._compatible_caregivers[i]) > 1:
                c = random.choice([c for c in p._compatible_caregivers[i]])
                if p.incompatible_caregivers is None:
                    p.incompatible_caregivers = set()
                p.incompatible_caregivers.add(c)
                p._compatible_caregivers[i].remove(c)        
                target_incompatible_matchings -= 1
        potential_patients.pop(0)
        # Possibly consider again the patient by putting at the end of the queue
        if any(len(cg) > 1 for cg in p._compatible_caregivers):
            potential_patients.append(p)          

    # Generate preferred caregivers
    click.echo("Genereting preferred caregivers")     
    assert preferred_caregiver_rate == 0.0 or config_patient.get('max_number_of_preferred_caregivers') is not None, "Preferred caregiver rate specified but no config information about 'max_number_of_preferred_caregivers'"
    for i, p in enumerate(generated_patients):
        if preferred_caregiver_patients[i]:
            _compatible_caregivers : set[str] = set()
            assert p._compatible_caregivers is not None
            for cgs in p._compatible_caregivers:
                _compatible_caregivers.update(cgs)
            p.preferred_caregivers = set(random.sample(list(_compatible_caregivers), random.randint(1, min(len(_compatible_caregivers), config_patient.get('max_number_of_preferred_caregivers')))))

    # TODO: remove caregivers that do not appear in any patient's service
    
    # Generate services, with an estimate of their duration based on the generated patients
    click.echo("Generating services")
    services = []
    for st in service_types.keys():
        for s in service_types[st]:
            s_durations = []
            for p in generated_patients:
                for rs in p.required_services:
                    if s in rs.service:
                        s_durations.append(rs.duration)
            if s_durations:
                services.append(Service(id=s, default_duration=mode(s_durations), type=st))
            else:
                services.append(Service(id=s, default_duration=random.choice(durations), type=st))    
    click.echo("Temporal generation done")
    return generated_departing_points, generated_arrival_points, generated_caregivers, generated_patients, services
    


