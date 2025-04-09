from typing import Optional, Literal, Annotated, Any, Self, Union, List
from pydantic import BaseModel, Field, model_validator, computed_field, AliasChoices
from collections.abc import Sequence
from collections import Counter
import hashlib
import click
import numpy as np
import warnings
import numbers

class TerminalPoint(BaseModel):
    """
    Represents a departure/arrival point for caregivers.
    Attributes:
        id (str): A unique identifier for the terminal point. Must be a non-empty string.
        distance_matrix_index (Optional[int]): An optional index for the distance matrix, must be a non-negative integer.
        location (tuple[float, float]): The geographical location of the terminal point represented as a tuple of latitude and longitude.
    """
    id: Annotated[str, Field(min_length=1, frozen=True)]
    distance_matrix_index: Optional[Annotated[int, Field(ge=0)]] = None
    public_distance_matrix_index: Optional[Annotated[int, Field(ge=0)]] = None
    location: Optional[tuple[float, float]] = None

    def __hash__(self) -> int:
        return id.__hash__()

class TimeWindow(BaseModel):
    """
    Represents a time window with a start and end time.

    Attributes:
        start (int): The start time of the time window. Must be greater than or equal to 0.
        end (int): The end time of the time window. Must be greater than or equal to 0 and must be greater than start.
    """
    start: Annotated[Union[int,float], Field(ge=0)]
    end: Annotated[Union[int,float], Field(ge=0)]

    @classmethod
    def overlap(cls, time_window_1 : 'TimeWindow', time_window_2 : 'TimeWindow') -> bool:
        if time_window_1.start < time_window_2.start:
            first, second = time_window_1, time_window_2
        else:
            first, second = time_window_2, time_window_1
        if second.start < first.end:
            return True
        return False

    @model_validator(mode="before")
    @classmethod
    def validate_tw(cls, values : Any) -> Any:
        # Allow input as a pair of integers (e.g., [start, end])
        if isinstance(values, Sequence) and not isinstance(values, str):
            assert len(values) == 2 and all(isinstance(v, numbers.Number) for v in values), "A time window should consist of two numeric values (e.g., as a tuple or as a list) with valid indexing."
            values = {"start": values[0], "end": values[1]}
        # Ensure start is <= end
        if values['start'] > values['end']:
            raise ValueError(f"Time-window start time ({values['start']}) must be less than or equal to end time ({values['end']}).")
        return values

class Caregiver(BaseModel):
    """
    Represents a caregiver with specific attributes and validation.
    Attributes:
        id (str): Unique identifier for the caregiver. Must be a non-empty string.
        abilities (list[str]): List of abilities the caregiver possesses. Must contain at least one ability.
        departing_point (str): The starting point or departing point identifier for the caregiver. Must be a non-empty string.
        arrival_point (str): The arrival point identifier for the caregiver. Can be empty.
        working_shift (TimeWindow, optional): Time-window representing the working hours of the caregiver. 
    """
    id : Annotated[str, Field(min_length=1, frozen=True)]
    abilities : Annotated[list[str], Field(min_length=0)] = None
    departing_point : Annotated[Optional[str], Field(min_length=1, alias=AliasChoices('starting_point_id', 'departing_point'))]
    arrival_point : Optional[str] = None
    # FIXME: working shift (in the generator) should be already mapped to the suitable type list(map(int, self.working_shift))
    working_shift : Optional[TimeWindow] = None
    lunch_break : Optional[bool] = None
    transportation_mode : Optional[Literal['car','public']] = None
    aspects : Optional[Sequence[Literal['female','male','smoker','dog','cat']]] = None

    @model_validator(mode="before")
    @classmethod
    def _validate_and_normalize(cls, values : Any) -> Any:             
        # Check if the old "starting_point_id" is used instead of "departing_point"
        if "starting_point_id" in values:
            warnings.warn(
                "'starting_point_id' is deprecated, it will still be used but if you're generating new instances, please use 'departing_point' instead.",
                DeprecationWarning
            )  
            values["departing_point"] = values["starting_point_id"]
            del values["starting_point_id"]
        
        if "arrival_point" not in values.keys():
            warnings.warn(
                "'arrival_point' is not present, it will be assumed to be equal to 'departing_point'",
                DeprecationWarning
            )  
            values["arrival_point"] = values["departing_point"]

        return values


class Synchronization(BaseModel):    
    """
    Represents synchronization settings.

    Attributes:
        type (Literal['independent', 'simultaneous', 'sequential']): The type of synchronization.
        distance (Optional[SynchronizationDistance]): The min/max syncrhonization distance for 'sequential' synchronization type.
    """
    class SynchronizationDistance(BaseModel):
        """
        SynchronizationDistance model to represent a range of syncrhonization distance with minimum and maximum values.

        Attributes:
            min (int): The minimum value of the synchronization distance.
            max (int): The maximum value of the synchronization distance.
        """
        min: int
        max: int

        @model_validator(mode="before")
        @classmethod
        def validate_sd(cls, values : Any) -> Any:
            # Allow input as a pair of integers (e.g., [min, max])
            if isinstance(values, Sequence):
                assert len(values) == 2, "A sync distance should comprise two values."            
                assert all(isinstance(v, numbers.Number) for v in values), f"A sync distance should be a pair of two numeric values (e.g., as a tuple or a list), found {type(values[0])}."
                values = {"min": values[0], "max": values[1]}
            # Allow input as a single integer (min=max)
            if isinstance(values, numbers.Number):
                values = {"min": values, "max": values}
            # Ensure min is <= max
            if values['min'] > values['max']:
                raise ValueError(f"Synchronization distance min time ({values['min']}) must be less than or equal to max time ({values['max']}).")
            return values    
    type: Literal['independent', 'simultaneous', 'sequential']
    distance: Optional[SynchronizationDistance] = None


    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        if self.type == 'sequential' and self.distance is None:
            raise ValueError(f"Syncrhonization distance [min, max] should be provided for 'sequential' synchronization type")
        return self

class Service(BaseModel):
    """
    Represents a service with an ID, type, and optional default duration.

    Attributes:
        id (str): The unique identifier for the service. Must be a non-empty string.
        type (str): The type of service. Must be a non-empty string.
        default_duration (Optional[int]): The default duration of the service in minutes. This is optional and can be None.
    """
    id: Annotated[str, Field(min_length=1, frozen=False)]
    type: Annotated[str, Field(min_length=1, frozen=False)] 
    default_duration: Optional[int] = None

    def __hash__(self) -> int:
        return id.__hash__()

    @model_validator(mode="before")
    @classmethod
    def _check_compliances(cls, values : Any) -> Any:             
        if "type" not in values.keys():
            warnings.warn(
                "'type' not present in instance -- this is probably due to the fact you are converting an old instance. Assuming it equal to the 'id' of the service",
                DeprecationWarning
            ) 
            values["type"] = values["id"]
        return values

class RequiredService(BaseModel):
    """
    RequiredService represents a service that is required with a specified duration.

    Attributes:
        service (str): The name of the service. Must be a non-empty string.
        duration (Optional[int]): The duration of the service in minutes. Must be greater than 0 if provided.
        _actual_duration (int): The actual duration of the service in minutes. Must be greater than 0. This attribute is excluded from serialization and not included in the representation, it is used to be set with the default or overwritten by duration.
    """
    service : Annotated[str, Field(min_length=1)]
    duration : Optional[Annotated[int, Field(gt=0)]] = None
    _actual_duration : Optional[Annotated[int, Field(gt=0, exclude=True, repr=False)]] = None

class Patient(BaseModel):
    """
    Represents a patient with specific requirements and attributes.

    Attributes:
        id (str): A unique identifier for the patient. Must be a non-empty string.
        required_services (list[RequiredService]): A list of required services for the patient. 
            Must contain between 1 and 2 services. Aliased as 'required_caregivers' or 'required_services'.
        distance_matrix_index (Optional[int]): An optional index for the distance matrix. Must be a non-negative integer.
        time_window (Optional[TimeWindow]): An optional time window for the service times.
        location (Optional[tuple[float, float]]): An optional geographical location represented as a tuple of latitude and longitude.
        synchronization (Optional[Synchronization]): An optional synchronization specification required if more than one caregiver is needed.
        incompatible_caregivers (set[str]): A set of caregiver IDs that are incompatible with the patient.
    """
    id : Annotated[str, Field(min_length=1, frozen=False)]
    required_services : Annotated[Sequence[RequiredService], 
                                 Field(min_length=1, alias=AliasChoices('required_caregivers', 'required_services'))]
    distance_matrix_index : Optional[Annotated[int, Field(ge=0)]] = None
    public_distance_matrix_index : Optional[Annotated[int, Field(ge=0)]] = None
    time_windows : Optional[Annotated[Sequence[TimeWindow], Field(alias=AliasChoices('time_window', 'time_windows'))]] 
    preferred_start_time : Optional[int] = None
    location : Optional[tuple[float, float]] = None
    synchronization : Optional[Synchronization] = None
    incompatible_caregivers : Optional[set[str]] = None
    preferred_caregivers : Optional[set[str]] = None
    optional : Optional[bool] = None
    aspects : Optional[Sequence[Literal['female','male','smoker','dog','cat']]] = None
    _compatible_caregivers : Optional[Annotated[list[set[str]], Field(min_length=1, exclude=True, repr=False)]] = None
    # FIXME: this should account for the incompatibles

    @model_validator(mode="before")
    @classmethod
    def validate_tw(cls, values : Any) -> Any:
        if 'time_window' in values.keys():
            warnings.warn(
            "'time_window' is deprecated -- substituted with 'time_windows'",
            DeprecationWarning
        ) 
            time_window = values.get('time_window')
            del values['time_window']
            values['time_windows'] = [time_window]
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_caregiver(cls, values : Any) -> Any:
        if "required_caregivers" in values.keys():
            warnings.warn(
            "'required_caregivers' is deprecated -- substituted with 'required_services'",
            DeprecationWarning
        ) 
        return values


    @model_validator(mode='after')
    def _validity_checks(self) -> Self:
        if len(self.required_services) > 1:
            assert self.synchronization is not None, "Synchronization specification is mandatory if more than one caregiver is required"
        # Check caregivers' consistency
        if self._compatible_caregivers and self.incompatible_caregivers:
            _total_compatible_caregivers = set()
            for cg in self._compatible_caregivers:
                _total_compatible_caregivers.update(cg)
            assert _total_compatible_caregivers.isdisjoint(self.incompatible_caregivers), f"Incompatible caregivers and compatible caregivers overlap {{{_total_compatible_caregivers & self.incompatible_caregivers}}}"
        
        return self
    
class LunchBreakSpec(BaseModel):   
    """
    A model representing a specification for a lunch break.

    Attributes:
        start (int): The minimum beginning time of the lunch break. 
        end (int): The maximum ending time of the lunch break. 
        min_duration (int): The minimum required duration for the lunch break in minutes.
    """
    start: int
    end: int
    min_duration: int   

class CostComponents(BaseModel):
    """
    Represents my cost components and weights
    """
    travel_time : Optional[Union[int, float, Literal['HARD']]] = None
    total_tardiness : Optional[Union[int, float, Literal['HARD']]] = None
    highest_tardiness :  Optional[Union[int, float, Literal['HARD']]] = None
    total_waiting_time : Optional[Union[int, float, Literal['HARD']]] = None
    total_extra_time : Optional[Union[int, float, Literal['HARD']]] = None
    max_idle_time :  Optional[Union[int, float, Literal['HARD']]] = None
    caregiver_preferences : Optional[Union[int, float, Literal['HARD']]] = None
    optional_patients : Optional[Union[int, float, Literal['HARD']]] = None
    missed_lunch_break : Optional[Union[int, float, Literal['HARD']]] = None
    qualification : Optional[Union[int, float, Literal['HARD']]] = None
    working_time : Optional[Union[int, float, Literal['HARD']]] = None
    incompabilities : Optional[Union[int, float, Literal['HARD']]] = None
    tw_max_dev_in_time : Optional[Union[int, float, Literal['HARD']]] = None
    tw_max_dev_in_desired_time : Optional[Union[int, float, Literal['HARD']]] = None
    workload_balance : Optional[Union[int, float, Literal['HARD']]] = None
    max_waiting_time : Optional[Union[int, float, Literal['HARD']]] = None

class PatientGeneratorInformation(BaseModel):
    time_windows : Sequence[int]
    treatment_durations : Sequence[int]
    treatment_duration_biases : Sequence[float]
    sequential_distances : Sequence[int]
    time_window_granularity : int

class CaregiverGeneratorInformation(BaseModel):
    shift_durations : Sequence[int]
    shift_granularity : int

class GeneratorInformation(BaseModel):
    city : str
    radius : float
    patients : int 
    departing_points : int
    arrival_points : Optional[int] = None
    arrival_point_type : str
    sync_rates : tuple[float, float, float, float]
    incompatibility : float
    service_classes : tuple[int, int]
    optional_rate : float
    double_time_window_rate : float
    preferred_caregiver_rate : float
    no_intersect_administrative : bool
    # from config
    patient_config : PatientGeneratorInformation
    caregiver_config : CaregiverGeneratorInformation
    random_seed : int
    # lunch break infos are omitted since this infos are already in the instance

class MetaData(BaseModel):
    """
    Represents the metadata of my instance.
    """
    time_window_met : Literal['at_service_start', 'at_service_end'] = 'at_service_start'
    # FIXME: this should be set from the generator and optionality should be removed
    cost_components : Optional[CostComponents] = None
    # FIXME: round area bounds outside the function tuple(np.around(area, decimals=4))
    area : Optional[tuple[float, float, float, float]] = None
    name : Optional[Annotated[str, Field(min_length=1, frozen=False)]] = None
    origin : Literal['generated', 'bazirha', 'bazirha-caie', 'oldies', 'kummer','extended', 'Italian', 'urli', 'mankowska', 'raidl'] = 'generated'
    # FIXME: (in the generator) compute times outside the function [[int(d.total_seconds() // 60) for d in r] for r in distances]
    horizon: Annotated[Optional[int], Field(gt=0)] = None
    generator_info : Optional[GeneratorInformation] = None
    time_window_service_level : Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _check_compliances(cls, values : Any) -> Any:             
        if "time_window_service_level" in values.keys() and values['origin'] not in ["raidl"]:
            raise ValueError("time_window_service_level is linked with 'raidl' instances")
        else:
            if values['origin'] == 'raidl':
                if "time_window_service_level" in values.keys():
                    if not values["time_window_service_level"]:
                        raise ValueError("time_window_service_level is linked with 'raidl' instances")
                else:
                    raise ValueError("time_window_service_level is linked with 'raidl' instances")
        return values
  

    @model_validator(mode="after")
    def _validate_generator_info(self) -> Self:
        if self.origin == "generated":
            if self.generator_info is None:
                raise ValueError("generator_info must be provided when origin is 'generated'")
        else:
            if self.generator_info is not None:
                raise ValueError("generator_info must not be provided when origin is not 'generated'")
        return self
        
def calculate_silver_metrics(ls : list) -> dict:
    dc = {
        "min":np.min(ls), 
        "max":np.max(ls),
        "range": np.max(ls)-np.min(ls) , 
        "mean":np.mean(ls), 
        "median":np.median(ls), 
        "25_perc":np.percentile(ls, 25), 
        "75_perc":np.percentile(ls, 75), 
        "std":np.std(ls)
    }
    for k in dc.keys():
        if isinstance(dc[k], np.integer):
            dc[k] = int(dc[k])
        elif isinstance(dc[k], np.floating):
            dc[k] = float(dc[k])
    return dc
    
class Instance(BaseModel):
    """
    Instance model representing a healthcare scheduling scenario.
    
    Attributes:
        name (str): The name of the instance.
        area (Optional[Sequence[float]]): The area bounds, rounded to 4 decimal places.
        distances (Sequence[Sequence[int]]): A matrix of distances between points.
        departing_points (set[DepartingPoint]): A list of departing points.
        arrival_points (Optional[set[TerminalPoint]]): A list of arrival points.
        caregivers (Sequence[Caregiver]): A list of caregivers.
        patients (Sequence[Patient]): A list of patients.
        services (Sequence[Service]): A list of services.
        _terminal_points (dict[str, DepartingPoint]): Internal dictionary for quick access to terminal points.
        _caregivers (dict[str, Caregiver]): Internal dictionary for quick access to caregivers.
        _patients (dict[str, Patient]): Internal dictionary for quick access to patients.
        _services (dict[str, Service]): Internal dictionary for quick access to services.
    
    Methods:
        _check_validity() -> Self:
            Validates the instance data after model creation.
        signature() -> str:
            Computes a unique signature for the instance based on its fields.
        features() -> dict[str, Any]:
            Computes and returns various features of the instance, such as the number of patients, caregivers, services, and statistics on compatible caregivers, time windows, service lengths, and distances.
    """
    metadata : MetaData
    distances : Sequence[Sequence[Union[int, float]]]
    public_distances : Optional[Sequence[Sequence[Union[int, float]]]] = None
    terminal_points : Annotated[List[TerminalPoint], Field(alias=AliasChoices('departing_points','central_offices','terminal_points'))]
    # arrival_points : Optional[set[TerminalPoint]] = set() # FIXME: it should be for all
    caregivers : Sequence[Caregiver]
    patients : Sequence[Patient]
    services : Sequence[Service]
    services_order : Optional[Sequence[str]] = None
    # time_window_met: Literal['at_service_start', 'at_service_end'] = 'at_service_start'
    lunch_breaks : Optional[LunchBreakSpec] = None
    _terminal_points : Optional[Annotated[dict[str, TerminalPoint], Field(exclude=True, repr=False)]] = None
    # _arrival_points : Optional[Annotated[dict[str, TerminalPoint], Field(exclude=True, repr=False)]] = None
    _caregivers : Optional[Annotated[dict[str, Caregiver], Field(exclude=True, repr=False)]] = None
    _patients : Optional[Annotated[dict[str, Patient], Field(exclude=True, repr=False)]] = None
    _services : Optional[Annotated[dict[str, Service], Field(exclude=True, repr=False)]] = None
    
    @model_validator(mode="before")
    @classmethod
    def _check_compliances(cls, values : Any) -> Any:             
        # Check the datastructure called metadata
        if "metadata" in values.keys():
            return values
        
        warnings.warn(
            "'metadata' not present in instance -- this is probably due to the fact you are converting an old instance",
            DeprecationWarning
        ) 
        
        name = values.get('name') or "not-given"
        if "name" in values.keys():
            del values['name']
        origin = "oldies" 
        area = values.get('area')
        metadata = MetaData(
            name=name,
            origin=origin,
            area=area,
        )
        values['metadata'] = metadata

        # FIXME: if type raidl, then you should have the order of the services
        
        if "central_offices" in values.keys():
            warnings.warn(
                "'central_offices' is deprecated -- this will be called 'terminal_points'",
                DeprecationWarning
            ) 
            values["terminal_points"] = values.get('central_offices') 
            del values['central_offices']
        
        if "departing_points" in values.keys():
            warnings.warn(
                "'central_offices' is deprecated -- this will be called 'terminal_points'",
                DeprecationWarning
            ) 
            values["terminal_points"] = values.get('departing_points') 
            del values['departing_points']
        
        c_i=0
        for c in values["caregivers"]:
            if "starting_point_id" not in c or 'departing_point' not in c:
                values["caregivers"][c_i]["departing_point"] = values["terminal_points"][0]["id"]
            c_i+=1
         
        return values

    @model_validator(mode="before")
    @classmethod
    def order_entities(cls, values : Any) -> Any:
        patients = values["patients"]
        ordered_patients = sorted(patients, key=lambda x: x.get("distance_matrix_index"))
        values["patients"] = ordered_patients

        terminals = values["terminal_points"]
        ordered_terminal = sorted(terminals, key=lambda x: x.get("distance_matrix_index"))
        seen = set()
        values["terminal_points"] = [tp for tp in ordered_terminal if (tp["id"] not in seen and not seen.add(tp["id"]))]
        # for i, route in enumerate(values["routes"]):
        #     if "patient" not in route.keys():
        #         continue
        #     patients = route["locations"]
        #     ordered = sorted(patients, key=lambda x: x.get("arrival_time", x.get("start_time", x.get("start_service_time"))))
        #     values["routes"][i]["locations"] = ordered
        #     # print(i,r["locations"])

        #route["locations"] = sorted(route["locations"], key=lambda x: x["arrival_time"])
        return values  

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        # Populate redundant data structure
        for p in range(len(self.patients)):
            if self.patients[p]._compatible_caregivers is None:
                # warnings.warn(
                #     "Populating fields in patients",
                #     RuntimeWarning 
                # )
                self.patients[p]._compatible_caregivers = [set() for _ in range(len(self.patients[p].required_services))]
                for c in range(len(self.caregivers)):
                    for i, rs in enumerate(self.patients[p].required_services):
                        for skill in self.caregivers[c].abilities:
                            if skill == rs.service:
                                self.patients[p]._compatible_caregivers[i].add(self.caregivers[c].id)
            if self.patients[p].incompatible_caregivers:
                for cg in  self.patients[p].incompatible_caregivers:
                    for i, rs in enumerate(self.patients[p].required_services):
                        if cg in self.patients[p]._compatible_caregivers[i]:
                            self.patients[p]._compatible_caregivers[i].remove(cg)
        # Matrix Size
        expected_matrix_size = len(self.terminal_points) + len(self.patients)
        assert len(self.distances) == expected_matrix_size, f"The distance matrix is supposed to have {expected_matrix_size} rows ({len(self.terminal_points)} terminal points + {len(self.patients)} patients)"
        for i, r in enumerate(self.distances):
            assert(len(r)) == expected_matrix_size, f"Row {i} of the distance matrix is supposed to have {expected_matrix_size} columns ({len(self.terminal_points)} terminal points + {len(self.patients)} patients)"
        # check terminal point indexes (i.e., old versions of the generator might not have them)
        for i, dp in enumerate(self.terminal_points):
            if dp.distance_matrix_index is None:
                if len(self.terminal_points) == 1:
                    dp.distance_matrix_index = i
                else:
                    extract_identifier = int(dp.id.replace("d",""))
                    dp.distance_matrix_index = extract_identifier
        # check patients indexes (i.e., old versions of the generator might not have them)
        for i, p in enumerate(self.patients):
            if p.distance_matrix_index is None:
                p.distance_matrix_index = i + len(self.terminal_points)
        # Foreign keys (caregivers to services and departing points)
        services_id = set(s.id for s in self.services)
        services = { s.id: s for s in self.services }
        caregivers_service_coverage = set()
        matrix_indexes = set()
        terminal_points = set(dp.id for dp in self.terminal_points or [])
        for c in self.caregivers:
            provided_services = set(c.abilities)
            assert provided_services <= set(services_id), f"Abilities of caregiver {c.id} ({provided_services - set(services_id)}) are not included in services"
            caregivers_service_coverage |= provided_services
            assert c.departing_point in terminal_points, f"Departing point {c.departing_point} of caregiver {c.id} is not present in the list of terminal points"
            if c.arrival_point is not None: # TODO: all of the caregiver should have the field returning point
                assert c.arrival_point in terminal_points, f"Arrival point {c.arrival_point} of caregiver {c.id} is not present in the list of terminal points"
        for d in self.terminal_points:
            assert d.distance_matrix_index not in matrix_indexes, f"Matrix index of departing point {d.id} is already present as a matrix index"
            matrix_indexes.add(d.distance_matrix_index)
        # Foreign keys (patients to required services), also setting 
        patients_service_requirement = set()
        for p in self.patients:
            required_services = set(s.service for s in p.required_services)
            assert required_services <= set(services_id), f"Services required by patient {p.id} ({required_services - set(services_id)}) are not included in services"
            patients_service_requirement |= required_services
            assert p.distance_matrix_index not in matrix_indexes, f"Matrix index of patient {p.id} is already present as a matrix index"
            matrix_indexes.add(p.distance_matrix_index)        
            # setting default durations if not given
            for s in p.required_services:
                s._actual_duration = s.duration or services[s.service].default_duration                
            service_types = set(services[rs].type for rs in required_services)
            if self.metadata.origin != 'raidl':
                assert len(service_types) == len(required_services), f"Patient {p.id} is requiring services {required_services} of the same types {service_types}"
            elif len(service_types) != len(required_services):
                click.secho(f"In an instance of origin '{self.metadata.origin}' patient {p.id} is requiring services {required_services} of the same types {service_types}", fg="yellow")
            # Checking incompatible caregivers and getting rid of those which are not providing the required services
            if p.incompatible_caregivers:
                possible_caregivers = set(c.id for c in self.caregivers if set(c.abilities) & required_services)
                # get rid of non meaningful caregivers
                if p.incompatible_caregivers & possible_caregivers != p.incompatible_caregivers:
                    click.secho(f'Patient {p.id} has a set of incompatible caregivers which includes also caregivers not directly involved with the required services, normalizing it', fg='yellow', err=True)
                    p.incompatible_caregivers = p.incompatible_caregivers & possible_caregivers
                for s in p.required_services:
                    possible_caregivers_for_service = set(c.id for c in self.caregivers if s.service in c.abilities and c.id not in p.incompatible_caregivers)
                    assert possible_caregivers_for_service, f"Patient {p.id} has no compatible caregiver for service {s.service} (possibly because of incompatibilities or the no caregiver exists for the service)"
                
        assert patients_service_requirement <= caregivers_service_coverage, f"Some services required by patients are not provided by any caregiver ({patients_service_requirement - caregivers_service_coverage})"        
        assert matrix_indexes == set(range(expected_matrix_size)), f"Some patients / departing point have been wrongly assigned their matrix index"
        # checking that the time window is compatible with the service time distance in case of sequential services
        for p in self.patients:
            for s in p.required_services:
                if p.synchronization and p.synchronization.type == 'sequential':
                    for tw in p.time_windows:
                        assert tw.end - tw.start >= p.synchronization.distance.min, f"Patient {p.id} has a time window too short for the synchronization services required"
        
        # fill in the dictionaries for quicker access
        self._terminal_points = { dp.id: dp for dp in self.terminal_points }
        self._caregivers = { c.id: c for c in self.caregivers }
        self._patients = { p.id: p for p in self.patients }        
        self._services = { s.id: s for s in self.services }
       
        return self

    @property
    @computed_field
    def signature(self) -> str:
        res = b''
        for f in self.model_fields:
            res += bytes(f"{getattr(self, f)}", 'utf-8')
        return hashlib.sha256(res).hexdigest()
    
    @property
    def features(self) -> dict[str, Any]:
        features : dict[str, Any]= {}

        internal_horizon = self.metadata.horizon if self.metadata.horizon is not None else 600

        features["instance_name"] = self.metadata.name
        features["instance_origin"] = self.metadata.origin
        # Patients related features
        features['patients'] = { 
            'total': len(self.patients), 
            'single_service': sum(len(p.required_services) == 1 for p in self.patients), 
            'double_service': sum(len(p.required_services) == 2 for p in self.patients), 
            'more_services': sum(len(p.required_services) > 2 for p in self.patients), 
            'simultaneous': sum(p.synchronization is not None and p.synchronization.type == 'simultaneous' for p in self.patients), 
            'sequential': sum(p.synchronization is not None and p.synchronization.type == 'sequential' for p in self.patients),
            'independent': sum(p.synchronization is not None and p.synchronization.type == 'independent' for p in self.patients),
            'optional': sum(p.optional is not None and p.optional for p in self.patients), # number of optional patients
            'with_preferred_caregivers': sum(p.preferred_caregivers is not None for p in self.patients), # number of pateints with preferred caregivers
            'with_incompatible_caregivers': sum(p.incompatible_caregivers is not None for p in self.patients), # number of pateints with incompatible caregivers
            'with_aspects': sum(p.aspects is not None for p in self.patients), # number of patients with aspects
        }

        patient_number_of_services = []
        service_length = []
        patient_number_tw = []
        tw_length = []
        total_tw_length = []
        patient_number_preferred_cg = []
        preferred_caregivers = []
        patient_number_incompatible_cg = []
        incompatible_caregivers = []
        patient_number_aspects = []
        service_occurencies = []
        for p in self.patients:
            patient_number_of_services.append(len(p.required_services))
            service_length += [s._actual_duration for s in p.required_services]

            for s in p.required_services:
                service_occurencies.append(s.service)

            patient_number_tw.append(len(p.time_windows))
            tw_tot = 0
            for tw in p.time_windows:
                tw_length.append(tw.end - tw.start)
                tw_tot += tw.end-tw.start
            total_tw_length.append(tw_tot)

            patient_number_preferred_cg.append(len(p.preferred_caregivers) if p.preferred_caregivers is not None else 0)
            if p.preferred_caregivers is not None:
                for cg in p.preferred_caregivers:
                    preferred_caregivers.append(cg)
            
            patient_number_incompatible_cg.append(len(p.incompatible_caregivers) if p.incompatible_caregivers is not None else 0)
            if p.incompatible_caregivers is not None:
                for cg in p.incompatible_caregivers:
                    incompatible_caregivers.append(cg)
            
            patient_number_aspects.append(len(p.aspects) if p.aspects is not None else 0)
            
        # occurrencies of the services
        counter_patient_required_services = dict(Counter(service_occurencies))
        for s in self.services:
            if s.id not in counter_patient_required_services:
                counter_patient_required_services[s.id] = 0
        features["occurencies_required_service"] = calculate_silver_metrics(list(counter_patient_required_services.values()))

        # metrics on the number of services per patients
        features['number_service_per_patient'] = calculate_silver_metrics(patient_number_of_services)
        # total services required
        features['total_required_services'] = sum(patient_number_of_services)
        # metrics on the length of the services
        features['service_length'] = calculate_silver_metrics(service_length)
        
        # number of time windows per patient
        features['number_tw_per_patient'] = calculate_silver_metrics(patient_number_tw)
        features["total_number_tw"] = sum(patient_number_tw)
        # metrics on the length of the tw
        features['tw_length'] = calculate_silver_metrics(tw_length)
        features["tw_length"]["total"] = sum(tw_length)

        features["availability_per_patient"] = calculate_silver_metrics(total_tw_length)
        
        # metrics on the number of preferred caregivers
        features["number_preferred_caregiver"] = calculate_silver_metrics(patient_number_preferred_cg)
        # metrics on the preferred_caregiver occurencies
        counter_preferred_caregiver = dict(Counter(preferred_caregivers))
        for cg in self.caregivers:
            if cg.id not in counter_preferred_caregiver:
                counter_preferred_caregiver[cg.id] = 0
        features["occurencies_preferred_caregiver"] = calculate_silver_metrics(list(counter_preferred_caregiver.values()))

        # metrics on the number of incompatible_caregivers
        features["number_incompatible_caregiver"] = calculate_silver_metrics(patient_number_incompatible_cg)
        # metrics on the incomatible_caregiver occurencies
        counter_incompatible_caregiver = dict(Counter(incompatible_caregivers))
        for cg in self.caregivers:
            if cg.id not in incompatible_caregivers:
                counter_incompatible_caregiver[cg.id] = 0
        features["occurencies_incompatible_caregiver"] = calculate_silver_metrics(list(counter_incompatible_caregiver.values()))

        # metrics on the number of aspects per patient
        features["number_aspects_per_patient"] = calculate_silver_metrics(patient_number_aspects)
        # TODO: also the occurencies of the aspects?

        # TODO: preferred start time?

        # Features related to the caregivers
        features['caregivers'] = {
            'total': len(self.caregivers),
            'with_lunch': sum(cg.lunch_break is not None and cg.lunch_break is True for cg in self.caregivers),
            'with_aspects': sum(cg.aspects is not None for cg in self.caregivers),
        }
        
        number_services_per_cg = []
        service_caregivers = []
        cg_number_aspects = []
        duration_shifts_per_cg = []
        departing_point_occurrencies =[]
        arrival_points_occurrencies = []
        reference_points_occurrencies = []
        for cg in self.caregivers:
            number_services_per_cg.append(len(cg.abilities) if cg.abilities is not None else 0)
            if cg.abilities is not None:
                for s in cg.abilities:
                    service_caregivers.append(s)
            
            cg_number_aspects.append(len(cg.aspects) if cg. aspects is not None else 0)
            
            cg_ws = 0
            if cg.working_shift is not None:
                cg_ws = cg.working_shift.end - cg.working_shift.start
            elif self.metadata.horizon is not None:
                # click.secho(f"No working shift found nor horizon, assuming the working shift for caregiver {cg.id} equal to the horizon", fg="yellow")
                cg_ws = internal_horizon
            else:
                # assume a ten hours working shif
                # click.secho(f"No working shift found nor horizon, assuming a 10 hours (600 minutes) working shift for caregiver {cg.id}", fg="yellow")
                cg_ws = internal_horizon
            duration_shifts_per_cg.append(cg_ws)

            departing_point_occurrencies.append(cg.departing_point)
            arrival_points_occurrencies.append(cg.arrival_point)
            reference_points_occurrencies.append(cg.departing_point)
            reference_points_occurrencies.append(cg.arrival_point)
        
        # number of services per careviger
        features["number_service_per_caregiver"] = calculate_silver_metrics(number_services_per_cg)
        # occurences of services looking at the caregiver
        counter_service_caregivers = dict(Counter(service_caregivers))
        for s in self.services:
            if s not in counter_service_caregivers:
                counter_service_caregivers[s] = 0
        features["occurencies_service_caregiver"] = calculate_silver_metrics(list(counter_service_caregivers.values()))

        # number of aspects per caregiver
        features["number_aspects_per_caregiver"] = calculate_silver_metrics(cg_number_aspects)
        # TODO: occurancies aspects?
        
        # shifts duration
        features["caregiver_shifts"] = calculate_silver_metrics(duration_shifts_per_cg)
        features["caregiver_shifts"]["total"] = sum(duration_shifts_per_cg)

        # occurencies of departing point and arrival point in caregivers
        counter_cg_departing_points = dict(Counter(departing_point_occurrencies))
        counter_cg_arrival_points = dict(Counter(arrival_points_occurrencies))
        counter_cg_reference_points = dict(Counter(reference_points_occurrencies))
        for dp_p in self.terminal_points:
            dp = dp_p.id
            if dp not in counter_cg_departing_points:
                counter_cg_departing_points[dp] = 0
            if dp not in counter_cg_reference_points:
                counter_cg_reference_points[dp] = 0
        for ap_p in self.terminal_points:
            ap = ap_p.id
            if  ap not in counter_cg_arrival_points:
                counter_cg_arrival_points[ap] = 0
            if ap not in counter_cg_reference_points:
                counter_cg_reference_points[ap] = 0
        features["occurrencies_reference_points"] = calculate_silver_metrics(list(counter_cg_reference_points.values()))
        features["occurrencies_arrival_points"] = calculate_silver_metrics(list(counter_cg_arrival_points.values()))
        features["occurrencies_departing_points"] = calculate_silver_metrics(list(counter_cg_departing_points.values()))

        # lunch break spectification
        features["lunch_break"] = {
            "start": self.lunch_breaks.start if self.lunch_breaks is not None else 0,
            "end": self.lunch_breaks.end if self.lunch_breaks is not None else 0,
            "length": (self.lunch_breaks.end - self.lunch_breaks.start) if self.lunch_breaks is not None else 0,
            "min_duration":  self.lunch_breaks.min_duration if self.lunch_breaks is not None else 0
        }

        features['services'] = len(self.services)
        features['terminal_points'] = len(self.terminal_points)
        
        compatible_cg = []
        for p in self.patients:
            for s in p.required_services:
                cg_per_s = []
                for cg in self.caregivers:
                    if s.service in cg.abilities: 
                        if p.incompatible_caregivers is None:
                            cg_per_s.append(cg)
                        elif cg.id not in p.incompatible_caregivers and s.service in cg.abilities:
                            cg_per_s.append(cg)
                compatible_cg.append(len(cg_per_s))
        features["compatible_caregivers_per_required_service"] = calculate_silver_metrics(compatible_cg)
        
        # metriche sulle tw
        shift_patient_overlap_patient_wise = []
        shift_tw_overlap = []
        shift_patient_overlap_tw_number = []
        for p in self.patients:
            p_overlap = 0
            p_overlap_number = 0
            inc_cg = p.incompatible_caregivers if p.incompatible_caregivers is not None else []
            p_services = []
            for s in p.required_services:
                p_services.append(s.service)
            for cg in self.caregivers:
                cg_workin_shift_start = cg.working_shift.start if cg.working_shift is not None else 0
                cg_workin_shift_end = cg.working_shift.end if cg.working_shift is not None else internal_horizon
                if len(list(set(p_services) & set(cg.abilities))) < 1 or cg.id in inc_cg:
                    continue
                for tw in p.time_windows:
                    overlap = 0
                    # print( cg_workin_shift_end)
                    if max(tw.start, cg_workin_shift_start) <= min(tw.end, cg_workin_shift_end):
                        overlap += min(tw.end, cg_workin_shift_end) - max(tw.start, cg_workin_shift_start)
                    p_overlap += overlap
                    shift_tw_overlap.append(overlap)
                    if overlap > 0:
                        p_overlap += 1
            shift_patient_overlap_patient_wise.append(p_overlap)
            shift_patient_overlap_tw_number.append(p_overlap_number)
        features["overlap_tw_shift"] = calculate_silver_metrics(shift_tw_overlap)
        features['overlap_tw_shift']["total"] = sum(shift_tw_overlap)
        features["overlap_tw_shift_patient"] = calculate_silver_metrics(shift_patient_overlap_patient_wise)
        features["overlap_tw_shift_patient"]["total"] = sum(shift_patient_overlap_patient_wise)
        features["overlap_tw_shift_patient_number"] = calculate_silver_metrics(shift_patient_overlap_tw_number)  
        features["overlap_tw_shift_patient_number"]["total"] = sum(shift_patient_overlap_tw_number)  


        not_fit_patient = []
        fit_patient = []
        not_fit_service = []
        fit_service = []
        total_not_fit = 0
        total_fit = 0
        for p in self.patients:
            inc_cg = p.incompatible_caregivers if p.incompatible_caregivers is not None else []
            p_fit = 0
            p_non_fit = 0
            for s in p.required_services:
                s_fit = 0
                s_no_fit = 0
                for cg in self.caregivers:
                    cg_workin_shift_start = cg.working_shift.start if cg.working_shift is not None else 0
                    cg_workin_shift_end = cg.working_shift.end if cg.working_shift is not None else internal_horizon
                    if s.service not in cg.abilities or cg.id in inc_cg:
                        continue 
                    for tw in p.time_windows:
                        effective_start = max(tw.start, cg_workin_shift_start)
                        effective_end = min(tw.end, cg_workin_shift_end)
                        # does the time window overlap with the shift?
                        if effective_start > effective_end:
                            continue
                        # in how many minutes can I start the service?
                        overlap = effective_end - effective_start
                        if overlap < s.duration:
                            total_not_fit += 1
                            p_non_fit +=1 
                            s_no_fit += 1
                        else:
                            total_fit += 1
                            p_fit += 1
                            s_fit += 1   
                not_fit_service.append(s_no_fit)
                fit_service.append(s_fit)
            not_fit_patient.append(p_non_fit)
            fit_patient.append(p_fit)

        features["total_fit"] = total_fit
        features["total_non_fit"] = total_not_fit
        features["service_fit"] = calculate_silver_metrics(fit_service)
        features["service_non_fit"] = calculate_silver_metrics(not_fit_service)
        features["patient_fit"] = calculate_silver_metrics(fit_patient)
        features["patient_non_fit"] = calculate_silver_metrics(not_fit_patient)

        features["time_window_met_end_positive"] = 1 if self.metadata.time_window_met == "at_service_end" else 0
        features["time_window_met_end_start"] = 0 if self.metadata.time_window_met == "at_service_end" else 1

        # for every service, if I consider that tw and shift, how late am I going to be?
        total_useful_time_to_start = 0
        useful_time_to_start_per_patient = []
        useful_time_to_start_per_service = []
        for p in self.patients:
            inc_cg = p.incompatible_caregivers if p.incompatible_caregivers is not None else [] 
            total_useful_time_to_start_patient = 0
            for s in p.required_services:
                total_useful_time_to_start_service = 0
                for cg in self.caregivers:
                    cg_workin_shift_start = cg.working_shift.start if cg.working_shift is not None else 0
                    cg_workin_shift_end = cg.working_shift.end if cg.working_shift is not None else internal_horizon
                    if s.service not in cg.abilities or cg.id in inc_cg:
                        continue 
                    for tw in p.time_windows:
                        effective_start = max(tw.start, cg_workin_shift_start)
                        effective_end = min(tw.end, cg_workin_shift_end)
                        # does the time window overlap with the shift?
                        if effective_start > effective_end:
                            continue
                        # if they overlap, then consider, how many minutes you can schedule it before tardiness occur
                        if self.metadata.time_window_met == "at_service_start":
                            total_useful_time_to_start += max(0, effective_end-effective_start)
                            total_useful_time_to_start_patient += max(0, effective_end-effective_start)
                            total_useful_time_to_start_service += max(0, effective_end-effective_start)
                        else: # "at_service_end" 
                            possible_start = effective_end - s.duration
                            total_useful_time_to_start += max(0, possible_start-effective_start)
                            total_useful_time_to_start_patient += max(0, effective_end-effective_start)
                            total_useful_time_to_start_service += max(0, effective_end-effective_start)
                useful_time_to_start_per_service.append(total_useful_time_to_start_service)
            useful_time_to_start_per_patient.append(total_useful_time_to_start_patient)
        features["useful_time_to_start_total"] = total_useful_time_to_start
        features["useful_time_to_start_per_patient"] = calculate_silver_metrics(useful_time_to_start_per_patient)
        features["useful_time_to_start_per_service"] = calculate_silver_metrics(useful_time_to_start_per_service)
        
        # Memo: working of the tardiness
        # if instance.metadata.time_window_met == "at_service_start":
        #     tardiness.append(max(0, l.start_service_time - p.time_windows[idx].end))
        # else:
        #     tardiness.append(max(0, l.end_service_time - p.time_windows[idx].end))
        
        
        distances = np.array(self.distances)
        mask = np.ones(distances.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        features["distances"] = calculate_silver_metrics(list(distances[mask]))
        average = np.mean(list(distances[mask]))
        above_avg = 0
        below_avg = 0
        for edge in list(distances[mask]):
            if edge > average:
                above_avg += 1
            elif edge < average:
                below_avg += 1
        features["number_edges_above_mean"] = above_avg
        features["number_edges_below_mean"] = below_avg
        sorted_distances = np.sort(distances[mask])
        sum_25_lower_edges =  np.sum(sorted_distances[:int(0.25*len(sorted_distances))])
        sum_25_lower_edges = int(sum_25_lower_edges) if isinstance(sum_25_lower_edges,np.integer) else float(sum_25_lower_edges)
        sum_25_bigger_edges =  np.sum(sorted_distances[-int(0.25*len(sorted_distances)):])
        sum_25_bigger_edges = int(sum_25_bigger_edges) if isinstance(sum_25_bigger_edges,np.integer) else float(sum_25_bigger_edges)  
        features["sum_25_lower_edges"] = sum_25_lower_edges
        features["sum_25_bigger_edges"] = sum_25_bigger_edges
        # FIXME: values of features identified for the TSP by K. Smith-Miles?

        return features
