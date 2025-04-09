from typing import Optional, Annotated, Any, Self
from pydantic import BaseModel, Field, model_validator, AliasChoices
from . instance_models import Instance, CostComponents
import math
import bisect
import click

class PatientVisit(BaseModel):
    """
    Represents a patient's visit with service times and related information.

    Attributes:
        start_service_time (int | float): The time when the service starts. Must be greater than or equal to 0.
        end_service_time (int | float): The time when the service ends. Must be greater than or equal to 0.
        patient (str): The name or identifier of the patient.
        service (str): The type of service being provided.
        arrival_at_patient (int | float, optional): The time of arrival at the patient's location. Must be less than or equal to the start service time if provided.

    Methods:
        _check_validity() -> Self:
            Validates that the start service time is less than the end service time and, if provided, that the arrival time is less than or equal to the start service time.
    """
    start_service_time: Annotated[int | float, Field(ge=0, alias=AliasChoices('start_service_time', 'arrival_time','start_time'))]
    end_service_time: Annotated[int | float, Field(ge=0, alias=AliasChoices('end_service_time', 'departure_time','end_time'))]
    patient: str
    service: str
    arrival_at_patient: Optional[int | float] = None

    #TODO: add warning for older service names (i.e., arrival/departure)

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        assert self.start_service_time < self.end_service_time, f"Start service time for service {self.service} at patient {self.patient} is greater than end service time"
        if self.arrival_at_patient is not None:
            assert self.arrival_at_patient <= self.start_service_time, f"Arrival at patient {self.patient} for service {self.service} is greater than start service time"
        return self
    
class DepotDeparture(BaseModel):
    """
    Represents a departure event from a depot.

    Attributes:
        departing_time (Annotated[int | float]): The time of departure from the depot. Must be a non-negative value.
        depot (str): The identifier or name of the depot.
    """
    departing_time: Annotated[int | float, Field(ge=0)]
    depot: str    

class DepotArrival(BaseModel):
    """
    Represents the arrival information at a depot.

    Attributes:
        arrival_time (int | float): The time of arrival at the depot. Must be greater than or equal to 0.
        depot (str): The name or identifier of the depot.
    """
    arrival_time: Annotated[int | float, Field(ge=0)]
    depot: str   

class CaregiverRoute(BaseModel):
    """
    Represents the route of a caregiver.

    Attributes:
        caregiver_id (str): The ID of the caregiver.
        locations (list[DepotDeparture | PatientVisit | DepotArrival]): List of locations in the caregiver's route.
        _visits (list[PatientVisit]): List of patient visits, excluded from serialization and representation.
        _full_route (bool): Indicates if the route is a full route (starting at a depot and ending at a depot), excluded from serialization and representation.

    Methods:
        _check_vallidity(): Validates the caregiver's route after model initialization. Ensures the route starts and ends at depots if it's a full route, checks the consistency of service times, and verifies the types of locations.
    """
    caregiver_id: str
    locations: Optional[list[DepotDeparture | PatientVisit | DepotArrival]] = None
    _visits: Annotated[list[PatientVisit], Field(exclude=True, repr=False)] = []
    _full_route: Annotated[bool, Field(exclude=True, repr=False)] = False

    @model_validator(mode='after')
    def _check_validity(self) -> Self:
        if self.locations:
            self._full_route = False
            if type(self.locations[0]) == DepotDeparture:
                assert type(self.locations[-1]) == DepotArrival, "First location is a depot but last location is not"
                self._full_route = True
            if self._full_route:                
                assert len(self.locations) > 2, f"Caregiver {self.caregiver_id} has a full route with no intermediate location"            
                self._visits = [self.locations[i] for i in range(1, len(self.locations) - 1)]
            else:
                self._visits = [l for l in self.locations]
            prev_time = 0 if not self._full_route else self.locations[0].departing_time
            for i, l in enumerate(self._visits):                
                assert type(l) == PatientVisit, f"Location {l} of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) is not a patient location"
                assert l.start_service_time >= prev_time, f"Start service time of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) {l.start_service_time} is not consistent with the previous one ({prev_time})"
                assert l.end_service_time  > l.start_service_time, f"End service time of caregiver {self.caregiver_id} (at step {i + 1 if self._full_route else i}) {l.end_service_time} is not greater than start service time {l.start_service_time}"
                prev_time = l.end_service_time
            if self._full_route:
                assert self.locations[-1].arrival_time >= prev_time, f"Arrival time at depot of caregiver {self.caregiver_id} {self.locations[-1].arrival_time} is not consistent with the previous end service time {prev_time}"
        return self    

class SolutionCostComponents(BaseModel):
    """
    Represents my cost components and weights
    """
    travel_time : Optional[int] = None
    total_tardiness : Optional[int] = None
    highest_tardiness :  Optional[int] = None
    total_waiting_time : Optional[int] = None
    total_extra_time : Optional[int] = None
    max_idle_time :  Optional[int] = None
    caregiver_preferences : Annotated[Optional[int], Field(alias=AliasChoices('preferences', 'caregiver_preferences'))] = None
    optional_patients : Annotated[Optional[int], Field(alias=AliasChoices('optional_patients', 'unscheduled',))] = None
    missed_lunch_break : Optional[int] = None
    qualification : Optional[int] = None
    working_time : Optional[int] = None
    incompabilities : Optional[int] = None
    tw_max_dev_in_time : Optional[int] = None
    tw_max_dev_in_desired_time : Optional[int] = None
    workload_balance : Optional[int] = None
    max_waiting_time : Optional[int] = None

class Solution(BaseModel):
    """
    Represents a solution for a healthcare scheduling scenario.

    Attributes:
        instance (Optional[str]): The name of the instance this solution is for.
        routes (list[CaregiverRoute]): A list of routes for each caregiver.
        _normalized (bool): Internal flag indicating if the routes have been normalized.
    
    Methods:
        _check_validity() -> 'Solution':
            Validates the solution data after model creation.
        check_validity(instance: Instance):
            Checks the validity of the solution against the given instance.
        compute_costs(instance: Instance) -> float:
            Computes and returns the costs associated with the solution.
    """
    instance: Optional[Annotated[str, Field(min_length=1)]] = None
    routes: list[CaregiverRoute]
    cost_components: Optional[SolutionCostComponents] = None
    _normalized: Annotated[bool, Field(exclude=True, repr=False)] = False

    @model_validator(mode="before")
    @classmethod
    def validate_sd(cls, values : Any) -> Any:
        print(values["cost_components"])
        for i, route in enumerate(values["routes"]):
            if "locations" not in route.keys():
                continue
            locations = route["locations"]
            ordered = sorted(locations, key=lambda x: x.get("arrival_time", x.get("start_time", x.get("start_service_time"))))
            values["routes"][i]["locations"] = ordered
            # print(i,r["locations"])

        #route["locations"] = sorted(route["locations"], key=lambda x: x["arrival_time"])
        return values    

    @model_validator(mode='after')
    def _normalize_solution(self) -> 'Solution':
        assert len(self.routes) == len(set(r.caregiver_id for r in self.routes)), "Some caregivers are repeated in the solution"
        self._normalized = True
        for r in self.routes:            
            if any(l.arrival_at_patient is not None for l in r._visits):
                self._normalized = False
                break

        return self
    
    def check_validity(self, instance : Instance):
        def is_component_hard(component, none_as_hard=True) -> bool:
            if none_as_hard:
                return component is None or component == "HARD"
            else:
                return component == "HARD"
            
        
        flag_lunch_break = True if instance.metadata.cost_components.missed_lunch_break is not None else False
        cg_doing_lunch_break = set()
        cg_doing_lunch_break_spec = dict()
    
        # check that all patients that are not optional are visited
        visited_patients = {l.patient for r in self.routes for l in r._visits if l.service!="lunch_break"} 
        
        # check that the visited patients exist
        for p in visited_patients:
            if p not in instance._patients.keys():
                click.secho(f"Patient {p} does not refer to the instance at hand.", fg="red")
                assert p in instance._patients.keys(), f"Patient {p} does not refer to the instance at hand."

        non_optional_patients = set()
        for p in instance.patients:
            if p.optional is None or p.optional is False:
                non_optional_patients.add(p.id)
        for p in non_optional_patients:
            if p not in visited_patients:
            # assert p in visited_patients, f"Patient {p} is not visited"
                click.secho(f"Patient {p} is not optional, but not scheduled", fg="red")

        # check that all services required by each patient are provided -- note you should consider only those patients that are served
        # note that you do not need to check further whether this is an optional patient, as you should have checked this before
        for p in instance.patients:
            if p.id in visited_patients:
                for s in p.required_services:
                    providing_caregivers = {r.caregiver_id for r in self.routes for l in r._visits if l.patient == p.id and l.service == s.service}
                    assert len(providing_caregivers) >= 1, f"Patient {p.id} requires service {s.service} which is not provided"
                    assert len(providing_caregivers) == 1, f"Patient {p.id} requires service {s.service} which is provided by more than one caregiver"

        
        # normalize routes, by transforming them into full routes
        # TODO: this does not account for the multi-modal
        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            if r._full_route:
                continue
            if r.locations is None:
                click.secho(f"Route of caregiver {r.caregiver_id} is empty", fg="yellow")
                continue
            if len(r.locations) == 1 and r.locations[0].service=="lunch_break":
                click.secho(f"Route of caregiver {r.caregiver_id} is empty, but lunch is consumed", fg="yellow")
    
            # search for the depot which is the first location of the caregiver
            depot_departure = instance._terminal_points[c.departing_point]           
            # search for the patient which is the first location of the caregiver
            first_service_index = None 
            if r._visits[0].service == "lunch_break" and r._visits[0].patient not in visited_patients:
                # FIXME: this is okay under the assumption that arrival point = departing point
                # this means that the lunch is consumed at the depot
                first_service_index = instance._terminal_points[instance._caregivers[r.caregiver_id].departing_point].distance_matrix_index
            else:
                first_service_index = instance._patients[r._visits[0].patient].distance_matrix_index
            travel_time = instance.distances[depot_departure.distance_matrix_index][first_service_index]
            # compute the latest time the caregiver can depart to arrive at first patient
            depot_departure_time = r.locations[0].start_service_time - travel_time
            if instance.metadata.origin not in ["bazirha","bazirha-caie"]:#, "generated"]:                        
                r.locations.insert(0, DepotDeparture(departing_time=depot_departure_time, depot=depot_departure.id))
            else:
                r.locations.insert(0, DepotDeparture(departing_time=c.working_shift.start, depot=depot_departure.id))
            
            # compute the earliest time the caregiver arrive at the depot after the last patient (not that the arrival depot might differ from the departure one)
            depot_arrival = instance._terminal_points[c.arrival_point]
            last_patient = None
            if r._visits[-1].service == "lunch_break" and r._visits[-1].patient not in visited_patients:
                last_patient = instance._terminal_points[instance._caregivers[r.caregiver_id].departing_point]
            else:
                last_patient = instance._patients[r._visits[-1].patient]

            travel_time = instance.distances[last_patient.distance_matrix_index][depot_arrival.distance_matrix_index]
            depot_arrival_time = r._visits[-1].end_service_time + travel_time
            r.locations.append(DepotArrival(arrival_time=depot_arrival_time, depot=depot_arrival.id))
            r._full_route = True

        # check that the times are consistent with the distances and normalize the arrival times
        # TODO: this does not account for the multi-modal
        for num_route,r in enumerate(self.routes):
            if r.locations is None:
                continue # Don't throw anything as you should have already checked this
            for i in range(len(r._visits) - 1):
                start_index = instance._patients[r._visits[i].patient].distance_matrix_index
                end_index = instance._patients[r._visits[i + 1].patient].distance_matrix_index
                travel_time = instance.distances[start_index][end_index]
                assert r._visits[i + 1].start_service_time >= r._visits[i].end_service_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances ({r.locations[i].end_service_time} + {travel_time} vs {r.locations[i + 1].start_service_time})"
                if not r._visits[i + 1].arrival_at_patient or r._visits[i + 1].arrival_at_patient  is None:
                    r._visits[i + 1].arrival_at_patient = r._visits[i].end_service_time + travel_time
                assert r._visits[i + 1].start_service_time >= r._visits[i + 1].arrival_at_patient, f"Route of caregiver {r.caregiver_id} is not consistent with the arrival at patient"                
            # check the first location (i.e., depot)
            start_index = instance._terminal_points[r.locations[0].depot].distance_matrix_index
            end_index = None
            if r.locations[1].service == "lunch_break" and r.locations[1].patient not in visited_patients:
                end_index = instance._terminal_points[instance._caregivers[r.caregiver_id].departing_point].distance_matrix_index
            else: 
                end_index = instance._patients[r.locations[1].patient].distance_matrix_index
            travel_time = instance.distances[start_index][end_index]
            assert r.locations[1].start_service_time >= r.locations[0].departing_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances from the depot ({r.locations[0].departing_time} + {travel_time} vs {r.locations[1].start_service_time})"
            if not r.locations[1].arrival_at_patient or r.locations[1].arrival_at_patient is None:
                r.locations[1].arrival_at_patient = r.locations[0].departing_time + travel_time
            assert r.locations[1].start_service_time >= r.locations[1].arrival_at_patient, f"Route of caregiver {r.caregiver_id} is not consistent with the arrival at patient"
            # check the last location (i.e., depot)
            start_index = None
            if r.locations[-2].service == "lunch_break" and r.locations[-2].patient not in visited_patients:
                start_index = instance._terminal_points[instance._caregivers[r.caregiver_id].departing_point].distance_matrix_index
            else:
                start_index = instance._patients[r.locations[-2].patient].distance_matrix_index
            end_index = instance._terminal_points[r.locations[-1].depot].distance_matrix_index
            travel_time = instance.distances[start_index][end_index]
            assert r.locations[-1].arrival_time >= r.locations[-2].end_service_time + travel_time, f"Route of caregiver {r.caregiver_id} is not consistent with the distances to the depot ({r.locations[-2].end_service_time} + {travel_time} vs {r.locations[-1].arrival_time})"
            # check that the times are consistent with the service duration
            for num_visits,l in enumerate(r._visits):
                if l.service == "lunch_break":
                    lunch_ok = True
                    if flag_lunch_break:
                        if instance.metadata.time_window_met == "at_service_start":
                        # assert l.start_service_time <= p.time_windows[idx].end, f"Patient {p_id} is late by {l.start_service_time - p.time_windows[idx].end}"
                            if l.start_service_time > instance.lunch_breaks.end:
                                click.secho(f"Caregiver {r.caregiver_id} is late for lunch.",fg="red")
                                lunch_ok = False
                        else:
                            # assert l.end_service_time <= p.time_windows[idx].end, f"Patient {p_id} is late by {l.end_service_time - p.time_windows[idx].end}"
                            if l.end_service_time  > instance.lunch_breaks.end:
                                click.secho(f"Caregiver {r.caregiver_id} is late for lunch.",fg="red")
                                lunch_ok = False
                        if l.start_service_time < instance.lunch_breaks.start:
                            click.secho(f"Caregiver {r.caregiver_id} is starting the lunch break before the lunch break official start ({l.start_service_time} vs. {instance.lunch_breaks.start})", fg="red")
                            lunch_ok = False
                        if l.end_service_time - l.start_service_time < instance.lunch_breaks.min_duration:
                            click.secho(f"Caregiver {r.caregiver_id} is having a short break", fg="red")
                            lunch_ok = False
                            # check the actual lunch break
                        if lunch_ok:
                            cg_doing_lunch_break.add(r.caregiver_id)
                        else:
                            click.secho(f"Lunch for caregiver {r.caregiver_id} is not compliant", fg="red")
                        cg_doing_lunch_break_spec[r.caregiver_id] = {"start": l.start_service_time, "end": l.end_service_time, "duration": l.end_service_time-l.start_service_time}
                else:
                    s = next(s for s in instance._patients[l.patient].required_services if s.service == l.service)
                    #if l.end_service_time - l.start_service_time > s._actual_duration:
                    #    print("inconsistnecy", l.arrival_at_patient, l.start_service_time, l.end_service_time - l.start_service_time, s.duration)
                    assert l.end_service_time - l.start_service_time >= s._actual_duration, f"Route of caregiver {r.caregiver_id} at {l} is not consistent with the service durations"
            # check that the times are consistent with the working shift
            c = instance._caregivers[r.caregiver_id]
            assert c.working_shift is None or r.locations[0].departing_time >= c.working_shift.start, f"Route of caregiver {r.caregiver_id} is not consistent with his/her working shift (start shift: {c.working_shift.start}, departing_time: {r.locations[0].departing_time})"
        
        # check that for patients requiring sequential and simultaneous services, the services are provided correctly
        # i.e., you need to exclude those cases where you have indipendent services 
        for p in instance.patients:  
            if p.id not in visited_patients:
                continue          
            if p.synchronization is not None and p.synchronization.type != "independent": 
                caregivers = {l.service: l for r in self.routes for l in r._visits if l.patient == p.id and l.service != "lunch_break"}
                assert len(caregivers) == 2, f"Patient {p.id} requires double service but they are provided by the same caregiver"
                if p.synchronization.type == 'simultaneous':
                    assert caregivers[p.required_services[0].service].start_service_time == caregivers[p.required_services[0].service].start_service_time, f"Patient {p.id} requires simultaneous service but they are not provided simultaneously {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time}"
                else:
                    assert caregivers[p.required_services[0].service].start_service_time + p.synchronization.distance.min <= caregivers[p.required_services[1].service].start_service_time, f"Patient {p.id} requires sequential service ({p.required_services[0].service}, {p.required_services[1].service}) but the order is not respected (second service starting too early {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time} while min distance should be {p.synchronization.distance.min})"
                    assert caregivers[p.required_services[0].service].start_service_time + p.synchronization.distance.max >= caregivers[p.required_services[1].service].start_service_time, f"Patient {p.id} requires sequential service ({p.required_services[0].service}, {p.required_services[1].service}) but the order is not respected (second service starting too late {caregivers[p.required_services[0].service].start_service_time} vs {caregivers[p.required_services[1].service].start_service_time} while max distance should be {p.synchronization.distance.max})"
       
        # check that a patient is served after his/her time window starts
        for p in instance.patients:
            for r in self.routes:
                for l in r._visits:
                    if l.patient == p.id and l.service!= "lunch_break":
                        assert int(l.start_service_time) >= int(p.time_windows[0].start), f"Patient {p.id} is served ({l.start_service_time}) before the first time window starts {p.time_windows[0].start}"

        # check that all caregivers provide services for which they are qualified
        # this is an hard constraint just in some type of instances, i.e., those where there is not the qualification cost component
        if is_component_hard(instance.metadata.cost_components.qualification):
            for r in self.routes:
                for l in r._visits:
                    c = instance._caregivers[r.caregiver_id]                    
                    if l.service not in c.abilities and l.service!="lunch_break":
                        click.secho(f"Caregiver {c.id} is providing a service {l.service} for which he/she is not qualified", fg="red")
        
        # check the lunch break if you have this flag
        if is_component_hard(instance.metadata.cost_components.missed_lunch_break):
            for cg in instance.caregivers:
                if cg.lunch_break:
                    if cg.id not in cg_doing_lunch_break:
                        click.secho(f"Caregiver {cg.id} should have a lunch break, but is not having lunch break", fg="red")
                else:
                    if cg.id in cg_doing_lunch_break:
                        click.secho(f"Caregiver {cg.id} should not have a lunch break, but is having lunch break", fg="red")
        
        # check that preferences are respected 
        if is_component_hard(instance.metadata.cost_components.caregiver_preferences):
            # for every done service
            for r in self.routes:
                for l in r._visits:
                    if l.service == "lunch_break":
                        continue
                    # if the patient accounts for the concept of preferred_caregiver
                    if instance._patients[l.patient].preferred_caregivers is not None:
                        if r.caregiver_id not in instance._patients[l.patient].preferred_caregivers:
                            click.secho(f"Patient {l.patient} is served by a caregiver ({r.caregiver_id}) that is not among their favourites", fg="red")
        
        # check incompatible caregivers
        if is_component_hard(instance.metadata.cost_components.incompabilities):
            # for every done service
            for r in self.routes:
                for l in r._visits:
                    if l.service == "lunch_break":
                        continue
                    # if the patient accounts for the concept of preferred_caregiver
                    if instance._patients[l.patient].incompatible_caregivers is not None:
                        if r.caregiver_id in instance._patients[l.patient].incompatible_caregivers:
                            click.secho(f"Patient {l.patient} is served by a caregiver ({r.caregiver_id}) that is among their incompatible caregivers", fg="red")
        
        # this should be already checked but for a comprehensive evaluation
        # optional_patients
        if is_component_hard(instance.metadata.cost_components.optional_patients):
            if len(visited_patients) != len(instance._patients):
                click.secho(f"The number of visited patients ({len(visited_patients)}) and the one of total patients ({len(instance._patients)}) should be equal", fg="red")
        
        # check working times of the caregiver
        if is_component_hard(instance.metadata.cost_components.working_time, False):
            for r in self.routes:
                c = instance._caregivers[r.caregiver_id]
                if c.working_shift is not None and r.locations is not None: # one for each non-empty route
                    if r.locations[-1].arrival_time > c.working_shift.end:
                        click.secho(f"The shift of caregiver {c.id} is not respected (finishing at {r.locations[-1].arrival_time}, instead of {c.working_shift.end})", fg="red")
        
        if is_component_hard(instance.metadata.cost_components.total_tardiness, False) or is_component_hard(instance.metadata.cost_components.highest_tardiness, False):
            for r in self.routes:
                for l in r._visits:
                    if l.service == "lunch_break":
                        continue
                    p_id = l.patient
                    p = instance._patients[p_id]
                    starts = [tw.start for tw in p.time_windows] 
                    idx = bisect.bisect_right(starts, l.start_service_time) - 1 # this give you the position where you should insert the service_start with respect the time of the starts of the windows. The -1 is so to have the index of the time window
                    if instance.metadata.time_window_met == "at_service_start":
                        # assert l.start_service_time <= p.time_windows[idx].end, f"Patient {p_id} is late by {l.start_service_time - p.time_windows[idx].end}"
                        if l.start_service_time > p.time_windows[idx].end:
                            click.secho(f"Patient {p_id} is late by {l.start_service_time - p.time_windows[idx].end}",fg="red")
                    else:
                        # assert l.end_service_time <= p.time_windows[idx].end, f"Patient {p_id} is late by {l.end_service_time - p.time_windows[idx].end}"
                        if l.end_service_time > p.time_windows[idx].end:
                            click.secho(f"Patient {p_id} is late by {l.end_service_time - p.time_windows[idx].end}",fg="red")

        click.secho("Solution correcly validated", fg="green")


    def compute_costs(self, instance : Instance) -> dict:
        click.secho("About to summurize the solution information", fg="green")
        
        visited_patients = {l.patient for r in self.routes for l in r._visits if l.service != "lunch_break"} 

        # compute distance travelled
        click.secho("*** TRAVELLED DISTANCES ***", fg="blue")
        travel_times = {c: 0 for c in instance._caregivers.keys()}
        for c in instance._caregivers.keys():
            travel_times[c] = 0
        distance_traveled = 0  
        for r in self.routes:
            distances_prints = [] 
            c = instance._caregivers[r.caregiver_id]
            start_index = instance._terminal_points[c.departing_point].distance_matrix_index
            for l in r._visits:
                end_index = None
                if l.patient in visited_patients:
                    end_index = instance._patients[l.patient].distance_matrix_index
                else: # FIXME: this is valid just because arrival and departing depot are the same
                    end_index = instance._terminal_points[instance._caregivers[r.caregiver_id].departing_point].distance_matrix_index
                tt = instance.distances[start_index][end_index]
                distances_prints.append(tt)
                distance_traveled += tt
                travel_times[c.id] += tt
                start_index = end_index
            end_index = instance._terminal_points[c.arrival_point].distance_matrix_index
            tt = instance.distances[start_index][end_index]
            distance_traveled += tt
            travel_times[c.id] += tt
            distances_prints.append(tt)
            click.secho(f"Distances of caregiver {r.caregiver_id} are: {distances_prints} (total: {travel_times[c.id]})", fg="blue")
        
        # compute tardiness
        click.secho("*** TARDINESS ***", fg="blue")
        tardiness = []
        for r in self.routes:
            for l in r._visits:
                if l.service == "lunch_break":
                    continue
                p_id = l.patient
                p = instance._patients[p_id]
                starts = [tw.start for tw in p.time_windows] 
                idx = bisect.bisect_right(starts, l.start_service_time) - 1 # this give you the position where you should insert the service_start with respect the time of the starts of the windows. The -1 is so to have the index of the time window
                if instance.metadata.time_window_met == "at_service_start":
                    tardiness.append(max(0, l.start_service_time - p.time_windows[idx].end))
                    if l.start_service_time > p.time_windows[idx].end:
                        click.secho(f"Patient {p.id} visited by {r.caregiver_id} is tardy ({l.start_service_time - p.time_windows[idx].end})", fg="blue")
                else:
                    tardiness.append(max(0, l.end_service_time - p.time_windows[idx].end))
                    if l.end_service_time > p.time_windows[idx].end:
                        click.secho(f"Patient {p.id} visited by {r.caregiver_id} is tardy ({l.end_service_time - p.time_windows[idx].end})", fg="blue")

                        
        # compute waiting times
        click.secho("*** WAITING TIMES ***", fg="blue")
        waiting_time = []        
        for r in self.routes:
            for n_visit, l in enumerate(r._visits):
                if n_visit == 1 and r._visits[0].service == "lunch_break":
                    click.secho(f"(Not counting as waiting time) Caregiver {r.caregiver_id} was having lunch before service {l.service} of patient {l.patient} (arrival: {l.arrival_at_patient}, start: {l.start_service_time}, end: {l.end_service_time})",fg="blue")
                    continue
                if l.service == "lunch_break" :
                    click.secho(f"(Not counting as waiting time) Caregiver {r.caregiver_id} is having lunch at patient {l.patient} (arrival: {l.arrival_at_patient}, start: {l.start_service_time}, end: {l.end_service_time})",fg="blue")
                #     continue
                if l.arrival_at_patient < l.start_service_time:
                    #print(l.service, l.start_service_time, l.arrival_at_patient)
                    waiting_time.append(l.start_service_time - l.arrival_at_patient)
                    click.secho(f"Caregiver {r.caregiver_id} is waiting {l.start_service_time - l.arrival_at_patient} time units at patient {l.patient} (arrival: {l.arrival_at_patient}, start: {l.start_service_time}, end: {l.end_service_time})",fg="blue")
                else:
                    click.secho(f"(Not counting as waiting time) Caregiver {r.caregiver_id} is at patient {l.patient} (arrival: {l.arrival_at_patient}, start: {l.start_service_time}, end: {l.end_service_time})",fg="blue")
        
        # compute extratime
        click.secho("*** EXTRA TIME ***", fg="blue")
        extra_time = 0 
        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            if c.working_shift is not None and r.locations is not None: # one for each non-empty route
                if r.locations[-1].arrival_time > c.working_shift.end:
                    extra_time += r.locations[-1].arrival_time - c.working_shift.end
                    click.secho(f"Caregiver {c.id} is working extra time ({r.locations[-1].arrival_time - c.working_shift.end})", fg="blue")
        
        
        click.secho("*** INCOMPATIBILITIES ***", fg="blue")
        # number of incompatible caregivers
        incompatible_caregiver = 0
        for r in self.routes:
            for l in r._visits:
                if l.service == "lunch_break":
                    continue
                # if the patient accounts for the concept of preferred_caregiver
                if instance._patients[l.patient].incompatible_caregivers is not None:
                    if r.caregiver_id in instance._patients[l.patient].incompatible_caregivers:
                        incompatible_caregiver += 1
                        click.secho(f"Patient {l.patient} is served by an incompatible caregiver ({r.caregiver_id})", fg="blue")
        
        # compute max idle time
        idle_times = {c: 0 for c in instance._caregivers.keys()}       
        for r in self.routes:
            c = instance._caregivers[r.caregiver_id]
            if c.working_shift is not None: 
                if r.locations is None:
                    idle_times[c.id] += (c.working_shift.end - c.working_shift.start)
                else:
                    if r.locations[0].departing_time > c.working_shift.start:
                        idle_times[c.id] += r.locations[0].departing_time - c.working_shift.start
                    for i, l in enumerate(r._visits):
                        if l.arrival_at_patient < l.start_service_time:
                            idle_times[c.id] += l.start_service_time - l.arrival_at_patient
                    if r.locations[-1].arrival_time <= c.working_shift.end:
                        idle_times[c.id] += c.working_shift.end - r.locations[-1].arrival_time
        max_idle_time = max(idle_times.values())
        click.secho("*** IDLE TIMES ***", fg="blue")
        for c in idle_times.keys():
            if idle_times[c] >= max_idle_time:
                click.secho(f"Caregiver {c} has an idle time of {idle_times[c]}, which corresponds to the maximum.", fg = "blue")
            else:
                click.secho(f"Caregiver {c} has an idle time of {idle_times[c]}.", fg = "blue")

         # compute worload balance
        click.secho("*** SERVICE TIMES ***", fg="blue")
        # the workload of a caregiver is given by the service time and the travel time (you already calculated the travel time while calculating the travel distance)
        service_times = {c: 0 for c in instance._caregivers.keys()}
        for r in self.routes:
            for i, l in enumerate(r._visits):
                if l.service == "lunch_break":
                    continue
                service_times[r.caregiver_id] += l.end_service_time - l.start_service_time
        for c in service_times.keys():
            click.secho(f"Caregiver {c} is at service for the following times: {service_times[c]}",fg="blue")
        
        total_workload = 0
        click.secho("*** WORKLOAD ***", fg="blue")
        for c in instance._caregivers.keys():
            total_workload += service_times[c] + travel_times[c]
            click.secho(f"Caregiver {c} has a workload of : {service_times[c] + travel_times[c]}",fg="blue")
        avg_workload = float(total_workload) / float(len(instance._caregivers))
        balance = 0
        for c in instance._caregivers.keys():
            balance += math.ceil(abs(service_times[c] + travel_times[c] - avg_workload))
        
        # wrong qualifications
        click.secho("*** QUALIFICATIONS ***", fg="blue")
        qualifications = 0
        for r in self.routes:
            for l in r._visits:
                if l.service == "lunch_break":
                    continue
                c = instance._caregivers[r.caregiver_id]
                if l.service not in c.abilities:
                    qualifications += 1
                    click.secho(f"Caregiver {r.caregiver_id} is performing a service ({l.service}) for which they are not abilitated.", fg="blue")
        
        cg_doing_lunch_break = set()
        for r in self.routes:
            if r.locations is None:
                continue 
            for l in r._visits:
                if l.service == "lunch_break":
                    cg_doing_lunch_break.add(r.caregiver_id)
        
        # number of not respected preferences
        click.secho("*** PREFERENCES ***", fg="blue")
        preferred_caregiver = 0
        for r in self.routes:
            for l in r._visits: 
                if l.service == "lunch_break":
                    continue
                if instance._patients[l.patient].preferred_caregivers is not None:
                    if r.caregiver_id not in instance._patients[l.patient].preferred_caregivers:
                        preferred_caregiver += 1
                        click.secho(f"Patient {l.patient} is served by a caregiver who is not one of their preferred ({r.caregiver_id})", fg="blue")
        
        # number of non visited optional patients
        click.secho("*** OPTIONAL PATIENTS ***", fg="blue")
        non_visited_optional_patient = 0 
        set_non_visited_optional = set()
        for p in instance.patients:
            if p.optional:
                if p.id not in visited_patients:
                    non_visited_optional_patient += 1
                    set_non_visited_optional.add(p.id)
                    # TODO: this should be only on activations
                    click.secho(f"Patient {p.id} is optional and not visited", fg="blue")
            else:
                if p.id not in visited_patients:
                    #click.secho(f"patient {p.id} should be visited", fg="red")
                    non_visited_optional_patient += 1
                    set_non_visited_optional.add(p.id)
        
        # number of missed lunch break
        click.secho("*** MISSED LUNCHES ***", fg="blue")
        missed_lunch_breaks = 0
        for cg in instance.caregivers:
            if cg.lunch_break:
                if cg.id not in cg_doing_lunch_break:
                    missed_lunch_breaks +=1

                    # TODO: this should be only on activations
                    click.secho(f"Caregiver {cg.id} is not having lunch", fg="blue")
        
        def get_multiplier_cost(cost_component):
            if cost_component is None:
                return 0
            if cost_component == "HARD":
                return 1
            return cost_component
        
        tot = get_multiplier_cost(instance.metadata.cost_components.optional_patients) * non_visited_optional_patient +\
            get_multiplier_cost(instance.metadata.cost_components.total_tardiness) *  (sum(tardiness) if len(tardiness) > 0 else 0) +\
            get_multiplier_cost(instance.metadata.cost_components.highest_tardiness) *  (max(tardiness) if len(tardiness) > 0 else 0) +\
            get_multiplier_cost(instance.metadata.cost_components.travel_time) * distance_traveled +\
            get_multiplier_cost(instance.metadata.cost_components.total_extra_time) * extra_time +\
            get_multiplier_cost(instance.metadata.cost_components.max_idle_time) * max_idle_time +\
            get_multiplier_cost(instance.metadata.cost_components.total_waiting_time) * (sum(waiting_time) if len(waiting_time) > 0 else 0) +\
            get_multiplier_cost(instance.metadata.cost_components.max_waiting_time) * ( max(waiting_time) if len(waiting_time) > 0 else 0) +\
            get_multiplier_cost(instance.metadata.cost_components.workload_balance) * balance +\
            get_multiplier_cost(instance.metadata.cost_components.working_time) * total_workload +\
            get_multiplier_cost(instance.metadata.cost_components.incompabilities) * incompatible_caregiver +\
            get_multiplier_cost(instance.metadata.cost_components.caregiver_preferences) * preferred_caregiver +\
            get_multiplier_cost(instance.metadata.cost_components.qualification) * qualifications +\
            get_multiplier_cost(instance.metadata.cost_components.missed_lunch_break) * missed_lunch_breaks

        cost_string_dict = {
            'non_visited_optional_patients': f"{get_multiplier_cost(instance.metadata.cost_components.optional_patients) * non_visited_optional_patient}  ({get_multiplier_cost(instance.metadata.cost_components.optional_patients)} x {non_visited_optional_patient})",
            'total_tardiness': f"{get_multiplier_cost(instance.metadata.cost_components.total_tardiness) *  (sum(tardiness) if len(tardiness) > 0 else 0)}  ({get_multiplier_cost(instance.metadata.cost_components.total_tardiness)} x { sum(tardiness) if len(tardiness) > 0 else 0})",
            'max_tardiness': f"{get_multiplier_cost(instance.metadata.cost_components.highest_tardiness) *  (max(tardiness) if len(tardiness) > 0 else 0)}  ({get_multiplier_cost(instance.metadata.cost_components.highest_tardiness)} x { max(tardiness) if len(tardiness) > 0 else 0})",
            'traveled_distance': f"{get_multiplier_cost(instance.metadata.cost_components.travel_time) * distance_traveled}  ({get_multiplier_cost(instance.metadata.cost_components.travel_time)} x {distance_traveled})",
            'total_extra_time': f"{get_multiplier_cost(instance.metadata.cost_components.total_extra_time) * extra_time}  ({get_multiplier_cost(instance.metadata.cost_components.total_extra_time)} x {extra_time})",
            'max_idle_time': f"{get_multiplier_cost(instance.metadata.cost_components.max_idle_time) * max_idle_time}  ({get_multiplier_cost(instance.metadata.cost_components.max_idle_time)} x {max_idle_time})",
            'total_waiting_time': f"{get_multiplier_cost(instance.metadata.cost_components.total_waiting_time) * (sum(waiting_time) if len(waiting_time) > 0 else 0)}  ({get_multiplier_cost(instance.metadata.cost_components.total_waiting_time)} x {(sum(waiting_time) if len(waiting_time) > 0 else 0)})",
            'max_waiting_time': f"{get_multiplier_cost(instance.metadata.cost_components.max_waiting_time) * ( max(waiting_time) if len(waiting_time) > 0 else 0)}  ({get_multiplier_cost(instance.metadata.cost_components.max_waiting_time)} x {(max(waiting_time) if len(waiting_time) > 0 else 0,)})",
            'workload_balance': f"{get_multiplier_cost(instance.metadata.cost_components.workload_balance) * balance}  ({get_multiplier_cost(instance.metadata.cost_components.workload_balance)} x {balance})",
            'total_working_time': f"{get_multiplier_cost(instance.metadata.cost_components.working_time) * total_workload}  ({get_multiplier_cost(instance.metadata.cost_components.working_time)} x {total_workload})",
            'incompatible_caregiver': f"{get_multiplier_cost(instance.metadata.cost_components.incompabilities) * incompatible_caregiver}  ({get_multiplier_cost(instance.metadata.cost_components.incompabilities)} x {incompatible_caregiver})",  
            'preferred_caregiver': f"{get_multiplier_cost(instance.metadata.cost_components.caregiver_preferences) * preferred_caregiver}  ({get_multiplier_cost(instance.metadata.cost_components.caregiver_preferences)} x {preferred_caregiver})",
            'qualifications': f"{get_multiplier_cost(instance.metadata.cost_components.qualification) * qualifications}  ({get_multiplier_cost(instance.metadata.cost_components.qualification)} x {qualifications})",
            'missed_lunch_breaks': f"{get_multiplier_cost(instance.metadata.cost_components.missed_lunch_break) * missed_lunch_breaks}  ({get_multiplier_cost(instance.metadata.cost_components.missed_lunch_break)} x {missed_lunch_breaks})", 
            'Total': f"{tot} (-)"
        }

        click.secho("Cost computed by validator are:", fg="green")

        max_key_length = max(len(key) for key in cost_string_dict)
        max_value_number_length = max(len(value.split()[0]) for value in cost_string_dict.values())

        for key, value in cost_string_dict.items():
            num, rest = value.split(maxsplit=1)
            formatted_line = f"{key.ljust(max_key_length + 5, '.')} {num.rjust(max_value_number_length)} ..... {rest}"
            click.secho(formatted_line, fg="blue")
        
        click.secho("About to check the costs of your solution:", fg="green")
        
        def checker(name, component_solution, component_validator, weight):
            if component_solution is not None:
                if component_solution == component_validator or component_solution == (get_multiplier_cost(weight)*component_validator):
                    click.secho(f"Cost related to {name} is consistent [ {component_solution } vs. {get_multiplier_cost(weight)*component_validator} ({get_multiplier_cost(weight)} x {component_validator}) ]", fg="blue")
                else:
                    click.secho(f"Cost related to {name} is not consistent [ {component_solution } vs. {get_multiplier_cost(weight)*component_validator} ({get_multiplier_cost(weight)} x {component_validator}) ]", fg="red")
        # get_multiplier_cost(instance.metadata.cost_components.optional_patients) * non_visited_optional_patient +\
        checker("non_visited_optional_patients", self.cost_components.optional_patients, non_visited_optional_patient, instance.metadata.cost_components.optional_patients)
        # get_multiplier_cost(instance.metadata.cost_components.total_tardiness) *  (sum(tardiness) if len(tardiness) > 0 else 0) +\
        checker("total_tardiness", self.cost_components.total_tardiness, (sum(tardiness) if len(tardiness) > 0 else 0), instance.metadata.cost_components.total_tardiness)
        # get_multiplier_cost(instance.metadata.cost_components.highest_tardiness) *  (max(tardiness) if len(tardiness) > 0 else 0) +\
        checker("max_tardiness", self.cost_components.highest_tardiness,  (max(tardiness) if len(tardiness) > 0 else 0), instance.metadata.cost_components.highest_tardiness)
        # get_multiplier_cost(instance.metadata.cost_components.travel_time) * distance_traveled +\
        checker("traveled_distance",self.cost_components.travel_time, distance_traveled, instance.metadata.cost_components.travel_time)
        # get_multiplier_cost(instance.metadata.cost_components.total_extra_time) * extra_time +\
        checker("total_extra_time", self.cost_components.total_extra_time, extra_time, instance.metadata.cost_components.total_extra_time)
        # get_multiplier_cost(instance.metadata.cost_components.max_idle_time) * max_idle_time +\
        checker("max_idle_time",self.cost_components.max_idle_time, max_idle_time, instance.metadata.cost_components.max_idle_time)
        # get_multiplier_cost(instance.metadata.cost_components.total_waiting_time) * (sum(waiting_time) if len(waiting_time) > 0 else 0) +\
        checker("total_waiting_time", self.cost_components.total_waiting_time,  (sum(waiting_time) if len(waiting_time) > 0 else 0), instance.metadata.cost_components.total_waiting_time )
        # get_multiplier_cost(instance.metadata.cost_components.max_waiting_time) * ( max(waiting_time) if len(waiting_time) > 0 else 0) +
        checker("max_waiting_time", self.cost_components.max_waiting_time, ( max(waiting_time) if len(waiting_time) > 0 else 0), instance.metadata.cost_components.max_waiting_time)
        # get_multiplier_cost(instance.metadata.cost_components.workload_balance) * balance +\
        checker("workload_balance", self.cost_components.workload_balance, balance, instance.metadata.cost_components.workload_balance)
        # get_multiplier_cost(instance.metadata.cost_components.working_time) * total_workload +\
        checker("total_workload", self.cost_components.working_time, total_workload, instance.metadata.cost_components.working_time)
        # get_multiplier_cost(instance.metadata.cost_components.incompabilities) * incompatible_caregiver +\
        checker("incompatible_caregiver", self.cost_components.incompabilities, incompatible_caregiver,instance.metadata.cost_components.incompabilities)
        # get_multiplier_cost(instance.metadata.cost_components.caregiver_preferences) * preferred_caregiver +\
        checker("preferred_caregiver", self.cost_components.caregiver_preferences, preferred_caregiver, instance.metadata.cost_components.caregiver_preferences)
        # get_multiplier_cost(instance.metadata.cost_components.qualification) * qualifications +\
        checker("qualifications", self.cost_components.qualification, qualifications, instance.metadata.cost_components.qualification)
        # get_multiplier_cost(instance.metadata.cost_components.missed_lunch_break) * missed_lunch_breaks
        checker("missed_lunch_breaks", self.cost_components.missed_lunch_break, missed_lunch_breaks, instance.metadata.cost_components.missed_lunch_break)

        return cost_string_dict