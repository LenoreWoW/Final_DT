"""
Hospital Operations Classical Baseline

Greedy heuristic + local search for hospital resource allocation and patient scheduling.
Simplified linear programming approach.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Patient priority level"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    """Hospital resource types"""
    BED = "bed"
    NURSE = "nurse"
    DOCTOR = "doctor"
    OR = "operating_room"
    ICU = "icu"


@dataclass
class Patient:
    """Patient record"""
    id: int
    arrival_time: float
    priority: Priority
    required_resources: Dict[ResourceType, int]
    treatment_duration: float
    wait_time: float = 0.0
    start_time: float = -1
    end_time: float = -1


@dataclass
class Resource:
    """Hospital resource"""
    type: ResourceType
    total_count: int
    available_count: int
    schedule: List[Tuple[float, float, int]]  # (start, end, patient_id)


class HospitalOperationsClassical:
    """Classical hospital operations optimization"""

    def __init__(self,
                 resources: Dict[ResourceType, int]):
        """
        Initialize hospital operations system

        Args:
            resources: Dictionary mapping resource types to counts
        """
        self.resources = {
            rtype: Resource(
                type=rtype,
                total_count=count,
                available_count=count,
                schedule=[]
            )
            for rtype, count in resources.items()
        }

        self.patients: List[Patient] = []
        self.scheduled_patients: List[Patient] = []
        self.current_time = 0.0

    def add_patient(self, patient: Patient):
        """Add patient to queue"""
        self.patients.append(patient)

    def _is_resource_available(self,
                              resource_type: ResourceType,
                              start_time: float,
                              duration: float,
                              required_count: int) -> bool:
        """Check if resource is available for the given time slot"""
        resource = self.resources[resource_type]
        end_time = start_time + duration

        # Count how many units are occupied during this time slot
        occupied = 0
        for sched_start, sched_end, _ in resource.schedule:
            # Check for overlap
            if not (end_time <= sched_start or start_time >= sched_end):
                occupied += 1

        available = resource.total_count - occupied
        return available >= required_count

    def _can_schedule_patient(self,
                             patient: Patient,
                             start_time: float) -> bool:
        """Check if patient can be scheduled at given time"""
        for resource_type, required_count in patient.required_resources.items():
            if not self._is_resource_available(
                resource_type,
                start_time,
                patient.treatment_duration,
                required_count
            ):
                return False
        return True

    def _find_earliest_slot(self, patient: Patient) -> float:
        """Find earliest time slot where patient can be scheduled"""
        # Start from patient arrival time
        current_time = patient.arrival_time

        # Try time slots
        max_time = current_time + 1000  # Don't search forever
        time_step = 0.5  # Check every 30 minutes

        while current_time < max_time:
            if self._can_schedule_patient(patient, current_time):
                return current_time

            current_time += time_step

        # If no slot found, schedule far in future
        return max_time

    def _schedule_patient(self, patient: Patient, start_time: float):
        """Schedule a patient"""
        patient.start_time = start_time
        patient.end_time = start_time + patient.treatment_duration
        patient.wait_time = start_time - patient.arrival_time

        # Reserve resources
        for resource_type, required_count in patient.required_resources.items():
            resource = self.resources[resource_type]
            for _ in range(required_count):
                resource.schedule.append((
                    patient.start_time,
                    patient.end_time,
                    patient.id
                ))

        self.scheduled_patients.append(patient)

    def greedy_schedule(self) -> Dict:
        """
        Greedy scheduling algorithm

        Schedules patients by priority, then arrival time
        """
        start_time = time.time()

        # Sort patients by priority (descending), then arrival time (ascending)
        sorted_patients = sorted(
            self.patients,
            key=lambda p: (-p.priority.value, p.arrival_time)
        )

        # Schedule each patient
        for patient in sorted_patients:
            earliest_slot = self._find_earliest_slot(patient)
            self._schedule_patient(patient, earliest_slot)

        elapsed_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics()
        metrics['optimization_time'] = elapsed_time
        metrics['method'] = 'Greedy + Local Search'

        return metrics

    def local_search_optimization(self, max_iterations: int = 100) -> Dict:
        """
        Improve schedule using local search

        Try swapping patient slots to reduce average wait time
        """
        start_time = time.time()

        # Reset scheduling state so greedy_schedule starts fresh
        self.scheduled_patients = []
        for resource in self.resources.values():
            resource.schedule = []

        # First do greedy schedule
        self.greedy_schedule()

        initial_wait = np.mean([p.wait_time for p in self.scheduled_patients])

        # Local search iterations
        for iteration in range(max_iterations):
            improved = False

            # Try swapping pairs of patients
            for i in range(len(self.scheduled_patients)):
                for j in range(i + 1, len(self.scheduled_patients)):
                    patient_i = self.scheduled_patients[i]
                    patient_j = self.scheduled_patients[j]

                    # Try swapping their start times
                    if self._can_swap(patient_i, patient_j):
                        # Calculate current wait
                        current_wait = patient_i.wait_time + patient_j.wait_time

                        # Simulate swap
                        new_wait_i = patient_j.start_time - patient_i.arrival_time
                        new_wait_j = patient_i.start_time - patient_j.arrival_time

                        # Only swap if it reduces total wait time
                        # and neither wait time becomes negative
                        if (new_wait_i >= 0 and new_wait_j >= 0 and
                                new_wait_i + new_wait_j < current_wait):
                            self._swap_patients(patient_i, patient_j)
                            improved = True

            if not improved:
                break

        elapsed_time = time.time() - start_time

        # Calculate final metrics
        metrics = self._calculate_metrics()
        metrics['optimization_time'] = elapsed_time
        metrics['initial_average_wait'] = initial_wait
        metrics['improvement'] = initial_wait - metrics['average_wait_time']
        metrics['method'] = 'Greedy + Local Search'

        return metrics

    def _can_swap(self, patient1: Patient, patient2: Patient) -> bool:
        """Check if two patients can swap time slots"""
        # For simplicity, only swap if they have same duration
        # and same resource requirements
        if patient1.treatment_duration != patient2.treatment_duration:
            return False

        if patient1.required_resources != patient2.required_resources:
            return False

        return True

    def _swap_patients(self, patient1: Patient, patient2: Patient):
        """Swap two patients' time slots"""
        # Save old times for schedule update
        old_start1, old_end1 = patient1.start_time, patient1.end_time
        old_start2, old_end2 = patient2.start_time, patient2.end_time

        # Swap start times
        patient1.start_time, patient2.start_time = patient2.start_time, patient1.start_time

        # Update end times
        patient1.end_time = patient1.start_time + patient1.treatment_duration
        patient2.end_time = patient2.start_time + patient2.treatment_duration

        # Update wait times
        patient1.wait_time = patient1.start_time - patient1.arrival_time
        patient2.wait_time = patient2.start_time - patient2.arrival_time

        # Update resource schedules to reflect the swap
        for resource in self.resources.values():
            for idx, (start, end, pid) in enumerate(resource.schedule):
                if pid == patient1.id and start == old_start1 and end == old_end1:
                    resource.schedule[idx] = (patient1.start_time, patient1.end_time, patient1.id)
                elif pid == patient2.id and start == old_start2 and end == old_end2:
                    resource.schedule[idx] = (patient2.start_time, patient2.end_time, patient2.id)

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.scheduled_patients:
            return {
                'average_wait_time': 0,
                'max_wait_time': 0,
                'patients_scheduled': 0,
                'resource_utilization': {}
            }

        # Wait time statistics
        wait_times = [p.wait_time for p in self.scheduled_patients]
        average_wait = np.mean(wait_times)
        max_wait = np.max(wait_times)
        median_wait = np.median(wait_times)

        # Priority-specific wait times
        priority_waits = {}
        for priority in Priority:
            patients_priority = [
                p for p in self.scheduled_patients
                if p.priority == priority
            ]
            if patients_priority:
                priority_waits[priority.name] = np.mean([p.wait_time for p in patients_priority])

        # Resource utilization
        total_time = max([p.end_time for p in self.scheduled_patients])
        resource_util = {}

        for rtype, resource in self.resources.items():
            if resource.schedule:
                # Calculate total busy time
                busy_time = sum([
                    (end - start)
                    for start, end, _ in resource.schedule
                ])

                # Utilization per resource unit
                utilization = busy_time / (total_time * resource.total_count) if total_time > 0 else 0
                resource_util[rtype.value] = utilization * 100

        return {
            'patients_scheduled': len(self.scheduled_patients),
            'average_wait_time': average_wait,
            'median_wait_time': median_wait,
            'max_wait_time': max_wait,
            'priority_wait_times': priority_waits,
            'resource_utilization': resource_util,
            'total_simulation_time': total_time
        }


def generate_patient_stream(n_patients: int = 100,
                           time_window: float = 24.0,
                           seed: Optional[int] = None) -> List[Patient]:
    """
    Generate random patient arrivals

    Args:
        n_patients: Number of patients
        time_window: Time window in hours
        seed: Random seed

    Returns:
        List of patients
    """
    rng = np.random.RandomState(seed)

    patients = []

    for i in range(n_patients):
        # Random arrival time
        arrival_time = rng.uniform(0, time_window)

        # Priority distribution (weighted towards lower priority)
        priority_values = [1, 2, 3, 4]
        priority_probs = [0.4, 0.3, 0.2, 0.1]
        priority_value = rng.choice(priority_values, p=priority_probs)
        priority = Priority(priority_value)

        # Treatment duration (higher priority = longer treatment)
        base_duration = rng.uniform(0.5, 3.0)
        duration = base_duration * priority_value

        # Resource requirements
        if priority == Priority.CRITICAL:
            required_resources = {
                ResourceType.BED: 1,
                ResourceType.DOCTOR: 2,
                ResourceType.NURSE: 3,
                ResourceType.ICU: 1
            }
        elif priority == Priority.HIGH:
            required_resources = {
                ResourceType.BED: 1,
                ResourceType.DOCTOR: 1,
                ResourceType.NURSE: 2
            }
        else:
            required_resources = {
                ResourceType.BED: 1,
                ResourceType.DOCTOR: 1,
                ResourceType.NURSE: 1
            }

        patient = Patient(
            id=i,
            arrival_time=arrival_time,
            priority=priority,
            required_resources=required_resources,
            treatment_duration=duration
        )

        patients.append(patient)

    return patients


def run_hospital_operations_classical(n_patients: int = 100) -> Dict:
    """
    Run classical hospital operations benchmark

    Args:
        n_patients: Number of patients to schedule

    Returns:
        Benchmark results
    """
    # Hospital resources — reduced to create realistic contention
    resources = {
        ResourceType.BED: max(n_patients // 3, 3),
        ResourceType.DOCTOR: max(n_patients // 4, 2),
        ResourceType.NURSE: max(n_patients // 3, 3),
        ResourceType.OR: 5,
        ResourceType.ICU: 5
    }

    # Generate patient stream
    patients = generate_patient_stream(
        n_patients=n_patients,
        time_window=24.0,
        seed=42
    )

    # Initialize hospital
    hospital = HospitalOperationsClassical(resources=resources)

    # Add patients
    for patient in patients:
        hospital.add_patient(patient)

    # Run optimization
    results = hospital.local_search_optimization(max_iterations=50)

    return results


if __name__ == '__main__':
    # Test the classical hospital operations
    print("Testing Classical Hospital Operations...")
    print("=" * 60)

    results = run_hospital_operations_classical(n_patients=100)

    print(f"\nPatients Scheduled: {results['patients_scheduled']}")
    print(f"Optimization Time: {results['optimization_time']:.2f} seconds")

    print(f"\nWait Time Statistics:")
    print(f"  Average: {results['average_wait_time']:.2f} hours")
    print(f"  Median: {results['median_wait_time']:.2f} hours")
    print(f"  Maximum: {results['max_wait_time']:.2f} hours")

    if 'initial_average_wait' in results:
        print(f"\nImprovement:")
        print(f"  Initial Wait: {results['initial_average_wait']:.2f} hours")
        print(f"  Final Wait: {results['average_wait_time']:.2f} hours")
        print(f"  Improvement: {results['improvement']:.2f} hours ({results['improvement']/results['initial_average_wait']*100:.1f}%)")

    print(f"\nPriority-Based Wait Times:")
    for priority, wait in results['priority_wait_times'].items():
        print(f"  {priority}: {wait:.2f} hours")

    print(f"\nResource Utilization:")
    for resource, util in results['resource_utilization'].items():
        print(f"  {resource}: {util:.1f}%")


def run_integer_programming_baseline(
    n_patients: int = 50,
    n_beds: int = 20,
    n_doctors: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Integer programming baseline for hospital scheduling using scipy.optimize.linprog.

    Formulates the patient-to-slot assignment as a linear program with
    integrality constraints approximated through LP relaxation + rounding.
    This is a proper QAOA counterpart (both solve combinatorial optimization).

    Args:
        n_patients: Number of patients to schedule.
        n_beds: Number of available beds.
        n_doctors: Number of available doctors.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with scheduling results including objective value,
        wait times, utilization, and solve time.
    """
    from scipy.optimize import linprog

    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_slots = n_beds  # one slot per bed
    n_vars = n_patients * n_slots  # binary assignment variables x_{i,j}

    # Cost vector: prefer earlier slots and respect priority
    priorities = rng.choice([1, 2, 3, 4], size=n_patients, p=[0.4, 0.3, 0.2, 0.1])
    c = np.zeros(n_vars)
    for i in range(n_patients):
        for j in range(n_slots):
            # Higher priority patients get lower cost for earlier slots
            c[i * n_slots + j] = (j + 1) / (priorities[i] + 1)

    # Constraints: each patient assigned to at most one slot
    A_eq_rows = []
    b_eq = []
    for i in range(n_patients):
        row = np.zeros(n_vars)
        row[i * n_slots: (i + 1) * n_slots] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    # Constraints: each slot has at most one patient
    A_ub_rows = []
    b_ub = []
    for j in range(n_slots):
        row = np.zeros(n_vars)
        for i in range(n_patients):
            row[i * n_slots + j] = 1.0
        A_ub_rows.append(row)
        b_ub.append(1.0)

    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub) if b_ub else None

    bounds = [(0, 1)] * n_vars

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    elapsed = time.time() - start_time

    # Round LP relaxation to integer solution
    x = np.round(result.x).astype(int) if result.success else np.zeros(n_vars, dtype=int)
    assignments = x.reshape(n_patients, n_slots)

    # Compute metrics
    assigned_slots = [np.argmax(assignments[i]) if np.any(assignments[i]) else -1 for i in range(n_patients)]
    wait_times = [max(0, s) for s in assigned_slots if s >= 0]
    avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
    utilization = float(np.sum(x)) / n_slots * 100 if n_slots > 0 else 0.0

    return {
        "method": "Integer Programming (LP relaxation + rounding)",
        "optimization_time": elapsed,
        "objective_value": float(result.fun) if result.success else float("inf"),
        "n_patients": n_patients,
        "n_slots": n_slots,
        "patients_assigned": int(np.sum(np.any(assignments, axis=1))),
        "average_wait_time": avg_wait,
        "bed_utilization": min(utilization, 100.0),
        "solver_status": result.message if result.success else "failed",
    }
