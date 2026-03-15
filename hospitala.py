#!/usr/bin/env python3
"""
Emergency Room Discrete Event Simulation
=========================================
A detailed simulation of an emergency room with:
- Patient arrivals (random, with varying severity)
- Resource management (doctors, nurses, beds, devices)
- Patient flow: arrival -> registration -> triage -> treatment -> discharge/death
- Doctor breaks and shift management
- Detailed statistics tracking
- Terminal-based real-time visualization
- Tkinter GUI visualization
"""

import random
import heapq
import time
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
from datetime import timedelta
import threading
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except ImportError:  # Allows CLI mode without tkinter installed
    tk = None
    ttk = None
    messagebox = None

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""
    # Time parameters (in minutes)
    simulation_duration: int = 1440  # 24 hours
    time_scale: float = 0.01  # Real seconds per simulated minute
    
    # Patient arrival parameters
    base_arrival_rate: float = 0.15  # Patients per minute base rate
    peak_arrival_rate: float = 0.35  # Patients per minute during peak
    peak_start: int = 480  # 8 AM
    peak_end: int = 1200  # 8 PM
    
    # Patient severity distribution
    critical_prob: float = 0.08
    severe_prob: float = 0.20
    moderate_prob: float = 0.35
    mild_prob: float = 0.25
    minor_prob: float = 0.12
    
    # Resource counts
    num_doctors: int = 4
    num_nurses: int = 8
    num_beds: int = 20
    num_monitors: int = 15
    num_ventilators: int = 5
    num_defibrillators: int = 3
    num_xray_machines: int = 2
    num_lab_stations: int = 4
    
    # Process times (mean, std dev in minutes)
    registration_time: tuple = (3, 1)
    triage_time: tuple = (5, 2)
    doctor_consult_time: tuple = (15, 5)
    nurse_care_time: tuple = (10, 3)
    treatment_time_critical: tuple = (60, 30)
    treatment_time_severe: tuple = (90, 40)
    treatment_time_moderate: tuple = (45, 20)
    treatment_time_mild: tuple = (25, 10)
    treatment_time_minor: tuple = (10, 5)
    
    # Death parameters
    critical_death_rate: float = 0.15  # Base death probability for critical
    severe_death_rate: float = 0.05
    critical_death_wait_factor: float = 0.02  # Additional death chance per minute waiting
    severe_death_wait_factor: float = 0.005
    
    # Doctor break parameters
    doctor_break_interval: int = 240  # Minutes between breaks
    doctor_break_duration: tuple = (15, 5)  # Mean, std dev for break duration
    doctor_shift_duration: int = 480  # 8 hours
    
    # Resource wait thresholds for statistics
    buildup_threshold: int = 5  # Minutes considered a buildup
    
    # Display parameters
    refresh_interval: float = 0.5  # Seconds between display updates

# =============================================================================
# ENUMERATIONS
# =============================================================================

class PatientSeverity(Enum):
    CRITICAL = auto()  # Life-threatening, immediate treatment needed
    SEVERE = auto()    # Serious condition, needs urgent care
    MODERATE = auto()  # Moderate condition, can wait some time
    MILD = auto()      # Mild condition, low priority
    MINOR = auto()     # Minor issue, lowest priority

class PatientState(Enum):
    ARRIVING = auto()
    WAITING_REGISTRATION = auto()
    REGISTERING = auto()
    WAITING_TRIAGE = auto()
    TRIAGING = auto()
    WAITING_TREATMENT = auto()
    GETTING_DOCTOR = auto()
    GETTING_NURSE = auto()
    IN_TREATMENT = auto()
    WAITING_DISCHARGE = auto()
    DISCHARGED = auto()
    DECEASED = auto()

class ResourceType(Enum):
    DOCTOR = auto()
    NURSE = auto()
    BED = auto()
    MONITOR = auto()
    VENTILATOR = auto()
    DEFIBRILLATOR = auto()
    XRAY = auto()
    LAB = auto()

class EventType(Enum):
    PATIENT_ARRIVAL = auto()
    REGISTRATION_COMPLETE = auto()
    TRIAGE_COMPLETE = auto()
    DOCTOR_CONSULT_COMPLETE = auto()
    NURSE_CARE_COMPLETE = auto()
    TREATMENT_COMPLETE = auto()
    PATIENT_DISCHARGE = auto()
    DOCTOR_BREAK_START = auto()
    DOCTOR_BREAK_END = auto()
    DOCTOR_SHIFT_END = auto()
    PATIENT_DEATH = auto()
    SIMULATION_END = auto()
    RESOURCE_CHECK = auto()


def format_sim_time(minutes: int) -> str:
    """Format minutes into HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

# =============================================================================
# PATIENT MODEL
# =============================================================================

@dataclass
class Patient:
    """Patient in the emergency room"""
    id: int
    arrival_time: int
    severity: PatientSeverity
    state: PatientState = PatientState.ARRIVING
    
    # Timing tracking
    registration_start: Optional[int] = None
    registration_complete: Optional[int] = None
    triage_start: Optional[int] = None
    triage_complete: Optional[int] = None
    treatment_start: Optional[int] = None
    treatment_complete: Optional[int] = None
    discharge_time: Optional[int] = None
    death_time: Optional[int] = None
    
    # Resource assignments
    assigned_doctor: Optional[int] = None
    assigned_nurse: Optional[int] = None
    assigned_bed: Optional[int] = None
    assigned_monitor: Optional[int] = None
    assigned_ventilator: Optional[int] = None
    assigned_defibrillator: Optional[int] = None
    assigned_xray: Optional[int] = None
    assigned_lab: Optional[int] = None
    
    # Treatment details
    needs_doctor: bool = True
    needs_nurse: bool = True
    needs_bed: bool = True
    needs_monitor: bool = False
    needs_ventilator: bool = False
    needs_defibrillator: bool = False
    needs_xray: bool = False
    needs_lab: bool = False
    treatment_complexity: int = 1  # Number of resources needed
    
    # Death tracking
    death_cause: Optional[str] = None  # "no_treatment", "resource_shortage", "natural"
    
    def __lt__(self, other):
        # Priority queue ordering - critical first, then by arrival time
        severity_order = {
            PatientSeverity.CRITICAL: 0,
            PatientSeverity.SEVERE: 1,
            PatientSeverity.MODERATE: 2,
            PatientSeverity.MILD: 3,
            PatientSeverity.MINOR: 4
        }
        if severity_order[self.severity] != severity_order[other.severity]:
            return severity_order[self.severity] < severity_order[other.severity]
        return self.arrival_time < other.arrival_time

# =============================================================================
# RESOURCE POOL
# =============================================================================

@dataclass
class Resource:
    """A single resource unit"""
    id: int
    type: ResourceType
    available: bool = True
    current_patient: Optional[int] = None
    in_use_since: Optional[int] = None

class ResourcePool:
    """Manages a pool of resources"""
    
    def __init__(self, resource_type: ResourceType, count: int):
        self.resource_type = resource_type
        self.resources: Dict[int, Resource] = {}
        self.available_queue: List[int] = []
        self.wait_queue: List[Patient] = []
        
        for i in range(count):
            res = Resource(id=i, type=resource_type)
            self.resources[i] = res
            self.available_queue.append(i)
    
    @property
    def available_count(self) -> int:
        return len(self.available_queue)
    
    @property
    def total_count(self) -> int:
        return len(self.resources)
    
    @property
    def in_use_count(self) -> int:
        return self.total_count - self.available_count
    
    def acquire(self, patient: Patient) -> Optional[int]:
        """Try to acquire a resource. Returns resource ID or None"""
        if self.available_queue:
            resource_id = self.available_queue.pop(0)
            self.resources[resource_id].available = False
            self.resources[resource_id].current_patient = patient.id
            return resource_id
        return None
    
    def release(self, resource_id: int) -> Optional[int]:
        """Release a resource. Returns patient ID if was assigned"""
        if resource_id in self.resources:
            patient_id = self.resources[resource_id].current_patient
            self.resources[resource_id].available = True
            self.resources[resource_id].current_patient = None
            self.available_queue.append(resource_id)
            return patient_id
        return None
    
    def add_to_wait_queue(self, patient: Patient):
        """Add patient to wait queue for this resource"""
        self.wait_queue.append(patient)
    
    def get_from_wait_queue(self) -> Optional[Patient]:
        """Get next patient from wait queue"""
        if self.wait_queue:
            return self.wait_queue.pop(0)
        return None
    
    def get_wait_queue_length(self) -> int:
        return len(self.wait_queue)

# =============================================================================
# EVENTS
# =============================================================================

@dataclass
class Event:
    """Simulation event"""
    time: int
    event_type: EventType
    patient: Optional[Patient] = None
    resource_id: Optional[int] = None
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        return self.time < other.time

# =============================================================================
# STATISTICS TRACKER
# =============================================================================

@dataclass
class Statistics:
    """Comprehensive statistics tracking"""
    
    # Patient counts
    total_arrivals: int = 0
    total_discharged: int = 0
    total_deceased: int = 0
    
    # Death causes
    deaths_no_treatment: int = 0  # Died before getting any treatment
    deaths_resource_shortage: int = 0  # Died because resources unavailable
    deaths_natural: int = 0  # Died despite treatment (illness)
    
    # By severity
    arrivals_by_severity: Dict[PatientSeverity, int] = field(default_factory=lambda: defaultdict(int))
    deaths_by_severity: Dict[PatientSeverity, int] = field(default_factory=lambda: defaultdict(int))
    
    # Timing statistics
    wait_times_registration: List[int] = field(default_factory=list)
    wait_times_triage: List[int] = field(default_factory=list)
    wait_times_treatment: List[int] = field(default_factory=list)
    total_time_in_er: List[int] = field(default_factory=list)
    
    # By severity
    wait_times_by_severity: Dict[PatientSeverity, List[int]] = field(default_factory=lambda: defaultdict(list))
    
    # Resource statistics
    resource_utilization: Dict[ResourceType, List[float]] = field(default_factory=lambda: defaultdict(list))
    max_wait_queue_lengths: Dict[ResourceType, int] = field(default_factory=lambda: defaultdict(int))
    
    # Buildup tracking
    buildup_periods: List[Dict] = field(default_factory=list)
    current_buildup: Optional[Dict] = None
    
    # Doctor statistics
    doctor_breaks_taken: int = 0
    doctor_total_break_time: int = 0
    patients_per_doctor: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Time-based tracking
    patients_in_system_over_time: List[tuple] = field(default_factory=list)
    resource_usage_over_time: List[tuple] = field(default_factory=list)
    
    # Critical patient tracking
    critical_patients_saved: int = 0
    critical_patients_lost: int = 0

    # Dean of medicine decisions
    dean_decisions: List[Dict[str, Any]] = field(default_factory=list)
    dean_impact_log: List[str] = field(default_factory=list)
    
    def record_arrival(self, patient: Patient):
        self.total_arrivals += 1
        self.arrivals_by_severity[patient.severity] += 1
    
    def record_discharge(self, patient: Patient, current_time: int):
        self.total_discharged += 1
        total_time = current_time - patient.arrival_time
        self.total_time_in_er.append(total_time)
        
        if patient.severity == PatientSeverity.CRITICAL:
            self.critical_patients_saved += 1
    
    def record_death(self, patient: Patient, cause: str, current_time: int):
        self.total_deceased += 1
        self.deaths_by_severity[patient.severity] += 1
        
        if cause == "no_treatment":
            self.deaths_no_treatment += 1
        elif cause == "resource_shortage":
            self.deaths_resource_shortage += 1
        else:
            self.deaths_natural += 1
        
        if patient.severity == PatientSeverity.CRITICAL:
            self.critical_patients_lost += 1
    
    def record_wait_time(self, patient: Patient, stage: str, wait_time: int):
        if stage == "registration":
            self.wait_times_registration.append(wait_time)
        elif stage == "triage":
            self.wait_times_triage.append(wait_time)
        elif stage == "treatment":
            self.wait_times_treatment.append(wait_time)
        
        self.wait_times_by_severity[patient.severity].append(wait_time)
    
    def record_resource_usage(self, resource_type: ResourceType, utilization: float):
        self.resource_utilization[resource_type].append(utilization)
    
    def update_max_wait_queue(self, resource_type: ResourceType, length: int):
        if length > self.max_wait_queue_lengths[resource_type]:
            self.max_wait_queue_lengths[resource_type] = length
    
    def start_buildup(self, time: int, resource_type: ResourceType, queue_length: int):
        if not self.current_buildup:
            self.current_buildup = {
                'start_time': time,
                'resource_type': resource_type,
                'max_queue': queue_length
            }
    
    def end_buildup(self, time: int):
        if self.current_buildup:
            self.current_buildup['end_time'] = time
            self.current_buildup['duration'] = time - self.current_buildup['start_time']
            self.buildup_periods.append(self.current_buildup)
            self.current_buildup = None

# =============================================================================
# DISCRETE EVENT SIMULATOR
# =============================================================================

class ERSimulator:
    """Main discrete event simulator for the Emergency Room"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.current_time: int = 0
        self.event_queue: List[Event] = []
        self.patients: Dict[int, Patient] = {}
        self.next_patient_id: int = 1
        
        # Resource pools
        self.resources: Dict[ResourceType, ResourcePool] = {}
        
        # Doctor management
        self.doctor_breaks: Dict[int, bool] = {}  # doctor_id -> on_break
        self.doctor_next_break: Dict[int, int] = {}  # doctor_id -> next break time
        self.doctor_shift_end: Dict[int, int] = {}  # doctor_id -> shift end
        
        # Statistics
        self.stats = Statistics()
        
        # Current patients in various stages
        self.patients_waiting_registration: List[Patient] = []
        self.patients_waiting_triage: List[Patient] = []
        self.patients_waiting_treatment: List[Patient] = []
        self.patients_in_treatment: List[Patient] = []
        
        # Running state
        self.running: bool = False
        self.paused: bool = False
        self.simulation_complete: bool = False
        
        # Initialize
        self._initialize_resources()
        self._schedule_initial_events()
    
    def _initialize_resources(self):
        """Initialize all resource pools"""
        cfg = self.config
        
        self.resources[ResourceType.DOCTOR] = ResourcePool(ResourceType.DOCTOR, cfg.num_doctors)
        self.resources[ResourceType.NURSE] = ResourcePool(ResourceType.NURSE, cfg.num_nurses)
        self.resources[ResourceType.BED] = ResourcePool(ResourceType.BED, cfg.num_beds)
        self.resources[ResourceType.MONITOR] = ResourcePool(ResourceType.MONITOR, cfg.num_monitors)
        self.resources[ResourceType.VENTILATOR] = ResourcePool(ResourceType.VENTILATOR, cfg.num_ventilators)
        self.resources[ResourceType.DEFIBRILLATOR] = ResourcePool(ResourceType.DEFIBRILLATOR, cfg.num_defibrillators)
        self.resources[ResourceType.XRAY] = ResourcePool(ResourceType.XRAY, cfg.num_xray_machines)
        self.resources[ResourceType.LAB] = ResourcePool(ResourceType.LAB, cfg.num_lab_stations)
        
        # Initialize doctor states
        for i in range(cfg.num_doctors):
            self.doctor_breaks[i] = False
            self.doctor_next_break[i] = cfg.doctor_break_interval
            self.doctor_shift_end[i] = cfg.doctor_shift_duration
    
    def _schedule_initial_events(self):
        """Schedule initial events"""
        # First patient arrival
        self._schedule_next_arrival()
        
        # Simulation end
        heapq.heappush(self.event_queue, Event(
            time=self.config.simulation_duration,
            event_type=EventType.SIMULATION_END
        ))
        
        # Initial resource check
        heapq.heappush(self.event_queue, Event(
            time=0,
            event_type=EventType.RESOURCE_CHECK
        ))
        
        # Schedule doctor breaks
        for doc_id in range(self.config.num_doctors):
            heapq.heappush(self.event_queue, Event(
                time=self.config.doctor_break_interval,
                event_type=EventType.DOCTOR_BREAK_START,
                resource_id=doc_id
            ))
    
    def _schedule_next_arrival(self):
        """Schedule the next patient arrival"""
        # Calculate arrival rate based on time of day
        if self.config.peak_start <= self.current_time <= self.config.peak_end:
            rate = self.config.peak_arrival_rate
        else:
            rate = self.config.base_arrival_rate
        
        # Exponential interarrival time
        interarrival = random.expovariate(rate)
        next_time = self.current_time + int(interarrival)
        
        if next_time < self.config.simulation_duration:
            heapq.heappush(self.event_queue, Event(
                time=next_time,
                event_type=EventType.PATIENT_ARRIVAL
            ))
    
    def _create_patient(self) -> Patient:
        """Create a new patient with random severity"""
        cfg = self.config
        r = random.random()
        
        if r < cfg.critical_prob:
            severity = PatientSeverity.CRITICAL
        elif r < cfg.critical_prob + cfg.severe_prob:
            severity = PatientSeverity.SEVERE
        elif r < cfg.critical_prob + cfg.severe_prob + cfg.moderate_prob:
            severity = PatientSeverity.MODERATE
        elif r < cfg.critical_prob + cfg.severe_prob + cfg.moderate_prob + cfg.mild_prob:
            severity = PatientSeverity.MILD
        else:
            severity = PatientSeverity.MINOR
        
        patient = Patient(
            id=self.next_patient_id,
            arrival_time=self.current_time,
            severity=severity
        )
        
        # Set resource needs based on severity
        if severity == PatientSeverity.CRITICAL:
            patient.needs_monitor = True
            patient.needs_ventilator = random.random() < 0.3
            patient.needs_defibrillator = random.random() < 0.1
            patient.treatment_complexity = 3
        elif severity == PatientSeverity.SEVERE:
            patient.needs_monitor = random.random() < 0.5
            patient.treatment_complexity = 2
        elif severity == PatientSeverity.MODERATE:
            patient.needs_monitor = random.random() < 0.2
            patient.needs_xray = random.random() < 0.3
            patient.needs_lab = random.random() < 0.3
        elif severity == PatientSeverity.MILD:
            patient.needs_nurse = random.random() < 0.7
            patient.needs_bed = random.random() < 0.8
        else:  # MINOR
            patient.needs_doctor = random.random() < 0.3
            patient.needs_nurse = random.random() < 0.5
            patient.needs_bed = random.random() < 0.3
        
        self.patients[self.next_patient_id] = patient
        self.next_patient_id += 1
        
        return patient
    
    def add_event(self, event: Event):
        """Add an event to the queue"""
        heapq.heappush(self.event_queue, event)
    
    def run(self):
        """Run the simulation"""
        self.running = True
        self.simulation_complete = False
        
        while self.running and self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            if event.time > self.current_time:
                self.current_time = event.time
            
            self._process_event(event)
            
            if event.event_type == EventType.SIMULATION_END:
                self.running = False
                self.simulation_complete = True

    def step(self) -> Optional[Event]:
        """Process a single event and return it (for GUI)"""
        if not self.event_queue or self.simulation_complete:
            self.simulation_complete = True
            return None
        
        event = heapq.heappop(self.event_queue)
        if event.time > self.current_time:
            self.current_time = event.time
        
        self._process_event(event)
        if event.event_type == EventType.SIMULATION_END:
            self.simulation_complete = True
        
        return event
    
    def _process_event(self, event: Event):
        """Process a single event"""
        if event.event_type == EventType.PATIENT_ARRIVAL:
            self._handle_patient_arrival()
        
        elif event.event_type == EventType.REGISTRATION_COMPLETE:
            self._handle_registration_complete(event.patient)
        
        elif event.event_type == EventType.TRIAGE_COMPLETE:
            self._handle_triage_complete(event.patient)
        
        elif event.event_type == EventType.DOCTOR_CONSULT_COMPLETE:
            self._handle_doctor_consult_complete(event.patient)
        
        elif event.event_type == EventType.NURSE_CARE_COMPLETE:
            self._handle_nurse_care_complete(event.patient)
        
        elif event.event_type == EventType.TREATMENT_COMPLETE:
            self._handle_treatment_complete(event.patient)
        
        elif event.event_type == EventType.PATIENT_DISCHARGE:
            self._handle_patient_discharge(event.patient)
        
        elif event.event_type == EventType.DOCTOR_BREAK_START:
            self._handle_doctor_break_start(event.resource_id)
        
        elif event.event_type == EventType.DOCTOR_BREAK_END:
            self._handle_doctor_break_end(event.resource_id)
        
        elif event.event_type == EventType.DOCTOR_SHIFT_END:
            self._handle_doctor_shift_end(event.resource_id)
        
        elif event.event_type == EventType.PATIENT_DEATH:
            self._handle_patient_death(event.patient)
        
        elif event.event_type == EventType.RESOURCE_CHECK:
            self._handle_resource_check()
        
        elif event.event_type == EventType.SIMULATION_END:
            pass  # Handled in run loop
    
    def _handle_patient_arrival(self):
        """Handle a new patient arrival"""
        patient = self._create_patient()
        self.stats.record_arrival(patient)
        
        # Schedule next arrival
        self._schedule_next_arrival()
        
        # Check for immediate death (critical patient with no resources)
        if patient.severity == PatientSeverity.CRITICAL:
            # Schedule potential death if waiting too long
            death_delay = random.expovariate(1 / 30)  # Mean 30 minutes to death
            heapq.heappush(self.event_queue, Event(
                time=self.current_time + int(death_delay),
                event_type=EventType.PATIENT_DEATH,
                patient=patient
            ))
        
        # Try to start registration
        self._try_start_registration(patient)
    
    def _try_start_registration(self, patient: Patient = None):
        """Try to start registration for a patient"""
        if patient is not None and patient.state == PatientState.ARRIVING:
            patient.state = PatientState.WAITING_REGISTRATION
            self.patients_waiting_registration.append(patient)
        
        # Check if resources available
        if self.resources[ResourceType.NURSE].available_count > 0:
            # Get next patient from queue (priority by severity)
            if self.patients_waiting_registration:
                # Sort by severity priority
                self.patients_waiting_registration.sort()
                next_patient = self.patients_waiting_registration.pop(0)
                
                # Acquire nurse for registration
                nurse_id = self.resources[ResourceType.NURSE].acquire(next_patient)
                if nurse_id is not None:
                    next_patient.assigned_nurse = nurse_id  # Track assigned nurse
                    next_patient.state = PatientState.REGISTERING
                    next_patient.registration_start = self.current_time
                    
                    # Schedule registration complete
                    reg_time = max(1, int(random.gauss(*self.config.registration_time)))
                    heapq.heappush(self.event_queue, Event(
                        time=self.current_time + reg_time,
                        event_type=EventType.REGISTRATION_COMPLETE,
                        patient=next_patient
                    ))
    
    def _handle_registration_complete(self, patient: Patient):
        """Handle registration completion"""
        patient.registration_complete = self.current_time
        patient.state = PatientState.WAITING_TRIAGE
        
        # Record wait time
        wait_time = patient.registration_start - patient.arrival_time
        self.stats.record_wait_time(patient, "registration", wait_time)
        
        # Release nurse
        if patient.assigned_nurse is not None:
            self.resources[ResourceType.NURSE].release(patient.assigned_nurse)
            patient.assigned_nurse = None
        
        # Move to triage queue
        self.patients_waiting_triage.append(patient)
        
        # Try to start triage
        self._try_start_triage()
        
        # Try to process next registration
        self._try_start_registration(None)
    
    def _try_start_triage(self):
        """Try to start triage for waiting patients"""
        if self.patients_waiting_triage:
            self.patients_waiting_triage.sort()
            patient = self.patients_waiting_triage[0]
            
            if self.resources[ResourceType.NURSE].available_count > 0:
                self.patients_waiting_triage.pop(0)
                
                nurse_id = self.resources[ResourceType.NURSE].acquire(patient)
                if nurse_id is not None:
                    patient.assigned_nurse = nurse_id  # Track assigned nurse
                    patient.state = PatientState.TRIAGING
                    patient.triage_start = self.current_time
                    
                    # Schedule triage complete
                    triage_time = max(1, int(random.gauss(*self.config.triage_time)))
                    heapq.heappush(self.event_queue, Event(
                        time=self.current_time + triage_time,
                        event_type=EventType.TRIAGE_COMPLETE,
                        patient=patient
                    ))
    
    def _handle_triage_complete(self, patient: Patient):
        """Handle triage completion"""
        patient.triage_complete = self.current_time
        patient.state = PatientState.WAITING_TREATMENT
        
        # Record wait time
        wait_time = patient.triage_start - patient.registration_complete
        self.stats.record_wait_time(patient, "triage", wait_time)
        
        # Release nurse
        if patient.assigned_nurse is not None:
            self.resources[ResourceType.NURSE].release(patient.assigned_nurse)
            patient.assigned_nurse = None
        
        # Move to treatment queue
        self.patients_waiting_treatment.append(patient)
        
        # Try to start treatment
        self._try_start_treatment()
        
        # Try to start next triage
        self._try_start_triage()
    
    def _try_start_treatment(self):
        """Try to start treatment for waiting patients"""
        if not self.patients_waiting_treatment:
            return
        
        self.patients_waiting_treatment.sort()
        
        # Process patients in priority order
        remaining = []
        for patient in self.patients_waiting_treatment:
            if self._try_assign_treatment_resources(patient):
                patient.state = PatientState.IN_TREATMENT
                patient.treatment_start = self.current_time
                self.patients_in_treatment.append(patient)
                
                # Schedule treatment complete based on severity
                if patient.severity == PatientSeverity.CRITICAL:
                    treatment_time = max(10, int(random.gauss(*self.config.treatment_time_critical)))
                elif patient.severity == PatientSeverity.SEVERE:
                    treatment_time = max(10, int(random.gauss(*self.config.treatment_time_severe)))
                elif patient.severity == PatientSeverity.MODERATE:
                    treatment_time = max(5, int(random.gauss(*self.config.treatment_time_moderate)))
                elif patient.severity == PatientSeverity.MILD:
                    treatment_time = max(5, int(random.gauss(*self.config.treatment_time_mild)))
                else:
                    treatment_time = max(2, int(random.gauss(*self.config.treatment_time_minor)))
                
                heapq.heappush(self.event_queue, Event(
                    time=self.current_time + treatment_time,
                    event_type=EventType.TREATMENT_COMPLETE,
                    patient=patient
                ))
            else:
                remaining.append(patient)
        
        self.patients_waiting_treatment = remaining
    
    def _try_assign_treatment_resources(self, patient: Patient) -> bool:
        """Try to assign all needed resources for treatment"""
        assigned_resources = {}
        
        # Try to get bed first (most critical)
        if patient.needs_bed:
            bed_id = self.resources[ResourceType.BED].acquire(patient)
            if bed_id is None:
                return False
            patient.assigned_bed = bed_id
            assigned_resources[ResourceType.BED] = bed_id
        
        # Try to get doctor
        if patient.needs_doctor:
            # Check if any doctor is available and not on break
            for doc_id in range(self.config.num_doctors):
                if not self.doctor_breaks.get(doc_id, True):
                    doc_id = self.resources[ResourceType.DOCTOR].acquire(patient)
                    if doc_id is not None:
                        patient.assigned_doctor = doc_id
                        assigned_resources[ResourceType.DOCTOR] = doc_id
                        break
            else:
                # Try any available doctor
                doc_id = self.resources[ResourceType.DOCTOR].acquire(patient)
                if doc_id is not None:
                    patient.assigned_doctor = doc_id
                    assigned_resources[ResourceType.DOCTOR] = doc_id
                else:
                    # Release any acquired resources
                    self._release_resources(patient, assigned_resources)
                    return False
        
        # Try to get nurse
        if patient.needs_nurse:
            nurse_id = self.resources[ResourceType.NURSE].acquire(patient)
            if nurse_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_nurse = nurse_id
            assigned_resources[ResourceType.NURSE] = nurse_id
        
        # Try to get monitor
        if patient.needs_monitor:
            monitor_id = self.resources[ResourceType.MONITOR].acquire(patient)
            if monitor_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_monitor = monitor_id
            assigned_resources[ResourceType.MONITOR] = monitor_id
        
        # Try to get ventilator
        if patient.needs_ventilator:
            vent_id = self.resources[ResourceType.VENTILATOR].acquire(patient)
            if vent_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_ventilator = vent_id
            assigned_resources[ResourceType.VENTILATOR] = vent_id
        
        # Try to get defibrillator
        if patient.needs_defibrillator:
            defib_id = self.resources[ResourceType.DEFIBRILLATOR].acquire(patient)
            if defib_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_defibrillator = defib_id
            assigned_resources[ResourceType.DEFIBRILLATOR] = defib_id
        
        # Try to get xray
        if patient.needs_xray:
            xray_id = self.resources[ResourceType.XRAY].acquire(patient)
            if xray_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_xray = xray_id
            assigned_resources[ResourceType.XRAY] = xray_id
        
        # Try to get lab
        if patient.needs_lab:
            lab_id = self.resources[ResourceType.LAB].acquire(patient)
            if lab_id is None:
                self._release_resources(patient, assigned_resources)
                return False
            patient.assigned_lab = lab_id
            assigned_resources[ResourceType.LAB] = lab_id
        
        return True
    
    def _release_resources(self, patient: Patient, resources: Dict[ResourceType, int]):
        """Release assigned resources"""
        for res_type, res_id in resources.items():
            self.resources[res_type].release(res_id)
            if res_type == ResourceType.BED:
                patient.assigned_bed = None
            elif res_type == ResourceType.DOCTOR:
                patient.assigned_doctor = None
            elif res_type == ResourceType.NURSE:
                patient.assigned_nurse = None
            elif res_type == ResourceType.MONITOR:
                patient.assigned_monitor = None
            elif res_type == ResourceType.VENTILATOR:
                patient.assigned_ventilator = None
            elif res_type == ResourceType.DEFIBRILLATOR:
                patient.assigned_defibrillator = None
            elif res_type == ResourceType.XRAY:
                patient.assigned_xray = None
            elif res_type == ResourceType.LAB:
                patient.assigned_lab = None
    
    def _handle_doctor_consult_complete(self, patient: Patient):
        """Handle doctor consultation completion"""
        pass  # Part of treatment flow
    
    def _handle_nurse_care_complete(self, patient: Patient):
        """Handle nurse care completion"""
        pass  # Part of treatment flow
    
    def _handle_treatment_complete(self, patient: Patient):
        """Handle treatment completion"""
        patient.treatment_complete = self.current_time
        patient.state = PatientState.WAITING_DISCHARGE
        
        # Record wait time for treatment
        if patient.triage_complete:
            wait_time = patient.treatment_start - patient.triage_complete
            self.stats.record_wait_time(patient, "treatment", wait_time)
        
        # Release all resources
        self._release_all_resources(patient)
        
        # Schedule discharge
        discharge_delay = random.randint(1, 5)
        heapq.heappush(self.event_queue, Event(
            time=self.current_time + discharge_delay,
            event_type=EventType.PATIENT_DISCHARGE,
            patient=patient
        ))
    
    def _handle_patient_discharge(self, patient: Patient):
        """Handle patient discharge"""
        patient.discharge_time = self.current_time
        patient.state = PatientState.DISCHARGED
        
        # Record discharge statistics
        self.stats.record_discharge(patient, self.current_time)
        
        # Try to start treatment for waiting patients (resources now available)
        self._try_start_treatment()
        self._try_start_triage()
        self._try_start_registration()
    
    def _release_all_resources(self, patient: Patient):
        """Release all resources assigned to patient"""
        if patient.assigned_bed is not None:
            self.resources[ResourceType.BED].release(patient.assigned_bed)
            patient.assigned_bed = None
        if patient.assigned_doctor is not None:
            self.resources[ResourceType.DOCTOR].release(patient.assigned_doctor)
            self.stats.patients_per_doctor[patient.assigned_doctor] += 1
            patient.assigned_doctor = None
        if patient.assigned_nurse is not None:
            self.resources[ResourceType.NURSE].release(patient.assigned_nurse)
            patient.assigned_nurse = None
        if patient.assigned_monitor is not None:
            self.resources[ResourceType.MONITOR].release(patient.assigned_monitor)
            patient.assigned_monitor = None
        if patient.assigned_ventilator is not None:
            self.resources[ResourceType.VENTILATOR].release(patient.assigned_ventilator)
            patient.assigned_ventilator = None
        if patient.assigned_defibrillator is not None:
            self.resources[ResourceType.DEFIBRILLATOR].release(patient.assigned_defibrillator)
            patient.assigned_defibrillator = None
        if patient.assigned_xray is not None:
            self.resources[ResourceType.XRAY].release(patient.assigned_xray)
            patient.assigned_xray = None
        if patient.assigned_lab is not None:
            self.resources[ResourceType.LAB].release(patient.assigned_lab)
            patient.assigned_lab = None
    
    def _handle_doctor_break_start(self, doctor_id: int):
        """Handle doctor break starting"""
        if self.doctor_breaks.get(doctor_id, False):
            return  # Already on break
        
        # Check if doctor can take break (no critical patients)
        can_break = True
        for patient in self.patients_in_treatment:
            if patient.assigned_doctor == doctor_id and patient.severity == PatientSeverity.CRITICAL:
                can_break = False
                break
        
        if can_break:
            self.doctor_breaks[doctor_id] = True
            self.stats.doctor_breaks_taken += 1
            
            # Schedule break end
            break_duration = max(5, int(random.gauss(*self.config.doctor_break_duration)))
            self.stats.doctor_total_break_time += break_duration
            
            heapq.heappush(self.event_queue, Event(
                time=self.current_time + break_duration,
                event_type=EventType.DOCTOR_BREAK_END,
                resource_id=doctor_id
            ))
        else:
            # Reschedule break for later
            heapq.heappush(self.event_queue, Event(
                time=self.current_time + 30,
                event_type=EventType.DOCTOR_BREAK_START,
                resource_id=doctor_id
            ))
    
    def _handle_doctor_break_end(self, doctor_id: int):
        """Handle doctor break ending"""
        self.doctor_breaks[doctor_id] = False
        
        # Schedule next break
        next_break = self.current_time + self.config.doctor_break_interval
        if next_break < self.config.simulation_duration:
            heapq.heappush(self.event_queue, Event(
                time=next_break,
                event_type=EventType.DOCTOR_BREAK_START,
                resource_id=doctor_id
            ))
        
        # Try to process waiting patients
        self._try_start_treatment()
    
    def _handle_doctor_shift_end(self, doctor_id: int):
        """Handle doctor shift ending"""
        pass  # Simplified - doctors continue working
    
    def _handle_patient_death(self, patient: Patient):
        """Handle patient death"""
        if patient.state == PatientState.DECEASED:
            return  # Already dead
        
        # Determine if patient should actually die
        should_die = False
        cause = "natural"
        
        if patient.state in [PatientState.WAITING_TREATMENT, PatientState.WAITING_TRIAGE, PatientState.WAITING_REGISTRATION]:
            # Check if death due to waiting
            wait_time = self.current_time - patient.arrival_time
            
            if patient.severity == PatientSeverity.CRITICAL:
                death_chance = self.config.critical_death_rate + (wait_time * self.config.critical_death_wait_factor)
                if random.random() < death_chance:
                    should_die = True
                    if self.resources[ResourceType.DOCTOR].available_count == 0 or \
                       self.resources[ResourceType.BED].available_count == 0:
                        cause = "resource_shortage"
                    else:
                        cause = "no_treatment"
            elif patient.severity == PatientSeverity.SEVERE:
                death_chance = self.config.severe_death_rate + (wait_time * self.config.severe_death_wait_factor)
                if random.random() < death_chance:
                    should_die = True
                    cause = "no_treatment"
        
        if should_die:
            patient.state = PatientState.DECEASED
            patient.death_time = self.current_time
            patient.death_cause = cause
            
            self.stats.record_death(patient, cause, self.current_time)
            
            # Release any assigned resources
            self._release_all_resources(patient)
            
            # Remove from queues
            if patient in self.patients_waiting_registration:
                self.patients_waiting_registration.remove(patient)
            if patient in self.patients_waiting_triage:
                self.patients_waiting_triage.remove(patient)
            if patient in self.patients_waiting_treatment:
                self.patients_waiting_treatment.remove(patient)
            if patient in self.patients_in_treatment:
                self.patients_in_treatment.remove(patient)
            
            # Try to process waiting patients
            self._try_start_treatment()
            self._try_start_triage()
            self._try_start_registration(None)
    
    def _handle_resource_check(self):
        """Periodic resource check for statistics"""
        # Record resource utilization
        for res_type, pool in self.resources.items():
            utilization = pool.in_use_count / pool.total_count if pool.total_count > 0 else 0
            self.stats.record_resource_usage(res_type, utilization)
            
            # Track wait queue lengths
            self.stats.update_max_wait_queue(res_type, pool.get_wait_queue_length())
            
            # Check for buildup
            if pool.get_wait_queue_length() >= self.config.buildup_threshold:
                if not self.stats.current_buildup:
                    self.stats.start_buildup(self.current_time, res_type, pool.get_wait_queue_length())
            elif self.stats.current_buildup:
                self.stats.end_buildup(self.current_time)
        
        # Record patients in system
        patients_in_system = len([p for p in self.patients.values() 
                                  if p.state not in [PatientState.DISCHARGED, PatientState.DECEASED]])
        self.stats.patients_in_system_over_time.append((self.current_time, patients_in_system))
        
        # Schedule next check
        if self.current_time < self.config.simulation_duration:
            heapq.heappush(self.event_queue, Event(
                time=self.current_time + 15,  # Every 15 minutes
                event_type=EventType.RESOURCE_CHECK
            ))

# =============================================================================
# TERMINAL DISPLAY
# =============================================================================

class TerminalDisplay:
    """Terminal-based display for the simulation"""
    
    def __init__(self, simulator: ERSimulator):
        self.simulator = simulator
        self.running = True
    
    def clear_screen(self):
        """Clear the terminal screen"""
        print("\033[2J\033[H", end="")
    
    def format_time(self, minutes: int) -> str:
        """Format minutes into HH:MM"""
        return format_sim_time(minutes)
    
    def get_severity_color(self, severity: PatientSeverity) -> str:
        """Get ANSI color for severity"""
        colors = {
            PatientSeverity.CRITICAL: "\033[91m",  # Red
            PatientSeverity.SEVERE: "\033[93m",    # Yellow
            PatientSeverity.MODERATE: "\033[92m",  # Green
            PatientSeverity.MILD: "\033[94m",      # Blue
            PatientSeverity.MINOR: "\033[90m"      # Gray
        }
        return colors.get(severity, "\033[0m")
    
    def get_state_symbol(self, state: PatientState) -> str:
        """Get symbol for patient state"""
        symbols = {
            PatientState.ARRIVING: "→",
            PatientState.WAITING_REGISTRATION: "R",
            PatientState.REGISTERING: "R*",
            PatientState.WAITING_TRIAGE: "T",
            PatientState.TRIAGING: "T*",
            PatientState.WAITING_TREATMENT: "W",
            PatientState.IN_TREATMENT: "TX",
            PatientState.DECEASED: "☠",
            PatientState.DISCHARGED: "✓"
        }
        return symbols.get(state, "?")
    
    def display_header(self):
        """Display simulation header"""
        cfg = self.simulator.config
        time_str = self.format_time(self.simulator.current_time)
        end_time = self.format_time(cfg.simulation_duration)
        
        print(f"╔{'═' * 78}╗")
        print(f"║ {'EMERGENCY ROOM DISCRETE EVENT SIMULATION':<76} ║")
        print(f"║ Time: {time_str} / {end_time} {' ' * 56} ║")
        print(f"╠{'═' * 78}╣")
    
    def display_resources(self):
        """Display resource status"""
        print("║ RESOURCES:")
        
        resource_rows = []
        for res_type, pool in self.simulator.resources.items():
            name = res_type.name.replace('_', ' ').title()
            available = pool.available_count
            total = pool.total_count
            in_use = pool.in_use_count
            waiting = pool.get_wait_queue_length()
            
            # Create bar
            bar_len = 20
            filled = int((in_use / total) * bar_len) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            
            status = f"{available}/{total}"
            wait_str = f"Wait:{waiting}" if waiting > 0 else "OK"
            
            resource_rows.append(f"  {name:<12}: [{bar}] {status:<8} {wait_str}")
        
        # Print in 2 columns
        for i in range(0, len(resource_rows), 2):
            left = resource_rows[i]
            right = resource_rows[i+1] if i+1 < len(resource_rows) else ""
            print(f"║ {left:<38} {right:<38} ║")
        
        print(f"╠{'═' * 78}╣")
    
    def display_doctors(self):
        """Display doctor status"""
        print("║ DOCTORS:")
        
        for doc_id in range(self.simulator.config.num_doctors):
            on_break = self.simulator.doctor_breaks.get(doc_id, False)
            patients_treated = self.simulator.stats.patients_per_doctor.get(doc_id, 0)
            
            status = "BREAK" if on_break else "ACTIVE"
            status_color = "\033[93m" if on_break else "\033[92m"
            
            print(f"║   Dr.{doc_id+1}: {status_color}{status:<6}\033[0m | Patients: {patients_treated:<3} | "
                  f"Next break: {self.format_time(self.simulator.doctor_next_break.get(doc_id, 0))} ║")
        
        print(f"╠{'═' * 78}╣")
    
    def display_patients(self):
        """Display patient queues"""
        print("║ PATIENT QUEUES:")
        
        # Waiting counts by severity
        waiting_reg = len(self.simulator.patients_waiting_registration)
        waiting_triage = len(self.simulator.patients_waiting_triage)
        waiting_treatment = len(self.simulator.patients_waiting_treatment)
        in_treatment = len(self.simulator.patients_in_treatment)
        
        print(f"║   Registration: {waiting_reg:<3} | Triage: {waiting_triage:<3} | "
              f"Treatment: {waiting_treatment:<3} | In Treatment: {in_treatment:<3} ║")
        
        # Recent patients
        print("║ RECENT PATIENTS:")
        
        recent = sorted(self.simulator.patients.values(), 
                       key=lambda p: p.arrival_time, reverse=True)[:5]
        
        for patient in recent:
            color = self.get_severity_color(patient.severity)
            state_symbol = self.get_state_symbol(patient.state)
            severity_name = patient.severity.name[:3]
            
            time_in_er = self.simulator.current_time - patient.arrival_time
            
            print(f"║   {color}#{patient.id:<4} [{severity_name}] {state_symbol:<4} "
                  f"Time: {time_in_er:>4}m\033[0m")
        
        print(f"╠{'═' * 78}╣")
    
    def display_statistics(self):
        """Display running statistics"""
        print("║ STATISTICS:")
        
        stats = self.simulator.stats
        
        print(f"║   Arrivals: {stats.total_arrivals:<5} | Discharged: {stats.total_discharged:<5} | "
              f"Deceased: {stats.total_deceased:<5} ║")
        
        print(f"║   Death Causes: No Treatment: {stats.deaths_no_treatment:<3} | "
              f"Resource: {stats.deaths_resource_shortage:<3} | Natural: {stats.deaths_natural:<3} ║")
        
        print(f"║   Critical Saved: {stats.critical_patients_saved:<3} | Critical Lost: {stats.critical_patients_lost:<3} | "
              f"Doctor Breaks: {stats.doctor_breaks_taken:<4} ║")
        
        # Wait time averages
        if stats.wait_times_registration:
            avg_reg = sum(stats.wait_times_registration) / len(stats.wait_times_registration)
            avg_triage = sum(stats.wait_times_triage) / len(stats.wait_times_triage) if stats.wait_times_triage else 0
            avg_treat = sum(stats.wait_times_treatment) / len(stats.wait_times_treatment) if stats.wait_times_treatment else 0
            
            print(f"║   Avg Wait: Reg: {avg_reg:.1f}m | Triage: {avg_triage:.1f}m | "
                  f"Treatment: {avg_treat:.1f}m ║")
        
        # Buildup periods
        if stats.buildup_periods:
            print(f"║   Buildup Periods: {len(stats.buildup_periods)} recorded ║")
        
        print(f"╚{'═' * 78}╝")
    
    def display(self):
        """Display full simulation state"""
        self.clear_screen()
        self.display_header()
        self.display_resources()
        self.display_doctors()
        self.display_patients()
        self.display_statistics()
    
    def run_display_loop(self):
        """Run the display loop in a separate thread"""
        while self.running:
            if not self.simulator.paused:
                self.display()
            time.sleep(self.simulator.config.refresh_interval)
    
    def stop(self):
        """Stop the display loop"""
        self.running = False


class ERGuiApp:
    """Tkinter GUI for the ER simulation with bubble-based visualization"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulator = ERSimulator(config)
        self.root = tk.Tk()
        self.root.title("🏥 ER Simulation")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0d1117")

        self.speed_multiplier = tk.DoubleVar(value=1.0)
        self.fast_forward = tk.BooleanVar(value=False)
        self.running = False
        self.start_time = None
        self.elapsed_real_time = 0.0
        self.last_frame_time = time.time()
        self.frame_delta = 0.0
        self.sim_step_accumulator = 0.0
        self.sim_steps_per_second = 6.0
        self.bubble_smoothing = 0.08

        self.zone_bubbles: Dict[str, Dict] = {}
        self.flow_particles: List[Dict] = []
        self.dean_event_cooldown = 0.0
        self.pending_dean_event: Optional[Dict[str, Any]] = None
        self.dean_decision_log: List[Dict[str, Any]] = []
        self.summary_shown: bool = False
        self.decision_pause_active: bool = False
        self.pre_decision_running: bool = False

        self._build_styles()
        self._build_layout()
        self._build_canvas_flow()
        self._build_legend()
        self._build_resource_panel()
        self._build_stats_panel()
        self._build_controls()
        self._refresh_ui()
        self._update_loop()

    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#0d1117")
        style.configure("TLabel", background="#0d1117", foreground="#e6edf3")
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#58a6ff")
        style.configure("SubHeader.TLabel", font=("Helvetica", 12, "bold"), foreground="#8b949e")
        style.configure("TButton", font=("Helvetica", 11, "bold"), background="#238636", foreground="white")
        style.configure("TScale", background="#0d1117")
        style.map("TButton", background=[("active", "#2ea043")])

    def _build_layout(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))

        self.right_frame = ttk.Frame(self.main_frame, width=320)
        self.right_frame.pack(side="right", fill="y")
        self.right_frame.pack_propagate(False)

    def _build_canvas_flow(self):
        header = ttk.Label(self.left_frame, text="🏥 Emergency Room Patient Flow", style="Header.TLabel")
        header.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(self.left_frame, width=1150, height=780, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Zone centers for bubble layout (spaced out for clarity)
        self.zone_centers = {
            "arrival": (140, 140),
            "registration": (360, 140),
            "triage": (600, 140),
            "waiting": (480, 360),
            "treatment": (860, 200),
            "discharge": (860, 460),
            "deceased": (860, 680),
        }

        self.zone_bubble_base = 45
        self.zone_bubble_max = 90
        self.zone_bubble_hard_max = 100

        # Flow paths
        self.flow_paths = [
            ("arrival", "registration"),
            ("registration", "triage"),
            ("triage", "waiting"),
            ("triage", "treatment"),
            ("waiting", "treatment"),
            ("treatment", "discharge"),
        ]

        # Draw flow lines with arrows
        for from_zone, to_zone in self.flow_paths:
            x1, y1 = self.zone_centers[from_zone]
            x2, y2 = self.zone_centers[to_zone]
            self.canvas.create_line(x1, y1, x2, y2, fill="#30363d", width=4, dash=(8, 6), 
                                   arrow=tk.LAST, arrowshape=(10, 12, 4), tags="flow_line")

        # Zone configurations
        zone_configs = {
            "arrival": {"color": "#21262d", "border": "#484f58", "label": "🚶 ARRIVAL", "icon": "🚶"},
            "registration": {"color": "#1c3a5e", "border": "#58a6ff", "label": "📝 REGISTRATION", "icon": "📝"},
            "triage": {"color": "#3d2c00", "border": "#f2cc60", "label": "🔍 TRIAGE", "icon": "🔍"},
            "waiting": {"color": "#2d2d00", "border": "#d4a72c", "label": "⏳ WAITING", "icon": "⏳"},
            "treatment": {"color": "#0d2818", "border": "#3fb950", "label": "🏥 TREATMENT", "icon": "🏥"},
            "discharge": {"color": "#0d2818", "border": "#3fb950", "label": "✅ DISCHARGED", "icon": "✅"},
            "deceased": {"color": "#2d0d0d", "border": "#f85149", "label": "☠ DECEASED", "icon": "☠"},
        }

        for name, center in self.zone_centers.items():
            cfg = zone_configs.get(name, {"color": "#21262d", "border": "#484f58", "label": name.upper(), "icon": ""})
            x, y = center

            # Draw zone background circle
            r = 80
            self.canvas.create_oval(x - r, y - r, x + r, y + r, 
                                   fill=cfg["color"], outline=cfg["border"], width=4, 
                                   tags=f"zone_{name}")

            # Create bubble for patient count (initially hidden)
            bubble = self.canvas.create_oval(x - 25, y - 25, x + 25, y + 25,
                                            fill="#58a6ff", outline="#ffffff", width=3, 
                                            tags=f"bubble_{name}", state="hidden")
            label_text = self.canvas.create_text(x, y + 10, text=cfg["label"], fill=cfg["border"],
                                                font=("Helvetica", 11, "bold"),
                                                tags=f"label_{name}", state="hidden")
            count_text = self.canvas.create_text(x, y - 14, text="0", fill="#0d1117", 
                                                font=("Helvetica", 16, "bold"),
                                                tags=f"count_{name}", state="hidden")

            self.zone_bubbles[name] = {
                "bubble": bubble,
                "label_text": label_text,
                "count_text": count_text,
                "center": center,
                "count": 0,
                "color": cfg["border"],
                "current_color": cfg["border"],
                "color_cooldown": 0.0,
                "current_radius": self.zone_bubble_base,
                "target_radius": self.zone_bubble_base,
                "visible": False,
                "display_center": center,
                "label_color": cfg["border"]
            }

    def _build_legend(self):
        legend_frame = ttk.Frame(self.right_frame)
        legend_frame.pack(fill="x", pady=(0, 10))

        header = ttk.Label(legend_frame, text="Patient Severity", style="SubHeader.TLabel")
        header.pack(anchor="w")

        severities = [
            ("Critical", "#f85149"),
            ("Severe", "#f2cc60"),
            ("Moderate", "#56d364"),
            ("Mild", "#58a6ff"),
            ("Minor", "#8b949e"),
        ]

        for name, color in severities:
            row = ttk.Frame(legend_frame)
            row.pack(anchor="w", pady=3)
            canvas = tk.Canvas(row, width=24, height=24, bg="#0d1117", highlightthickness=0)
            canvas.pack(side="left", padx=(0, 8))
            canvas.create_oval(3, 3, 21, 21, fill=color, outline="")
            ttk.Label(row, text=name, font=("Helvetica", 10)).pack(side="left")

    def _build_resource_panel(self):
        header = ttk.Label(self.right_frame, text="📦 Resources", style="Header.TLabel")
        header.pack(anchor="w", pady=(10, 5))

        self.resource_frame = ttk.Frame(self.right_frame)
        self.resource_frame.pack(fill="x", pady=5)

        self.resource_labels: Dict[ResourceType, ttk.Label] = {}
        for res_type in ResourceType:
            label = ttk.Label(self.resource_frame, text="", font=("Helvetica", 10))
            label.pack(anchor="w", pady=2)
            self.resource_labels[res_type] = label

    def _build_stats_panel(self):
        header = ttk.Label(self.right_frame, text="📊 Live Stats", style="Header.TLabel")
        header.pack(anchor="w", pady=(15, 5))

        self.stats_frame = ttk.Frame(self.right_frame)
        self.stats_frame.pack(fill="x", pady=5)

        self.stats_labels = {
            "time": ttk.Label(self.stats_frame, text="⏰ Time: 00:00", font=("Helvetica", 12, "bold")),
            "arrivals": ttk.Label(self.stats_frame, text="📥 Arrivals: 0"),
            "discharged": ttk.Label(self.stats_frame, text="✅ Discharged: 0"),
            "deceased": ttk.Label(self.stats_frame, text="☠ Deceased: 0"),
            "critical": ttk.Label(self.stats_frame, text="🔴 Critical: 0 saved / 0 lost"),
            "waits": ttk.Label(self.stats_frame, text="⏱ Avg Wait: 0m"),
            "queues": ttk.Label(self.stats_frame, text=""),
        }
        for label in self.stats_labels.values():
            label.pack(anchor="w", pady=2)

    def _build_controls(self):
        header = ttk.Label(self.right_frame, text="🎮 Controls", style="Header.TLabel")
        header.pack(anchor="w", pady=(15, 5))

        controls = ttk.Frame(self.right_frame)
        controls.pack(fill="x", pady=5)

        btn_frame = ttk.Frame(controls)
        btn_frame.pack(fill="x", pady=5)

        self.start_button = ttk.Button(btn_frame, text="▶ Start", command=self.toggle_start, width=12)
        self.start_button.pack(side="left", padx=(0, 5))

        self.step_button = ttk.Button(btn_frame, text="⏭ Step", command=self.step_once, width=12)
        self.step_button.pack(side="left")

        self.fast_button = ttk.Checkbutton(controls, text="⚡ Fast Forward", variable=self.fast_forward, command=self.toggle_fast)
        self.fast_button.pack(anchor="w", pady=5)

        ttk.Label(controls, text="Speed:").pack(anchor="w")
        self.speed_scale = ttk.Scale(controls, from_=0.5, to=5.0, orient="horizontal",
                                     variable=self.speed_multiplier)
        self.speed_scale.pack(fill="x", pady=5)

        self.reset_button = ttk.Button(controls, text="🔄 Reset", command=self.reset_simulation, width=25)
        self.reset_button.pack(fill="x", pady=10)

        info = ttk.Label(controls, text="💡 Bubbles grow with patient count\n   Particles show patient flow", 
                        font=("Helvetica", 9), foreground="#8b949e")
        info.pack(anchor="w", pady=10)

    def _severity_color(self, severity: PatientSeverity) -> str:
        return {
            PatientSeverity.CRITICAL: "#f85149",
            PatientSeverity.SEVERE: "#f2cc60",
            PatientSeverity.MODERATE: "#56d364",
            PatientSeverity.MILD: "#58a6ff",
            PatientSeverity.MINOR: "#8b949e",
        }[severity]

    def _update_bubbles(self, zone_patients: Dict[str, List[Patient]]):
        zone_mapping = {
            "arrival": [],
            "registration": [],
            "triage": [],
            "waiting": [],
            "treatment": [],
            "discharge": [],
            "deceased": [],
        }

        for patient in self.simulator.patients.values():
            if patient.state in [PatientState.ARRIVING, PatientState.WAITING_REGISTRATION]:
                zone_mapping["arrival"].append(patient)
            elif patient.state == PatientState.REGISTERING:
                zone_mapping["registration"].append(patient)
            elif patient.state in [PatientState.WAITING_TRIAGE, PatientState.TRIAGING]:
                zone_mapping["triage"].append(patient)
            elif patient.state == PatientState.WAITING_TREATMENT:
                zone_mapping["waiting"].append(patient)
            elif patient.state == PatientState.IN_TREATMENT:
                zone_mapping["treatment"].append(patient)
            elif patient.state == PatientState.WAITING_DISCHARGE:
                zone_mapping["discharge"].append(patient)
            elif patient.state == PatientState.DISCHARGED:
                zone_mapping["discharge"].append(patient)
            elif patient.state == PatientState.DECEASED:
                zone_mapping["deceased"].append(patient)

        for zone_name, patients in zone_mapping.items():
            if zone_name not in self.zone_bubbles:
                continue

            bubble_info = self.zone_bubbles[zone_name]
            count = len(patients)
            bubble_info["count"] = count

            if count == 0:
                bubble_info["target_radius"] = self.zone_bubble_base
                bubble_info["visible"] = True
            else:
                import math
                target_radius = self.zone_bubble_base + math.log(count + 1) * 18
                bubble_info["target_radius"] = min(self.zone_bubble_max, target_radius)
                bubble_info["visible"] = True

            # Smooth radius transitions with hard cap
            current = bubble_info["current_radius"]
            target = bubble_info["target_radius"]
            bubble_info["current_radius"] = current + (target - current) * self.bubble_smoothing
            radius = min(bubble_info["current_radius"], self.zone_bubble_hard_max)

            # Dominant severity with cooldown to avoid flicker
            desired_color = bubble_info["current_color"]
            if patients:
                severity_counts = {}
                for p in patients:
                    sev = p.severity
                    severity_counts[sev] = severity_counts.get(sev, 0) + 1
                dominant = max(severity_counts, key=severity_counts.get)
                desired_color = self._severity_color(dominant)

            if bubble_info["current_color"] != desired_color and bubble_info["color_cooldown"] <= 0:
                bubble_info["current_color"] = desired_color
                bubble_info["color_cooldown"] = 1.8
            else:
                bubble_info["color_cooldown"] = max(0.0, bubble_info["color_cooldown"] - self.frame_delta)

            # Contrast-aware text colors
            def _label_color(hex_color: str) -> str:
                hex_color = hex_color.lstrip("#")
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                return "#0d1117" if luminance > 160 else "#f0f6fc"

            label_color = _label_color(bubble_info["current_color"])

            cx, cy = bubble_info["center"]
            min_gap = 28
            adjusted_x, adjusted_y = bubble_info["display_center"]
            target_x, target_y = cx, cy
            for other_name, other_info in self.zone_bubbles.items():
                if other_name == zone_name:
                    continue
                ox, oy = other_info["display_center"]
                dx = target_x - ox
                dy = target_y - oy
                dist_sq = dx * dx + dy * dy
                min_dist = radius + other_info["current_radius"] + min_gap
                if dist_sq < min_dist * min_dist and dist_sq > 0:
                    dist = dist_sq ** 0.5
                    push = (min_dist - dist) / dist
                    target_x += dx * push
                    target_y += dy * push

            # Smoothly move bubble toward non-colliding target
            adjusted_x += (target_x - adjusted_x) * 0.2
            adjusted_y += (target_y - adjusted_y) * 0.2
            bubble_info["display_center"] = (adjusted_x, adjusted_y)

            self.canvas.coords(bubble_info["bubble"],
                             adjusted_x - radius, adjusted_y - radius,
                             adjusted_x + radius, adjusted_y + radius)

            if bubble_info["visible"]:
                self.canvas.itemconfig(bubble_info["bubble"], fill=bubble_info["current_color"], state="normal")
                self.canvas.itemconfig(bubble_info["count_text"], text=str(count), state="normal", fill=label_color)
                self.canvas.itemconfig(bubble_info["label_text"], state="normal", fill=label_color)
                font_size = min(26, 12 + count // 4)
                self.canvas.itemconfig(bubble_info["count_text"], font=("Helvetica", font_size, "bold"))
                self.canvas.coords(bubble_info["count_text"], adjusted_x, adjusted_y - radius * 0.25)
                self.canvas.coords(bubble_info["label_text"], adjusted_x, adjusted_y + radius * 0.25)
            else:
                self.canvas.itemconfig(bubble_info["bubble"], state="hidden")
                self.canvas.itemconfig(bubble_info["count_text"], state="hidden")
                self.canvas.itemconfig(bubble_info["label_text"], state="hidden")

    def _spawn_flow_particle(self, from_zone: str, to_zone: str):
        if from_zone not in self.zone_centers or to_zone not in self.zone_centers:
            return

        x1, y1 = self.zone_centers[from_zone]
        x2, y2 = self.zone_centers[to_zone]

        particle = {
            "id": self.canvas.create_oval(x1 - 4, y1 - 4, x1 + 4, y1 + 4,
                                         fill="#58a6ff", outline="#ffffff", width=1),
            "x": x1, "y": y1,
            "target_x": x2, "target_y": y2,
            "progress": 0.0,
            "speed": 0.9 + (self.speed_multiplier.get() * 0.15)
        }
        self.flow_particles.append(particle)

    def _update_flow_particles(self):
        completed = []
        for particle in self.flow_particles:
            delta = max(0.016, self.frame_delta)
            particle["progress"] += 0.04 * particle["speed"] * (delta * 60)
            if particle["progress"] >= 1.0:
                completed.append(particle)
                self.canvas.delete(particle["id"])
            else:
                t = particle["progress"]
                x = particle["x"] + (particle["target_x"] - particle["x"]) * t
                y = particle["y"] + (particle["target_y"] - particle["y"]) * t
                self.canvas.coords(particle["id"], x - 4, y - 4, x + 4, y + 4)

        for p in completed:
            self.flow_particles.remove(p)

    def _trigger_flow_animations(self):
        import random
        for from_zone, to_zone in self.flow_paths:
            from_count = self.zone_bubbles.get(from_zone, {}).get("count", 0)
            if from_count > 0 and random.random() < 0.35:
                self._spawn_flow_particle(from_zone, to_zone)

    def _update_resources(self):
        for res_type, pool in self.simulator.resources.items():
            label = self.resource_labels[res_type]
            utilization = pool.in_use_count / pool.total_count * 100 if pool.total_count > 0 else 0
            
            if utilization > 90:
                color = "#f85149"
            elif utilization > 60:
                color = "#f2cc60"
            else:
                color = "#56d364"
            
            label.config(text=f"{res_type.name.title():<12}: {pool.available_count}/{pool.total_count} ({utilization:.0f}%)",
                        foreground=color)

    def _update_stats(self):
        stats = self.simulator.stats
        time_str = format_sim_time(self.simulator.current_time)
        
        self.stats_labels["time"].config(text=f"⏰ Time: {time_str}")
        self.stats_labels["arrivals"].config(text=f"📥 Arrivals: {stats.total_arrivals}")
        self.stats_labels["discharged"].config(text=f"✅ Discharged: {stats.total_discharged}")
        
        deceased_color = "#f85149" if stats.total_deceased > 0 else "#8b949e"
        self.stats_labels["deceased"].config(text=f"☠ Deceased: {stats.total_deceased}", foreground=deceased_color)
        
        self.stats_labels["critical"].config(
            text=f"🔴 Critical: {stats.critical_patients_saved} saved / {stats.critical_patients_lost} lost"
        )
        
        avg_wait = 0
        if stats.wait_times_treatment:
            avg_wait = sum(stats.wait_times_treatment) / len(stats.wait_times_treatment)
        self.stats_labels["waits"].config(text=f"⏱ Avg Treatment Wait: {avg_wait:.1f}m")

        self.stats_labels["queues"].config(
            text=(
                f"📋 Queues: Reg={len(self.simulator.patients_waiting_registration)} | "
                f"Triage={len(self.simulator.patients_waiting_triage)} | "
                f"Treat={len(self.simulator.patients_waiting_treatment)} | "
                f"InTx={len(self.simulator.patients_in_treatment)}"
            )
        )

    def _maybe_trigger_dean_event(self):
        """Check conditions and trigger dean decisions"""
        if not self.running or self.simulator.simulation_complete or self.decision_pause_active:
            return

        if self.dean_event_cooldown > 0:
            self.dean_event_cooldown = max(0.0, self.dean_event_cooldown - self.frame_delta)
            return

        if self.pending_dean_event:
            return

        waiting = len(self.simulator.patients_waiting_treatment)
        reg_wait = len(self.simulator.patients_waiting_registration)
        triage_wait = len(self.simulator.patients_waiting_triage)
        critical_waiting = len([p for p in self.simulator.patients.values() if p.state == PatientState.WAITING_TREATMENT and p.severity == PatientSeverity.CRITICAL])

        import random
        if waiting > 10 or critical_waiting > 3 or reg_wait > 8 or triage_wait > 8:
            if random.random() < 0.02:
                self._trigger_dean_event()
        elif random.random() < 0.005:
            self._trigger_dean_event()

    def _trigger_dean_event(self):
        """Pick and display a dean of medicine decision event"""
        import random

        events = [
            {
                "title": "Bed Allocation Crisis",
                "description": "Two departments request 5 extra beds: ICU (critical) and General Ward (moderate). Only one can be prioritized. The ICU head and Ward supervisor are both in your office demanding immediate action.",
                "options": [
                    {"label": "Prioritize ICU - Give all 5 beds to critical care", 
                     "effects": {"beds": 5, "critical_bonus": 0.08, "general_penalty": 0.08, "description": "ICU receives 5 beds. Critical patients get faster treatment. General Ward patients wait longer."},
                     "impact_log": "ICU prioritized: 5 beds added, critical deaths reduced, general ward backs up"},
                    {"label": "Prioritize General Ward - Balance patient flow", 
                     "effects": {"beds": 5, "critical_penalty": 0.05, "general_bonus": 0.08, "description": "General Ward receives 5 beds. Moderate/severe patients flow faster. Critical care slightly strained."},
                     "impact_log": "General Ward prioritized: 5 beds added, moderate/severe deaths reduced, ICU strained"},
                    {"label": "DENY BOTH - 'Figure it out yourselves' (WORST CHOICE)", 
                     "effects": {"beds": -3, "critical_penalty": 0.15, "general_penalty": 0.15, "treatment_delay": 0.15, "description": "You refuse both requests. Hospital morale crashes. 3 beds are lost to administrative chaos. Both departments suffer severely. Death rates spike across all severities."},
                     "impact_log": "CATASTROPHE: Dean denied both bed requests. 3 beds lost, critical death rate +15%, severe death rate +15%, treatment slowed 15%. Staff morale collapsed."},
                ],
            },
            {
                "title": "Staff Burnout Report",
                "description": "Nurse fatigue is at dangerous levels. Multiple nurses have called in sick. Triage is backed up. You must make a decision.",
                "options": [
                    {"label": "Shift 2 nurses from triage to treatment - prioritize critical patients", 
                     "effects": {"nurses_treatment": 2, "triage_delay": 0.12, "description": "2 nurses moved to treatment. Critical patients treated faster. Triage waits increase."},
                     "impact_log": "Staff reallocation: 2 nurses to treatment, triage waits increased 12%"},
                    {"label": "Keep triage staffed - maintain patient intake flow", 
                     "effects": {"nurses_triage": 0, "treatment_delay": 0.12, "description": "Triage stays staffed. New patients processed faster. Treatment waits increase."},
                     "impact_log": "Triage maintained: new patient intake stable, treatment waits increased 12%"},
                    {"label": "FIRE 2 NURSES - 'We can't afford lazy staff' (WORST CHOICE)", 
                     "effects": {"nurses": -2, "triage_delay": 0.25, "treatment_delay": 0.25, "critical_penalty": 0.12, "description": "You fire 2 nurses for 'laziness'. The remaining staff is horrified. 2 nurses removed from pool. ALL wait times explode. Critical death rate increases."},
                     "impact_log": "CATASTROPHE: Dean fired 2 nurses. Nurses removed: -2, triage delay +25%, treatment delay +25%, critical death rate +12%. Remaining staff morale crushed."},
                ],
            },
            {
                "title": "Imaging Jam",
                "description": "X-ray queue is backing up badly. Patients needing imaging are stuck for hours. You have $50k in the discretionary fund.",
                "options": [
                    {"label": "Lease Mobile X-ray unit immediately - $40k", 
                     "effects": {"xray": 1, "treatment_speed": 0.08, "description": "Mobile X-ray leased. Imaging capacity increased. Treatment speeds improve."},
                     "impact_log": "Mobile X-ray leased: +1 X-ray machine, treatment speed improved 8%"},
                    {"label": "Hire temp radiologist - $30k, faster interpretation", 
                     "effects": {"lab_delay": -0.05, "treatment_speed": 0.03, "description": "Temp radiologist hired. Faster reading of images. Lab delays reduced."},
                     "impact_log": "Temp radiologist hired: lab delays reduced 5%, treatment speed improved 3%"},
                    {"label": "POCKET THE MONEY - 'Hospital finances need it' (WORST CHOICE)", 
                     "effects": {"xray": -1, "critical_penalty": 0.10, "general_penalty": 0.10, "treatment_delay": 0.20, "description": "You pocket the $50k budget. The X-ray machine breaks from overuse with no replacement. Death rates spike. Treatment grinds to a halt."},
                     "impact_log": "CATASTROPHE: Dean embezzled imaging budget. X-ray machine lost, critical death rate +10%, severe death rate +10%, treatment delay +20%. Hospital investigation pending."},
                ],
            },
            {
                "title": "Supply Chain Alert",
                "description": "PPE shipments delayed by 3 weeks. Masks and gloves running critically low. Multiple suppliers available but at different costs.",
                "options": [
                    {"label": "Expedite from premium supplier - $25k", 
                     "effects": {"critical_bonus": 0.05, "general_bonus": 0.03, "description": "Premium PPE expedited. Staff protected. Infection rates drop. Death rates reduced."},
                     "impact_log": "Premium PPE expedited: critical death rate -5%, severe death rate -3%"},
                    {"label": "Accept delay - ration remaining PPE", 
                     "effects": {"critical_penalty": 0.04, "general_penalty": 0.04, "description": "PPE rationed. Staff at slightly higher risk. Death rates increase slightly."},
                     "impact_log": "PPE rationed: critical death rate +4%, severe death rate +4%"},
                    {"label": "BAN ALL PPE - 'It's just a cold' (WORST CHOICE)", 
                     "effects": {"critical_penalty": 0.25, "general_penalty": 0.20, "nurses": -1, "doctors": -1, "description": "You ban PPE use, claiming it's unnecessary. Staff infections spike. 1 nurse and 1 doctor quit in protest. Death rates explode."},
                     "impact_log": "CATASTROPHE: Dean banned PPE. Critical death rate +25%, severe death rate +20%. 1 nurse and 1 doctor quit. Hospital infection outbreak begins."},
                ],
            },
            {
                "title": "Temporary Ward Proposal",
                "description": "A temporary ward can open with 6 beds but needs 1 doctor assigned. The union threatens action if you force overtime.",
                "options": [
                    {"label": "Open Temporary Ward - assign 1 doctor", 
                     "effects": {"beds": 6, "doctors": -1, "general_bonus": 0.05, "description": "Temporary ward opens. 6 beds added. 1 doctor reassigned. General patient flow improves."},
                     "impact_log": "Temporary ward opened: +6 beds, -1 doctor, severe death rate -5%"},
                    {"label": "Decline - keep all doctors in main ER", 
                     "effects": {"general_penalty": 0.05, "description": "Ward not opened. Doctors stay in main ER. General ward slightly strained."},
                     "impact_log": "Ward declined: severe death rate +5%"},
                    {"label": "FIRE 2 DOCTORS - 'We need to cut costs' (WORST CHOICE)", 
                     "effects": {"doctors": -2, "beds": -4, "critical_penalty": 0.20, "general_penalty": 0.25, "description": "You fire 2 doctors to 'cut costs'. 4 beds become unusable without staff. All death rates spike catastrophically."},
                     "impact_log": "CATASTROPHE: Dean fired 2 doctors. Doctors removed: -2, beds lost: -4, critical death rate +20%, severe death rate +25%. ER in crisis mode."},
                ],
            },
            {
                "title": "Ventilator Shortage",
                "description": "Only one ventilator shipment arrives. Trauma and Respiratory units both claim urgent need. Patients are dying.",
                "options": [
                    {"label": "Give to Trauma - immediate life-saving priority", 
                     "effects": {"ventilators": 1, "critical_bonus": 0.08, "general_penalty": 0.05, "description": "Ventilator to Trauma. Critical patients survive more. Respiratory patients slightly worse off."},
                     "impact_log": "Ventilator to Trauma: +1 ventilator, critical death rate -8%, severe death rate +5%"},
                    {"label": "Give to Respiratory - long-term patient stability", 
                     "effects": {"ventilators": 1, "critical_penalty": 0.05, "general_bonus": 0.08, "description": "Ventilator to Respiratory. Severe patients stabilize better. Some critical patients lost."},
                     "impact_log": "Ventilator to Respiratory: +1 ventilator, critical death rate +5%, severe death rate -8%"},
                    {"label": "SELL THE VENTILATOR - 'Fund the research wing' (WORST CHOICE)", 
                     "effects": {"ventilators": -2, "critical_penalty": 0.30, "description": "You sell the ventilator AND another one to fund research. 2 ventilators lost. Critical death rate explodes. Patients suffocate."},
                     "impact_log": "CATASTROPHE: Dean sold 2 ventilators for research funding. Ventilators: -2, critical death rate +30%. Patients dying from lack of ventilation."},
                ],
            },
            {
                "title": "Critical Patient Surge",
                "description": "Ambulance dispatch reports 15 critical patients incoming from a mass casualty event. You have 30 minutes to prepare.",
                "options": [
                    {"label": "Call in all available staff - overtime authorized", 
                     "effects": {"doctors": 2, "nurses": 4, "critical_bonus": 0.10, "description": "All staff called in. 2 doctors and 4 nurses added. Critical survival rate improves dramatically."},
                     "impact_log": "Surge response: +2 doctors, +4 nurses, critical death rate -10%"},
                    {"label": "Prioritize existing patients - divert ambulances", 
                     "effects": {"critical_penalty": 0.15, "general_bonus": 0.05, "description": "Existing patients prioritized. Ambulances diverted. Current critical patients saved. Incoming surge patients suffer."},
                     "impact_log": "Ambulance diversion: critical death rate +15%, severe death rate -5%"},
                    {"label": "CLOSE THE ER - 'We're not equipped' (WORST CHOICE)", 
                     "effects": {"critical_penalty": 0.40, "general_penalty": 0.35, "doctors": -1, "nurses": -2, "description": "You close the ER doors. All incoming patients die. 1 doctor and 2 nurses quit in disgust. Death rates skyrocket."},
                     "impact_log": "CATASTROPHE: Dean closed the ER. Critical death rate +40%, severe death rate +35%. 1 doctor and 2 nurses quit. Hospital reputation destroyed."},
                ],
            },
            {
                "title": "Lab Bottleneck",
                "description": "Lab results are delayed 4+ hours. Doctors can't proceed with treatment. 1 nurse can be temporarily assigned to lab.",
                "options": [
                    {"label": "Assign 1 nurse to lab - speed up results", 
                     "effects": {"lab": 1, "treatment_speed": 0.10, "description": "Nurse assigned to lab. Results come faster. Treatment speeds improve."},
                     "impact_log": "Lab assisted: +1 lab station, treatment speed +10%"},
                    {"label": "Keep nurses in treatment - patients need care", 
                     "effects": {"lab_delay": 0.08, "description": "Nurses stay in treatment. Lab stays slow. Moderate patients wait longer."},
                     "impact_log": "Nurses kept: lab delays +8%"},
                    {"label": "CLOSE THE LAB - 'We'll eyeball it' (WORST CHOICE)", 
                     "effects": {"lab": -2, "treatment_delay": 0.30, "critical_penalty": 0.20, "description": "You close 2 lab stations, claiming 'doctors can guess'. Lab capacity halved. Treatment grinds to a halt. Critical death rate spikes."},
                     "impact_log": "CATASTROPHE: Dean closed 2 lab stations. Lab stations: -2, treatment delay +30%, critical death rate +20%. Patients treated blindly."},
                ],
            },
        ]

        event = random.choice(events)
        self.pending_dean_event = event

        self._show_dean_popup(event)

    def _apply_dean_decision(self, event: Dict[str, Any], option: Dict[str, Any]):
        """Apply effects of decision to simulation"""
        effects = option.get("effects", {})
        stats = self.simulator.stats
        
        # Log the decision with detailed impact
        impact_log_text = option.get("impact_log", f"Decision: {option['label']}")
        description_text = effects.get("description", "")
        
        record = {
            "time": format_sim_time(self.simulator.current_time),
            "event": event["title"],
            "choice": option["label"],
            "effects": effects,
            "impact_log": impact_log_text,
            "description": description_text
        }
        self.dean_decision_log.append(record)
        stats.dean_decisions.append(record)
        
        # Add the dramatic impact log
        stats.dean_impact_log.append(f"[{format_sim_time(self.simulator.current_time)}] {impact_log_text}")

        # Handle resource additions/removals
        if "beds" in effects:
            delta = effects["beds"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.BED].resources)
                    self.simulator.resources[ResourceType.BED].resources[new_id] = Resource(id=new_id, type=ResourceType.BED)
                    self.simulator.resources[ResourceType.BED].available_queue.append(new_id)
                stats.dean_impact_log.append(f"  → {delta} beds added to hospital")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.BED].available_queue))
                for _ in range(remove_count):
                    bed_id = self.simulator.resources[ResourceType.BED].available_queue.pop()
                    self.simulator.resources[ResourceType.BED].resources.pop(bed_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} beds LOST from hospital")

        if "doctors" in effects:
            delta = effects["doctors"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.DOCTOR].resources)
                    self.simulator.resources[ResourceType.DOCTOR].resources[new_id] = Resource(id=new_id, type=ResourceType.DOCTOR)
                    self.simulator.resources[ResourceType.DOCTOR].available_queue.append(new_id)
                    self.simulator.doctor_breaks[new_id] = False
                    self.simulator.doctor_next_break[new_id] = self.simulator.current_time + self.simulator.config.doctor_break_interval
                    self.simulator.doctor_shift_end[new_id] = self.simulator.current_time + self.simulator.config.doctor_shift_duration
                stats.dean_impact_log.append(f"  → {delta} doctor(s) added to staff")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.DOCTOR].available_queue))
                for _ in range(remove_count):
                    doc_id = self.simulator.resources[ResourceType.DOCTOR].available_queue.pop()
                    self.simulator.resources[ResourceType.DOCTOR].resources.pop(doc_id, None)
                    self.simulator.doctor_breaks.pop(doc_id, None)
                    self.simulator.doctor_next_break.pop(doc_id, None)
                    self.simulator.doctor_shift_end.pop(doc_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} doctor(s) FIRED/QUIT")

        if "nurses" in effects:
            delta = effects["nurses"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.NURSE].resources)
                    self.simulator.resources[ResourceType.NURSE].resources[new_id] = Resource(id=new_id, type=ResourceType.NURSE)
                    self.simulator.resources[ResourceType.NURSE].available_queue.append(new_id)
                stats.dean_impact_log.append(f"  → {delta} nurse(s) added to staff")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.NURSE].available_queue))
                for _ in range(remove_count):
                    nurse_id = self.simulator.resources[ResourceType.NURSE].available_queue.pop()
                    self.simulator.resources[ResourceType.NURSE].resources.pop(nurse_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} nurse(s) FIRED/QUIT")

        if "ventilators" in effects:
            delta = effects["ventilators"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.VENTILATOR].resources)
                    self.simulator.resources[ResourceType.VENTILATOR].resources[new_id] = Resource(id=new_id, type=ResourceType.VENTILATOR)
                    self.simulator.resources[ResourceType.VENTILATOR].available_queue.append(new_id)
                stats.dean_impact_log.append(f"  → {delta} ventilator(s) added")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.VENTILATOR].available_queue))
                for _ in range(remove_count):
                    vent_id = self.simulator.resources[ResourceType.VENTILATOR].available_queue.pop()
                    self.simulator.resources[ResourceType.VENTILATOR].resources.pop(vent_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} ventilator(s) LOST/SOLD")

        if "lab" in effects:
            delta = effects["lab"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.LAB].resources)
                    self.simulator.resources[ResourceType.LAB].resources[new_id] = Resource(id=new_id, type=ResourceType.LAB)
                    self.simulator.resources[ResourceType.LAB].available_queue.append(new_id)
                stats.dean_impact_log.append(f"  → {delta} lab station(s) added")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.LAB].available_queue))
                for _ in range(remove_count):
                    lab_id = self.simulator.resources[ResourceType.LAB].available_queue.pop()
                    self.simulator.resources[ResourceType.LAB].resources.pop(lab_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} lab station(s) CLOSED")

        if "xray" in effects:
            delta = effects["xray"]
            if delta > 0:
                for _ in range(delta):
                    new_id = len(self.simulator.resources[ResourceType.XRAY].resources)
                    self.simulator.resources[ResourceType.XRAY].resources[new_id] = Resource(id=new_id, type=ResourceType.XRAY)
                    self.simulator.resources[ResourceType.XRAY].available_queue.append(new_id)
                stats.dean_impact_log.append(f"  → {delta} X-ray machine(s) added")
            elif delta < 0:
                remove_count = min(abs(delta), len(self.simulator.resources[ResourceType.XRAY].available_queue))
                for _ in range(remove_count):
                    xray_id = self.simulator.resources[ResourceType.XRAY].available_queue.pop()
                    self.simulator.resources[ResourceType.XRAY].resources.pop(xray_id, None)
                stats.dean_impact_log.append(f"  → {remove_count} X-ray machine(s) BROKEN/LOST")

        # Handle death rate changes
        if "critical_bonus" in effects:
            old_rate = self.simulator.config.critical_death_rate
            self.simulator.config.critical_death_rate = max(0.01, self.simulator.config.critical_death_rate - effects["critical_bonus"])
            stats.dean_impact_log.append(f"  → Critical death rate: {old_rate*100:.1f}% → {self.simulator.config.critical_death_rate*100:.1f}%")
        if "critical_penalty" in effects:
            old_rate = self.simulator.config.critical_death_rate
            self.simulator.config.critical_death_rate += effects["critical_penalty"]
            stats.dean_impact_log.append(f"  → Critical death rate: {old_rate*100:.1f}% → {self.simulator.config.critical_death_rate*100:.1f}%")

        if "general_bonus" in effects:
            old_rate = self.simulator.config.severe_death_rate
            self.simulator.config.severe_death_rate = max(0.01, self.simulator.config.severe_death_rate - effects["general_bonus"])
            stats.dean_impact_log.append(f"  → Severe death rate: {old_rate*100:.1f}% → {self.simulator.config.severe_death_rate*100:.1f}%")
        if "general_penalty" in effects:
            old_rate = self.simulator.config.severe_death_rate
            self.simulator.config.severe_death_rate += effects["general_penalty"]
            stats.dean_impact_log.append(f"  → Severe death rate: {old_rate*100:.1f}% → {self.simulator.config.severe_death_rate*100:.1f}%")

        # Handle treatment speed changes
        if "treatment_speed" in effects:
            old_mean = self.simulator.config.treatment_time_severe[0]
            self.simulator.config.treatment_time_severe = (max(20, self.simulator.config.treatment_time_severe[0] - int(15 * effects["treatment_speed"] / 0.05)), self.simulator.config.treatment_time_severe[1])
            stats.dean_impact_log.append(f"  → Treatment time improved: {old_mean}m → {self.simulator.config.treatment_time_severe[0]}m")
        if "treatment_delay" in effects:
            old_mean = self.simulator.config.treatment_time_severe[0]
            self.simulator.config.treatment_time_severe = (self.simulator.config.treatment_time_severe[0] + int(15 * effects["treatment_delay"] / 0.1), self.simulator.config.treatment_time_severe[1])
            stats.dean_impact_log.append(f"  → Treatment time increased: {old_mean}m → {self.simulator.config.treatment_time_severe[0]}m")

        if "triage_delay" in effects:
            old_mean = self.simulator.config.triage_time[0]
            self.simulator.config.triage_time = (self.simulator.config.triage_time[0] + int(3 * effects["triage_delay"] / 0.1), self.simulator.config.triage_time[1])
            stats.dean_impact_log.append(f"  → Triage time increased: {old_mean}m → {self.simulator.config.triage_time[0]}m")
        if "lab_delay" in effects:
            old_mean = self.simulator.config.treatment_time_moderate[0]
            self.simulator.config.treatment_time_moderate = (self.simulator.config.treatment_time_moderate[0] + int(10 * effects["lab_delay"] / 0.05), self.simulator.config.treatment_time_moderate[1])
            stats.dean_impact_log.append(f"  → Moderate treatment time increased: {old_mean}m → {self.simulator.config.treatment_time_moderate[0]}m")

        self.pending_dean_event = None
        self.dean_event_cooldown = 8.0  # Longer cooldown after major decisions

    def _show_dean_popup(self, event: Dict[str, Any]):
        """Show decision popup"""
        # Pause simulation until decision is made
        self.decision_pause_active = True
        self.pre_decision_running = self.running
        self.running = False

        popup = tk.Toplevel(self.root)
        popup.title("🏥 Dean of Medicine Decision")
        popup.geometry("750x550")
        popup.configure(bg="#0d1117")
        popup.transient(self.root)
        popup.grab_set()

        # Header
        header = ttk.Label(popup, text="🏥 DEAN OF MEDICINE DECISION", style="Header.TLabel")
        header.pack(pady=(15, 5))
        
        title_label = ttk.Label(popup, text=event["title"], style="SubHeader.TLabel")
        title_label.pack(pady=5)
        
        # Description
        desc_frame = ttk.Frame(popup)
        desc_frame.pack(fill="x", padx=20, pady=10)
        desc_label = ttk.Label(desc_frame, text=event["description"], wraplength=700, justify="left")
        desc_label.pack()

        # Separator
        sep = ttk.Separator(popup, orient="horizontal")
        sep.pack(fill="x", padx=20, pady=10)

        # Options
        options_frame = ttk.Frame(popup)
        options_frame.pack(fill="both", expand=True, padx=20, pady=10)

        for i, option in enumerate(event["options"], 1):
            opt_frame = ttk.Frame(options_frame)
            opt_frame.pack(fill="x", pady=8)
            
            # Choice number and label
            label_text = f"{i}. {option['label']}"
            btn = ttk.Button(opt_frame, text=label_text, command=lambda opt=option: self._make_dean_choice(event, opt, popup))
            btn.pack(fill="x")
            
            # Description of consequences
            if "effects" in option and "description" in option["effects"]:
                desc_text = option["effects"]["description"]
                desc = ttk.Label(opt_frame, text=f"   → {desc_text}", wraplength=650, justify="left", foreground="#8b949e")
                desc.pack(anchor="w", padx=20, pady=(2, 0))

    def _make_dean_choice(self, event: Dict[str, Any], option: Dict[str, Any], popup: Any):
        """Handle dean choice selection"""
        self._apply_dean_decision(event, option)
        popup.destroy()
        self.decision_pause_active = False
        if self.pre_decision_running:
            self.running = True
        self.pre_decision_running = False
        self.start_button.config(text="⏸ Pause" if self.running else "▶ Start")

    def toggle_start(self):
        if self.decision_pause_active:
            return
        self.running = not self.running
        self.start_button.config(text="⏸ Pause" if self.running else "▶ Start")
        if self.running:
            if self.start_time is None:
                self.start_time = time.time()
            else:
                self.start_time = time.time() - self.elapsed_real_time

    def toggle_fast(self):
        if self.fast_forward.get():
            self.speed_multiplier.set(8.0)
        else:
            self.speed_multiplier.set(1.0)

    def step_once(self):
        if not self.simulator.simulation_complete:
            self.simulator.step()
            self._refresh_ui()
        else:
            self._show_summary()

    def reset_simulation(self):
        self.simulator = ERSimulator(self.config)
        self.zone_bubbles.clear()
        self.flow_particles.clear()
        self.dean_event_cooldown = 0.0
        self.pending_dean_event = None
        self.dean_decision_log.clear()
        self.summary_shown = False
        self.decision_pause_active = False
        self.pre_decision_running = False
        self.canvas.delete("all")
        self._build_canvas_flow()
        self._refresh_ui()
        self.running = False
        self.start_button.config(text="▶ Start")
        self.start_time = None
        self.elapsed_real_time = 0.0
        if getattr(self, "summary_window", None) and self.summary_window.winfo_exists():
            self.summary_window.destroy()

    def _refresh_ui(self):
        self._update_bubbles({})
        self._update_flow_particles()
        self._trigger_flow_animations()
        self._update_resources()
        self._update_stats()
        self._maybe_trigger_dean_event()

    def _update_loop(self):
        now = time.time()
        delta = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_delta = delta

        if self.running and not self.simulator.simulation_complete:
            speed_factor = self.speed_multiplier.get()
            if self.fast_forward.get():
                speed_factor *= 3.0

            self.sim_step_accumulator += delta * self.sim_steps_per_second * speed_factor
            steps_to_run = int(self.sim_step_accumulator)
            self.sim_step_accumulator -= steps_to_run

            for _ in range(steps_to_run):
                self.simulator.step()
                if self.simulator.simulation_complete:
                    break
            self._refresh_ui()
        else:
            self._refresh_ui()
        
        if self.simulator.simulation_complete and not self.summary_shown:
            self._show_summary()
            self.running = False
            self.start_button.config(text="▶ Start")
            if self.start_time is not None:
                self.elapsed_real_time = time.time() - self.start_time
            self.summary_shown = True

        if self.running and self.start_time is not None:
            self.elapsed_real_time = time.time() - self.start_time

        delay = 30
        self.root.after(delay, self._update_loop)

    def _show_summary(self):
        if getattr(self, "summary_window", None) and self.summary_window.winfo_exists():
            return
        if self.summary_shown:
            return
        summary_lines = build_summary_lines(self.simulator, self.elapsed_real_time)
        self.summary_window = tk.Toplevel(self.root)
        self.summary_window.title("Simulation Summary")
        self.summary_window.geometry("800x600")
        
        def on_close():
            self.summary_window.destroy()
        
        self.summary_window.protocol("WM_DELETE_WINDOW", on_close)

        text = tk.Text(self.summary_window, wrap="word", bg="#0b1620", fg="#f5f7fa")
        text.pack(fill="both", expand=True)
        text.insert("1.0", "\n".join(summary_lines))
        text.config(state="disabled")

    def run(self):
        self.root.mainloop()

# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================

def run_simulation(config: SimulationConfig = None, interactive: bool = True):
    """Run the ER simulation"""
    
    if config is None:
        config = SimulationConfig()
    
    print("\n" + "=" * 80)
    print("  EMERGENCY ROOM DISCRETE EVENT SIMULATION")
    print("=" * 80)
    print(f"\n  Configuration:")
    print(f"    Duration: {config.simulation_duration // 60} hours")
    print(f"    Doctors: {config.num_doctors}, Nurses: {config.num_nurses}, Beds: {config.num_beds}")
    print(f"    Monitors: {config.num_monitors}, Ventilators: {config.num_ventilators}")
    print(f"    Arrival rate: {config.base_arrival_rate:.2f} patients/min (peak: {config.peak_arrival_rate:.2f})")
    print(f"    Critical patients: {config.critical_prob * 100:.0f}%")
    print("\n" + "=" * 80)
    
    if interactive:
        input("\n  Press Enter to start simulation...")
    
    # Create simulator
    simulator = ERSimulator(config)
    
    # Create display
    display = TerminalDisplay(simulator)
    
    # Start display thread if interactive
    display_thread = None
    if interactive:
        display_thread = threading.Thread(target=display.run_display_loop)
        display_thread.daemon = True
        display_thread.start()
    
    # Run simulation
    start_real_time = time.time()
    simulator.run()
    end_real_time = time.time()
    
    # Stop display
    display.stop()
    if display_thread:
        time.sleep(0.5)  # Give display thread time to stop
    
    # Final display
    display.display()
    
    # Print final summary
    print_final_summary(simulator, end_real_time - start_real_time)
    
    return simulator

def build_summary_lines(simulator: ERSimulator, real_time: float) -> List[str]:
    """Build final summary lines for console or GUI display"""
    stats = simulator.stats
    cfg = simulator.config
    lines = []
    
    lines.append("=" * 80)
    lines.append("  FINAL SIMULATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  SIMULATION TIME: {cfg.simulation_duration // 60} hours ({cfg.simulation_duration} minutes)")
    lines.append(f"  REAL TIME ELAPSED: {real_time:.2f} seconds")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  PATIENT STATISTICS")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append(f"    Total Arrivals:     {stats.total_arrivals}")
    lines.append(f"    Total Discharged:   {stats.total_discharged}")
    lines.append(f"    Total Deceased:     {stats.total_deceased}")
    if stats.total_arrivals:
        lines.append(f"    Survival Rate:      {(stats.total_discharged / stats.total_arrivals * 100):.1f}%")
    
    lines.append("")
    lines.append("  By Severity:")
    for severity in PatientSeverity:
        arrivals = stats.arrivals_by_severity.get(severity, 0)
        deaths = stats.deaths_by_severity.get(severity, 0)
        if arrivals > 0:
            lines.append(f"    {severity.name:<10}: {arrivals} arrivals, {deaths} deaths ({deaths/arrivals*100:.1f}%)")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  DEATH ANALYSIS")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    if stats.total_deceased > 0:
        lines.append(f"    Deaths - No Treatment:    {stats.deaths_no_treatment} ({stats.deaths_no_treatment/stats.total_deceased*100:.1f}% of deaths)")
        lines.append(f"    Deaths - Resource Shortage: {stats.deaths_resource_shortage} ({stats.deaths_resource_shortage/stats.total_deceased*100:.1f}% of deaths)")
        lines.append(f"    Deaths - Natural:         {stats.deaths_natural} ({stats.deaths_natural/stats.total_deceased*100:.1f}% of deaths)")
    else:
        lines.append("    No deaths recorded")
    lines.append("")
    lines.append(f"    Critical Patients Saved:  {stats.critical_patients_saved}")
    lines.append(f"    Critical Patients Lost:   {stats.critical_patients_lost}")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  WAIT TIME STATISTICS")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    
    if stats.wait_times_registration:
        lines.append(f"    Registration Wait:  Avg: {sum(stats.wait_times_registration)/len(stats.wait_times_registration):.1f}m, "
                     f"Max: {max(stats.wait_times_registration)}m")
    if stats.wait_times_triage:
        lines.append(f"    Triage Wait:        Avg: {sum(stats.wait_times_triage)/len(stats.wait_times_triage):.1f}m, "
                     f"Max: {max(stats.wait_times_triage)}m")
    if stats.wait_times_treatment:
        lines.append(f"    Treatment Wait:     Avg: {sum(stats.wait_times_treatment)/len(stats.wait_times_treatment):.1f}m, "
                     f"Max: {max(stats.wait_times_treatment)}m")
    
    if stats.total_time_in_er:
        lines.append(f"")
        lines.append(f"    Total ER Time:      Avg: {sum(stats.total_time_in_er)/len(stats.total_time_in_er):.1f}m, "
                     f"Max: {max(stats.total_time_in_er)}m")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  RESOURCE UTILIZATION")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    
    for res_type, utilizations in stats.resource_utilization.items():
        if utilizations:
            avg_util = sum(utilizations) / len(utilizations) * 100
            max_util = max(utilizations) * 100
            lines.append(f"    {res_type.name.replace('_', ' ').title():<15}: "
                     f"Avg: {avg_util:.1f}%, Max: {max_util:.1f}%")
    
    lines.append("")
    lines.append("  Max Wait Queues:")
    for res_type, max_len in stats.max_wait_queue_lengths.items():
        if max_len > 0:
            lines.append(f"    {res_type.name.replace('_', ' ').title():<15}: {max_len} patients")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  DOCTOR STATISTICS")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append(f"    Total Breaks Taken:     {stats.doctor_breaks_taken}")
    lines.append(f"    Total Break Time:       {stats.doctor_total_break_time} minutes")
    
    lines.append("")
    lines.append("  Patients per Doctor:")
    for doc_id in range(cfg.num_doctors):
        count = stats.patients_per_doctor.get(doc_id, 0)
        lines.append(f"    Dr.{doc_id+1}: {count} patients")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  BUILDUP PERIODS")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append(f"    Total Buildup Periods:  {len(stats.buildup_periods)}")
    
    if stats.buildup_periods:
        for i, buildup in enumerate(stats.buildup_periods[:5], 1):
            start = format_sim_time(buildup['start_time'])
            end = format_sim_time(buildup['end_time'])
            resource = buildup['resource_type'].name.replace('_', ' ').title()
            lines.append(f"    {i}. {resource}: {start} - {end} ({buildup['duration']}m)")
    
    lines.append("")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    lines.append("  DEAN OF MEDICINE DECISIONS & CONSEQUENCES")
    lines.append("  ───────────────────────────────────────────────────────────────────────")
    if stats.dean_decisions:
        for decision in stats.dean_decisions:
            lines.append(f"    [{decision['time']}] {decision['event']}")
            lines.append(f"      CHOICE: {decision['choice']}")
            # Show impact log if available
            if "impact_log" in decision:
                lines.append(f"      IMPACT: {decision['impact_log']}")
            if "description" in decision and decision["description"]:
                lines.append(f"      EFFECT: {decision['description']}")
            lines.append("")
    else:
        lines.append("    No dean decisions recorded")

    if stats.dean_impact_log:
        lines.append("\n  CHAIN OF EVENTS LOG:")
        for impact in stats.dean_impact_log:
            lines.append(f"    {impact}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF SIMULATION")
    lines.append("=" * 80)
    return lines


def print_final_summary(simulator: ERSimulator, real_time: float):
    """Print final simulation summary"""
    for line in build_summary_lines(simulator, real_time):
        print(line)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency Room Discrete Event Simulation")
    parser.add_argument("--duration", type=int, default=1440, help="Simulation duration in minutes")
    parser.add_argument("--doctors", type=int, default=4, help="Number of doctors")
    parser.add_argument("--nurses", type=int, default=8, help="Number of nurses")
    parser.add_argument("--beds", type=int, default=20, help="Number of beds")
    parser.add_argument("--no-interactive", action="store_true", help="Run without interactive display")
    parser.add_argument("--gui", action="store_true", help="Run with GUI visualization")
    parser.add_argument("--fast", action="store_true", help="Run at faster time scale")
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        simulation_duration=args.duration,
        num_doctors=args.doctors,
        num_nurses=args.nurses,
        num_beds=args.beds,
        time_scale=0.001 if args.fast else 0.01
    )
    
    if args.gui:
        if tk is None:
            print("Tkinter is not available. Please install tkinter to use the GUI.")
            return
        app = ERGuiApp(config)
        app.run()
    else:
        run_simulation(config, interactive=not args.no_interactive)

if __name__ == "__main__":
    main()
