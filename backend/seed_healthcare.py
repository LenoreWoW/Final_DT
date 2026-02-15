"""
Healthcare Quantum Digital Twin Seeder
=======================================

Creates a pre-built, fully-configured healthcare Quantum Digital Twin
so the platform has meaningful demo data on first launch.

Usage:
    # Programmatic (e.g. in main.py startup)
    from backend.seed_healthcare import seed_if_empty
    from backend.models.database import engine
    seed_if_empty(engine)

    # Standalone
    python -m backend.seed_healthcare
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from backend.models.database import (
    TwinModel,
    ConversationModel,
    SimulationModel,
)
from backend.engine.quantum_modules import _generate_qasm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed, predictable IDs
# ---------------------------------------------------------------------------
TWIN_ID = "healthcare-demo-twin-0001"
CONVERSATION_ID = "healthcare-demo-conv-0001"
SIMULATION_ID = "healthcare-demo-sim-0001"

# Timestamps for realistic history
_NOW = datetime.utcnow()
_CREATED = _NOW - timedelta(hours=2)
_UPDATED = _NOW - timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Extracted system: a rich healthcare model
# ---------------------------------------------------------------------------

def _build_extracted_system() -> dict:
    """Build the extracted_system JSON for the healthcare twin."""
    return {
        "entities": [
            {
                "id": "hospital-1",
                "name": "Central General Hospital",
                "type": "hospital",
                "properties": {
                    "total_beds": 500,
                    "icu_beds": 60,
                    "er_bays": 40,
                    "operating_rooms": 12,
                    "location": [24.7136, 46.6753],
                    "accreditation": "JCI",
                },
            },
            {
                "id": "patients",
                "name": "Patient Population",
                "type": "patient_group",
                "properties": {
                    "average_daily_admissions": 120,
                    "average_daily_discharges": 115,
                    "average_los_days": 4.2,
                    "acuity_distribution": {
                        "low": 0.30,
                        "moderate": 0.40,
                        "high": 0.20,
                        "critical": 0.10,
                    },
                },
            },
            {
                "id": "doctors",
                "name": "Physician Staff",
                "type": "staff_group",
                "properties": {
                    "total_physicians": 85,
                    "specialties": [
                        "emergency_medicine",
                        "internal_medicine",
                        "surgery",
                        "cardiology",
                        "pulmonology",
                        "neurology",
                        "pediatrics",
                        "oncology",
                    ],
                    "attending_per_shift": 12,
                },
            },
            {
                "id": "nurses",
                "name": "Nursing Staff",
                "type": "staff_group",
                "properties": {
                    "total_nurses": 320,
                    "per_shift": 80,
                    "specializations": [
                        "icu",
                        "er",
                        "med_surg",
                        "oncology",
                        "pediatrics",
                    ],
                },
            },
            {
                "id": "icu",
                "name": "Intensive Care Unit",
                "type": "department",
                "properties": {
                    "beds": 60,
                    "ventilators": 30,
                    "monitors": 60,
                    "isolation_rooms": 10,
                    "average_occupancy": 0.82,
                    "nurse_patient_ratio": "1:2",
                },
            },
            {
                "id": "pharmacy",
                "name": "Hospital Pharmacy",
                "type": "department",
                "properties": {
                    "formulary_items": 2400,
                    "daily_dispenses": 850,
                    "critical_stock_items": 45,
                    "automated_dispensing_cabinets": 18,
                    "average_turnaround_minutes": 22,
                },
            },
        ],
        "relationships": [
            {
                "source_id": "patients",
                "target_id": "hospital-1",
                "type": "admitted_to",
                "strength": 1.0,
                "properties": {"flow_rate": "120/day"},
            },
            {
                "source_id": "doctors",
                "target_id": "patients",
                "type": "treats",
                "strength": 0.95,
                "properties": {"avg_patients_per_doctor": 8},
            },
            {
                "source_id": "nurses",
                "target_id": "patients",
                "type": "cares_for",
                "strength": 0.98,
                "properties": {"general_ratio": "1:5", "icu_ratio": "1:2"},
            },
            {
                "source_id": "patients",
                "target_id": "icu",
                "type": "escalated_to",
                "strength": 0.10,
                "properties": {"trigger": "clinical_deterioration"},
            },
            {
                "source_id": "pharmacy",
                "target_id": "patients",
                "type": "dispenses_to",
                "strength": 0.90,
                "properties": {"avg_meds_per_patient": 6.3},
            },
            {
                "source_id": "doctors",
                "target_id": "pharmacy",
                "type": "prescribes_via",
                "strength": 0.85,
                "properties": {"electronic_prescribing": True},
            },
            {
                "source_id": "icu",
                "target_id": "hospital-1",
                "type": "department_of",
                "strength": 1.0,
                "properties": {},
            },
            {
                "source_id": "pharmacy",
                "target_id": "hospital-1",
                "type": "department_of",
                "strength": 1.0,
                "properties": {},
            },
        ],
        "rules": [
            {
                "id": "triage-rule",
                "name": "Emergency Triage Protocol",
                "description": (
                    "Incoming patients are triaged into four acuity levels "
                    "(critical, high, moderate, low). Critical patients are "
                    "seen within 5 minutes; high within 15 minutes; moderate "
                    "within 30 minutes; low within 60 minutes."
                ),
                "formula": "wait_target = {critical: 5, high: 15, moderate: 30, low: 60}",
                "type": "clinical",
            },
            {
                "id": "staffing-ratio-rule",
                "name": "Nurse Staffing Ratios",
                "description": (
                    "ICU maintains a 1:2 nurse-to-patient ratio. "
                    "Med/Surg floors maintain 1:5. Emergency department "
                    "maintains 1:4. If ratios exceed thresholds, overtime "
                    "staff are called in."
                ),
                "formula": "required_nurses = ceil(patients / ratio)",
                "type": "staffing",
            },
            {
                "id": "bed-turnover-rule",
                "name": "Bed Turnover Management",
                "description": (
                    "Discharge planning begins at admission. Target bed "
                    "turnover time after discharge is 90 minutes (cleaning, "
                    "preparation). Delayed discharges trigger escalation "
                    "after 2 hours past planned discharge."
                ),
                "formula": "turnover_time = cleaning_time + prep_time <= 90 min",
                "type": "operations",
            },
            {
                "id": "medication-safety-rule",
                "name": "Medication Safety Protocol",
                "description": (
                    "All medications require double-verification for high-risk "
                    "drugs. Pharmacy turnaround for STAT orders must be under "
                    "15 minutes. Drug interaction checks are mandatory."
                ),
                "formula": "stat_turnaround <= 15 min AND interaction_check = True",
                "type": "clinical",
            },
        ],
        "constraints": [
            {
                "id": "bed-capacity",
                "name": "Maximum Bed Capacity",
                "type": "boundary",
                "value": 500,
                "operator": "<=",
            },
            {
                "id": "icu-capacity",
                "name": "ICU Bed Capacity",
                "type": "boundary",
                "value": 60,
                "operator": "<=",
            },
            {
                "id": "er-wait-limit",
                "name": "ER Maximum Wait Time",
                "type": "time",
                "value": 240,
                "operator": "<=",
            },
            {
                "id": "nurse-ratio-floor",
                "name": "Minimum Nurse-to-Patient Ratio (General)",
                "type": "requirement",
                "value": 0.20,
                "operator": ">=",
            },
            {
                "id": "ventilator-capacity",
                "name": "Ventilator Availability",
                "type": "boundary",
                "value": 30,
                "operator": "<=",
            },
            {
                "id": "pharmacy-turnaround",
                "name": "Pharmacy STAT Turnaround",
                "type": "time",
                "value": 15,
                "operator": "<=",
            },
        ],
        "goal": "optimize patient flow and reduce wait times",
        "domain": "healthcare",
    }


# ---------------------------------------------------------------------------
# QASM circuits for healthcare algorithms
# ---------------------------------------------------------------------------

def _build_qasm_circuits() -> dict:
    """Generate QASM circuits for the healthcare twin's algorithms."""
    return {
        "hospital_operations_qaoa": _generate_qasm("hospital_operations", 8, 3),
        "patient_flow_optimization": _generate_qasm("qaoa", 6, 3),
        "resource_allocation": _generate_qasm("qaoa", 10, 4),
        "epidemic_forecasting": _generate_qasm("epidemic_modeling", 8, 3),
        "treatment_optimization": _generate_qasm("personalized_medicine", 8, 4),
        "drug_interaction_analysis": _generate_qasm("drug_discovery", 8, 4),
        "medical_imaging_qnn": _generate_qasm("medical_imaging", 8, 3),
        "genomic_profiling": _generate_qasm("genomic_analysis", 8, 4),
    }


# ---------------------------------------------------------------------------
# Quantum metrics
# ---------------------------------------------------------------------------

def _build_quantum_metrics() -> dict:
    """Build quantum_metrics showing quantum advantage."""
    return {
        "total_qubits_used": 68,
        "total_circuits_executed": 8,
        "algorithms_applied": [
            "qaoa",
            "hospital_operations",
            "epidemic_modeling",
            "personalized_medicine",
            "drug_discovery",
            "medical_imaging",
            "genomic_analysis",
        ],
        "quantum_advantage": {
            "patient_flow_optimization": {
                "classical_time_seconds": 12.4,
                "quantum_time_seconds": 1.8,
                "speedup": 6.89,
                "accuracy_improvement_percent": 23.5,
                "description": (
                    "QAOA-based patient flow optimization achieved 6.89x speedup "
                    "over classical integer programming with 23.5% better resource "
                    "utilization."
                ),
            },
            "resource_allocation": {
                "classical_time_seconds": 8.7,
                "quantum_time_seconds": 1.2,
                "speedup": 7.25,
                "accuracy_improvement_percent": 18.2,
                "description": (
                    "Quantum resource allocation reduced nurse scheduling computation "
                    "time by 7.25x while improving coverage fairness by 18.2%."
                ),
            },
            "epidemic_forecasting": {
                "classical_time_seconds": 45.3,
                "quantum_time_seconds": 5.1,
                "speedup": 8.88,
                "accuracy_improvement_percent": 31.4,
                "description": (
                    "Quantum Monte Carlo epidemic simulation explored 8.88x more "
                    "scenarios than classical methods, improving prediction accuracy "
                    "by 31.4%."
                ),
            },
            "treatment_optimization": {
                "classical_time_seconds": 15.6,
                "quantum_time_seconds": 2.3,
                "speedup": 6.78,
                "accuracy_improvement_percent": 27.8,
                "description": (
                    "Personalized treatment optimization via VQE-based molecular "
                    "simulation achieved 27.8% improvement in predicted response "
                    "rates."
                ),
            },
        },
        "overall_speedup": 7.45,
        "overall_accuracy_gain_percent": 25.2,
        "circuit_statistics": {
            "max_circuit_depth": 40,
            "avg_circuit_depth": 22,
            "total_gate_count": 1530,
            "two_qubit_gate_count": 412,
        },
    }


# ---------------------------------------------------------------------------
# Conversation history showing creation flow
# ---------------------------------------------------------------------------

def _build_conversation_messages() -> list:
    """Build conversation messages showing the twin creation flow."""
    base = _CREATED
    return [
        {
            "role": "user",
            "content": (
                "I want to create a digital twin of a hospital to optimize "
                "patient flow and reduce emergency room wait times. The hospital "
                "has 500 beds, 60 ICU beds, and serves about 120 admissions per day."
            ),
            "timestamp": (base + timedelta(seconds=0)).isoformat(),
            "metadata": {},
        },
        {
            "role": "assistant",
            "content": (
                "I have detected a healthcare domain system. I have identified the "
                "following entities so far:\n\n"
                "- **Central General Hospital** (500 beds, 60 ICU, 40 ER bays)\n"
                "- **Patient Population** (~120 daily admissions)\n\n"
                "To build a comprehensive digital twin, I need a few more details:\n"
                "1. How many physicians and nurses are on staff?\n"
                "2. What are your current triage protocols and staffing ratios?\n"
                "3. Do you have a hospital pharmacy, and what is its average "
                "turnaround time?\n"
                "4. Are there specific bottlenecks you have already identified?"
            ),
            "timestamp": (base + timedelta(seconds=3)).isoformat(),
            "metadata": {
                "entities_found": 2,
                "domain_detected": "healthcare",
            },
        },
        {
            "role": "user",
            "content": (
                "We have 85 physicians across 8 specialties with 12 attendings per "
                "shift. Nursing staff is 320 total, 80 per shift. ICU ratio is 1:2, "
                "general floors are 1:5, ER is 1:4. The pharmacy has 2400 formulary "
                "items and dispenses about 850 medications daily with a 22-minute "
                "average turnaround. STAT orders need to be under 15 minutes. "
                "The main bottleneck is ER-to-floor bed placement -- patients wait "
                "too long for bed assignments after being admitted from the ER."
            ),
            "timestamp": (base + timedelta(seconds=45)).isoformat(),
            "metadata": {},
        },
        {
            "role": "assistant",
            "content": (
                "Excellent -- I now have a complete picture. Here is the system I "
                "have extracted:\n\n"
                "**Entities:** Hospital, Patients, Physicians (85), Nurses (320), "
                "ICU (60 beds), Pharmacy\n\n"
                "**Key Rules:**\n"
                "- Triage protocol with 4 acuity levels and time targets\n"
                "- Staffing ratios (ICU 1:2, General 1:5, ER 1:4)\n"
                "- Bed turnover target of 90 minutes\n"
                "- Medication safety with STAT turnaround under 15 min\n\n"
                "**Constraints:** 500 bed cap, 60 ICU cap, 240 min ER wait limit, "
                "30 ventilators\n\n"
                "**Goal:** Optimize patient flow and reduce wait times\n\n"
                "I am now generating your Quantum Digital Twin using QAOA for "
                "patient flow optimization, quantum Monte Carlo for capacity "
                "forecasting, and VQE-based resource scheduling. This will take "
                "a moment..."
            ),
            "timestamp": (base + timedelta(seconds=50)).isoformat(),
            "metadata": {
                "entities_found": 6,
                "relationships_found": 8,
                "rules_found": 4,
                "constraints_found": 6,
                "goal_detected": True,
                "twin_status": "generating",
            },
        },
        {
            "role": "assistant",
            "content": (
                "Your Healthcare Quantum Digital Twin is now **ACTIVE** and ready "
                "for simulation.\n\n"
                "**Quantum circuits generated:**\n"
                "- Hospital Operations QAOA (8 qubits)\n"
                "- Patient Flow Optimization (6 qubits)\n"
                "- Resource Allocation (10 qubits)\n"
                "- Epidemic Forecasting (8 qubits)\n"
                "- Treatment Optimization (8 qubits)\n"
                "- Drug Interaction Analysis (8 qubits)\n"
                "- Medical Imaging QNN (8 qubits)\n"
                "- Genomic Profiling (8 qubits)\n\n"
                "**Quantum advantage detected:** 7.45x overall speedup with "
                "25.2% accuracy improvement over classical methods.\n\n"
                "You can now:\n"
                "- Run simulations to forecast patient flow\n"
                "- Ask 'what-if' questions about staffing changes\n"
                "- Optimize bed allocation in real time\n"
                "- Predict ER surge events"
            ),
            "timestamp": (base + timedelta(seconds=68)).isoformat(),
            "metadata": {
                "twin_status": "active",
                "circuits_generated": 8,
                "quantum_advantage": 7.45,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

def _build_simulation() -> dict:
    """Build pre-populated simulation parameters and results."""
    return {
        "parameters": {
            "scenario": "baseline_patient_flow",
            "time_horizon_hours": 72,
            "admission_rate": 120,
            "discharge_rate": 115,
            "icu_escalation_rate": 0.10,
            "er_surge_probability": 0.15,
            "staffing_model": "current",
        },
        "results": {
            "average_er_wait_minutes": 42.3,
            "average_bed_placement_minutes": 68.7,
            "icu_occupancy_peak": 0.91,
            "bed_utilization_rate": 0.87,
            "pharmacy_stat_compliance": 0.94,
            "patient_satisfaction_index": 7.2,
            "staff_overtime_hours": 124,
            "bottleneck_analysis": {
                "primary_bottleneck": "ER-to-floor bed placement",
                "secondary_bottleneck": "discharge processing delay",
                "suggested_interventions": [
                    {
                        "intervention": "Implement predictive discharge planning",
                        "expected_improvement_percent": 22,
                        "implementation_effort": "medium",
                    },
                    {
                        "intervention": "Add dedicated bed-assignment coordinator",
                        "expected_improvement_percent": 18,
                        "implementation_effort": "low",
                    },
                    {
                        "intervention": "Deploy real-time bed tracking dashboard",
                        "expected_improvement_percent": 15,
                        "implementation_effort": "medium",
                    },
                ],
            },
            "hourly_forecast": {
                "peak_admission_hours": [9, 10, 11, 14, 15, 19, 20],
                "peak_discharge_hours": [10, 11, 14, 15],
                "critical_capacity_hours": [11, 12, 20, 21],
            },
            "quantum_optimization_results": {
                "optimal_bed_assignment_order": (
                    "Quantum QAOA found a patient-to-bed assignment that "
                    "reduces average placement time by 31% compared to "
                    "first-come-first-served."
                ),
                "staffing_recommendation": (
                    "Shift 12% of nursing hours from 06:00-14:00 to "
                    "14:00-22:00 to match predicted demand curves."
                ),
                "er_surge_protocol": (
                    "Quantum Monte Carlo predicts 3 surge events in 72h "
                    "window. Pre-positioning 8 additional beds in overflow "
                    "area at hours 19-21 prevents capacity breach."
                ),
            },
        },
        "predictions": [
            {
                "metric": "ER wait time",
                "current_value": 42.3,
                "predicted_value_with_optimization": 29.1,
                "improvement_percent": 31.2,
                "confidence": 0.89,
                "time_horizon": "72 hours",
            },
            {
                "metric": "Bed placement time",
                "current_value": 68.7,
                "predicted_value_with_optimization": 41.2,
                "improvement_percent": 40.0,
                "confidence": 0.85,
                "time_horizon": "72 hours",
            },
            {
                "metric": "ICU occupancy peak",
                "current_value": 0.91,
                "predicted_value_with_optimization": 0.84,
                "improvement_percent": 7.7,
                "confidence": 0.82,
                "time_horizon": "72 hours",
            },
            {
                "metric": "Staff overtime hours",
                "current_value": 124,
                "predicted_value_with_optimization": 86,
                "improvement_percent": 30.6,
                "confidence": 0.78,
                "time_horizon": "72 hours",
            },
        ],
        "quantum_advantage": {
            "simulation_speedup": 7.45,
            "scenarios_explored": 10000,
            "classical_equivalent_scenarios": 1342,
            "accuracy_improvement_percent": 25.2,
            "qubits_used": 68,
            "circuit_executions": 8,
        },
    }


# ---------------------------------------------------------------------------
# Core seeding functions
# ---------------------------------------------------------------------------

def seed_healthcare_twin(db_session: Session) -> TwinModel:
    """
    Insert a fully-configured healthcare Quantum Digital Twin into the database.

    Returns the created TwinModel instance.
    """
    # -- Build all data -------------------------------------------------------
    extracted_system = _build_extracted_system()
    qasm_circuits = _build_qasm_circuits()
    quantum_metrics = _build_quantum_metrics()
    conversation_messages = _build_conversation_messages()
    simulation_data = _build_simulation()

    # -- Twin -----------------------------------------------------------------
    twin = TwinModel(
        id=TWIN_ID,
        name="Hospital Patient Flow Optimizer",
        description=(
            "A Quantum Digital Twin of a 500-bed general hospital, modelling "
            "patient flow from ER admission through discharge. Uses QAOA for "
            "bed assignment optimization, quantum Monte Carlo for capacity "
            "forecasting, and VQE-based resource scheduling to reduce ER wait "
            "times and improve bed utilization."
        ),
        status="active",
        domain="healthcare",
        extracted_system=extracted_system,
        state={
            "entities": {
                "hospital-1": {"occupancy": 0.87, "er_wait_min": 42.3},
                "icu": {"occupancy": 0.82, "ventilators_in_use": 22},
                "patients": {"admitted_today": 120, "discharged_today": 115},
                "pharmacy": {"stat_compliance": 0.94, "pending_orders": 12},
            },
            "time_step": 0,
            "metrics": {
                "er_wait_minutes": 42.3,
                "bed_placement_minutes": 68.7,
                "icu_occupancy": 0.82,
                "patient_satisfaction": 7.2,
            },
        },
        algorithms_used=[
            "qaoa",
            "vqe",
            "qnn",
            "tensor_network",
            "quantum_simulation",
            "vqc",
        ],
        quantum_metrics=quantum_metrics,
        qasm_circuits=qasm_circuits,
        is_public=True,
        created_at=_CREATED,
        updated_at=_UPDATED,
    )
    db_session.add(twin)

    # -- Conversation ---------------------------------------------------------
    conversation = ConversationModel(
        id=CONVERSATION_ID,
        twin_id=TWIN_ID,
        messages=conversation_messages,
        context={
            "domain": "healthcare",
            "entities_count": 6,
            "relationships_count": 8,
            "rules_count": 4,
            "constraints_count": 6,
            "goal": "optimize patient flow and reduce wait times",
            "generation_complete": True,
        },
        created_at=_CREATED,
        updated_at=_UPDATED,
    )
    db_session.add(conversation)

    # -- Simulation -----------------------------------------------------------
    simulation = SimulationModel(
        id=SIMULATION_ID,
        twin_id=TWIN_ID,
        time_steps=72,
        scenarios_run=10000,
        parameters=simulation_data["parameters"],
        results=simulation_data["results"],
        predictions=simulation_data["predictions"],
        quantum_advantage=simulation_data["quantum_advantage"],
        execution_time_seconds=5.1,
        created_at=_CREATED,
    )
    db_session.add(simulation)

    # -- Commit ---------------------------------------------------------------
    db_session.commit()
    db_session.refresh(twin)

    logger.info(
        "Seeded healthcare demo twin: id=%s, name='%s'",
        twin.id,
        twin.name,
    )
    return twin


def seed_if_empty(engine) -> bool:
    """
    Check if any twins exist in the database. If not, seed the healthcare twin.

    Safe to call on every application startup -- it is a no-op when data
    already exists.

    Args:
        engine: SQLAlchemy engine instance.

    Returns:
        True if seeding was performed, False if data already existed.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        twin_count = session.query(TwinModel).count()
        if twin_count > 0:
            logger.debug(
                "Database already contains %d twin(s); skipping seed.",
                twin_count,
            )
            return False

        logger.info("No twins found in database. Seeding healthcare demo twin...")
        seed_healthcare_twin(session)
        return True

    except Exception:
        session.rollback()
        logger.exception("Failed to seed healthcare demo twin.")
        return False
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from backend.models.database import engine, init_database

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    init_database(engine)
    seeded = seed_if_empty(engine)
    if seeded:
        print(f"Healthcare demo twin seeded successfully (id={TWIN_ID}).")
    else:
        print("Database already contains twins; no seeding performed.")
