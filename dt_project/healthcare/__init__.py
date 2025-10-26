"""
🏥 Healthcare Quantum Digital Twin Platform
==========================================

Specialized healthcare implementations leveraging quantum computing for:
- Personalized medicine and treatment planning
- Drug discovery and molecular simulation
- Medical imaging and diagnostics
- Genomic analysis and precision oncology
- Epidemic modeling and public health
- Hospital operations optimization

All modules are research-grounded and HIPAA-compliant.

Author: Hassan Al-Sahli
Purpose: Healthcare-focused quantum digital twin platform
Reference: docs/HEALTHCARE_FOCUS_STRATEGIC_PLAN.md
"""

from .personalized_medicine import PersonalizedMedicineQuantumTwin
from .drug_discovery import DrugDiscoveryQuantumTwin
from .medical_imaging import MedicalImagingQuantumTwin
from .genomic_analysis import GenomicAnalysisQuantumTwin
from .epidemic_modeling import EpidemicModelingQuantumTwin
from .hospital_operations import HospitalOperationsQuantumTwin
from .clinical_validation import ClinicalValidationFramework
from .hipaa_compliance import HIPAAComplianceFramework
from .healthcare_conversational_ai import HealthcareConversationalAI

__all__ = [
    'PersonalizedMedicineQuantumTwin',
    'DrugDiscoveryQuantumTwin',
    'MedicalImagingQuantumTwin',
    'GenomicAnalysisQuantumTwin',
    'EpidemicModelingQuantumTwin',
    'HospitalOperationsQuantumTwin',
    'ClinicalValidationFramework',
    'HIPAAComplianceFramework',
    'HealthcareConversationalAI',
]

__version__ = '1.0.0'
__author__ = 'Hassan Al-Sahli'
