#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE HEALTHCARE MODULE TESTS
=========================================

Comprehensive test suite for all 6 healthcare quantum digital twin modules:
1. Personalized Medicine
2. Drug Discovery
3. Medical Imaging
4. Genomic Analysis
5. Epidemic Modeling
6. Hospital Operations

Test Coverage:
    - Unit tests for each module
    - Integration tests across modules
    - Clinical validation against benchmarks
    - Performance benchmarks (quantum advantages)
    - HIPAA compliance validation
    - Edge cases and error handling

Author: Hassan Al-Sahli
Purpose: Clinical validation testing for healthcare quantum platform
Reference: HEALTHCARE_FOCUS_STRATEGIC_PLAN.md - Week 2 Testing
"""

import pytest
import asyncio
import numpy as np
import logging
from typing import List
from datetime import datetime

# Import healthcare modules
try:
    from dt_project.healthcare.personalized_medicine import (
        PersonalizedMedicineQuantumTwin,
        CancerType
    )
    from dt_project.healthcare.drug_discovery import DrugDiscoveryQuantumTwin
    from dt_project.healthcare.medical_imaging import MedicalImagingQuantumTwin
    from dt_project.healthcare.genomic_analysis import GenomicAnalysisQuantumTwin
    from dt_project.healthcare.epidemic_modeling import (
        EpidemicModelingQuantumTwin,
        InterventionType
    )
    from dt_project.healthcare.hospital_operations import HospitalOperationsQuantumTwin
    from dt_project.healthcare.clinical_validation import (
        ClinicalValidationFramework,
        ClinicalBenchmark,
        RegulatoryFramework
    )
    from dt_project.healthcare.hipaa_compliance import (
        HIPAAComplianceFramework,
        PHICategory,
        AccessLevel
    )
    from dt_project.healthcare.healthcare_conversational_ai import (
        HealthcareConversationalAI,
        UserRole
    )
    HEALTHCARE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Healthcare modules not available: {e}")
    HEALTHCARE_MODULES_AVAILABLE = False
    pytest.skip("Healthcare modules not available", allow_module_level=True)

# Import synthetic data generator
try:
    from tests.synthetic_patient_data_generator import SyntheticPatientDataGenerator
    DATA_GENERATOR_AVAILABLE = True
except ImportError:
    logging.warning("Synthetic data generator not available")
    DATA_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def data_generator():
    """Synthetic data generator fixture"""
    if not DATA_GENERATOR_AVAILABLE:
        pytest.skip("Data generator not available")
    return SyntheticPatientDataGenerator(random_seed=42)


@pytest.fixture
def personalized_medicine_twin():
    """Personalized medicine quantum twin fixture"""
    return PersonalizedMedicineQuantumTwin()


@pytest.fixture
def drug_discovery_twin():
    """Drug discovery quantum twin fixture"""
    return DrugDiscoveryQuantumTwin()


@pytest.fixture
def medical_imaging_twin():
    """Medical imaging quantum twin fixture"""
    return MedicalImagingQuantumTwin()


@pytest.fixture
def genomic_analysis_twin():
    """Genomic analysis quantum twin fixture"""
    return GenomicAnalysisQuantumTwin()


@pytest.fixture
def epidemic_modeling_twin():
    """Epidemic modeling quantum twin fixture"""
    return EpidemicModelingQuantumTwin()


@pytest.fixture
def hospital_operations_twin():
    """Hospital operations quantum twin fixture"""
    return HospitalOperationsQuantumTwin()


@pytest.fixture
def clinical_validator():
    """Clinical validation framework fixture"""
    return ClinicalValidationFramework()


@pytest.fixture
def hipaa_framework():
    """HIPAA compliance framework fixture"""
    return HIPAAComplianceFramework()


@pytest.fixture
def healthcare_ai():
    """Healthcare conversational AI fixture"""
    return HealthcareConversationalAI()


# ============================================================================
# TEST CLASS 1: PERSONALIZED MEDICINE
# ============================================================================

class TestPersonalizedMedicine:
    """Tests for personalized medicine quantum twin"""

    @pytest.mark.asyncio
    async def test_create_treatment_plan(self, personalized_medicine_twin, data_generator):
        """Test personalized treatment plan creation"""
        # Generate test patient
        patient = data_generator.generate_cancer_patient(cancer_type=CancerType.NSCLC)

        # Create treatment plan
        plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)

        # Assertions
        assert plan is not None
        assert plan.patient_id == patient.patient_id
        assert plan.primary_treatment is not None
        assert plan.confidence_score > 0.0
        assert len(plan.quantum_modules_used) > 0

        logger.info(f"✅ Treatment plan created for {patient.patient_id}")
        logger.info(f"   Primary treatment: {plan.primary_treatment.therapy_name}")
        logger.info(f"   Confidence: {plan.confidence_score:.1%}")

    @pytest.mark.asyncio
    async def test_multiple_cancer_types(self, personalized_medicine_twin, data_generator):
        """Test treatment planning for multiple cancer types"""
        cancer_types = [CancerType.NSCLC, CancerType.BREAST, CancerType.MELANOMA]

        for cancer_type in cancer_types:
            patient = data_generator.generate_cancer_patient(cancer_type=cancer_type)
            plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)

            assert plan is not None
            assert plan.confidence_score > 0.0

            logger.info(f"✅ {cancer_type.value} treatment plan - confidence: {plan.confidence_score:.1%}")

    @pytest.mark.asyncio
    async def test_quantum_advantage_tracking(self, personalized_medicine_twin, data_generator):
        """Test quantum advantage tracking in treatment planning"""
        patient = data_generator.generate_cancer_patient()
        plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)

        # Check quantum advantage metrics
        assert 'quantum_advantage_summary' in plan.__dict__
        assert len(plan.quantum_modules_used) >= 3  # Should use multiple quantum modules

        logger.info(f"✅ Quantum modules used: {', '.join(plan.quantum_modules_used)}")


# ============================================================================
# TEST CLASS 2: DRUG DISCOVERY
# ============================================================================

class TestDrugDiscovery:
    """Tests for drug discovery quantum twin"""

    @pytest.mark.asyncio
    async def test_discover_drug_candidates(self, drug_discovery_twin, data_generator):
        """Test drug candidate discovery"""
        # Generate target protein
        target = data_generator.generate_target_protein(protein_id="EGFR")

        # Discover candidates
        result = await drug_discovery_twin.discover_drug_candidates(
            target,
            num_candidates=100
        )

        # Assertions
        assert result is not None
        assert len(result.top_candidates) > 0
        assert result.quantum_speedup > 1.0
        assert result.confidence_score > 0.0

        logger.info(f"✅ Discovered {len(result.top_candidates)} drug candidates")
        logger.info(f"   Quantum speedup: {result.quantum_speedup:.0f}x")
        logger.info(f"   Top candidate: {result.top_candidates[0].smiles[:50]}...")

    @pytest.mark.asyncio
    async def test_drug_properties(self, drug_discovery_twin, data_generator):
        """Test drug candidate properties"""
        target = data_generator.generate_target_protein()
        result = await drug_discovery_twin.discover_drug_candidates(target, num_candidates=50)

        # Check top candidate properties
        top_candidate = result.top_candidates[0]

        assert top_candidate.binding_affinity_kcal < 0  # Should be negative
        assert 0.0 <= top_candidate.druglikeness_score <= 1.0
        assert 'oral_bioavailability' in top_candidate.admet_profile

        logger.info(f"✅ Top candidate properties validated")
        logger.info(f"   Binding affinity: {top_candidate.binding_affinity_kcal:.1f} kcal/mol")
        logger.info(f"   Druglikeness: {top_candidate.druglikeness_score:.2f}")


# ============================================================================
# TEST CLASS 3: MEDICAL IMAGING
# ============================================================================

class TestMedicalImaging:
    """Tests for medical imaging quantum twin"""

    @pytest.mark.asyncio
    async def test_analyze_medical_image(self, medical_imaging_twin, data_generator):
        """Test medical image analysis"""
        # Generate image metadata
        from dt_project.healthcare.medical_imaging import ImageModality, AnatomicalRegion

        image = data_generator.generate_medical_image_metadata(
            modality=ImageModality.CT,
            region=AnatomicalRegion.CHEST
        )

        # Analyze image
        report = await medical_imaging_twin.analyze_medical_image(image)

        # Assertions
        assert report is not None
        assert report.image_id == image.image_id
        assert report.confidence_score > 0.0
        assert len(report.quantum_modules_used) > 0

        logger.info(f"✅ Medical image analyzed: {image.modality.value}")
        logger.info(f"   Classification: {report.classification}")
        logger.info(f"   Confidence: {report.confidence_score:.1%}")

    @pytest.mark.asyncio
    async def test_multiple_modalities(self, medical_imaging_twin, data_generator):
        """Test analysis of multiple imaging modalities"""
        from dt_project.healthcare.medical_imaging import ImageModality

        modalities = [ImageModality.CT, ImageModality.MRI, ImageModality.X_RAY]

        for modality in modalities:
            image = data_generator.generate_medical_image_metadata(modality=modality)
            report = await medical_imaging_twin.analyze_medical_image(image)

            assert report is not None
            assert report.confidence_score > 0.0

            logger.info(f"✅ {modality.value} analysis - confidence: {report.confidence_score:.1%}")


# ============================================================================
# TEST CLASS 4: GENOMIC ANALYSIS
# ============================================================================

class TestGenomicAnalysis:
    """Tests for genomic analysis quantum twin"""

    @pytest.mark.asyncio
    async def test_analyze_genomic_profile(self, genomic_analysis_twin, data_generator):
        """Test genomic profile analysis"""
        # Generate patient with variants
        patient = data_generator.generate_cancer_patient(
            cancer_type=CancerType.NSCLC,
            include_mutations=True
        )

        # Generate genetic variants
        variants = data_generator.generate_genetic_variants(
            num_variants=5,
            cancer_type=CancerType.NSCLC
        )

        # Analyze genomic profile
        result = await genomic_analysis_twin.analyze_genomic_profile(
            patient_id=patient.patient_id,
            variants=variants
        )

        # Assertions
        assert result is not None
        assert len(result.actionable_mutations) >= 0
        assert result.confidence_score > 0.0

        logger.info(f"✅ Genomic analysis complete")
        logger.info(f"   Actionable mutations: {len(result.actionable_mutations)}")
        logger.info(f"   Confidence: {result.confidence_score:.1%}")

    @pytest.mark.asyncio
    async def test_pathway_analysis(self, genomic_analysis_twin, data_generator):
        """Test pathway analysis capability"""
        variants = data_generator.generate_genetic_variants(
            num_variants=10,
            cancer_type=CancerType.BREAST
        )

        result = await genomic_analysis_twin.analyze_genomic_profile(
            patient_id="TEST_001",
            variants=variants
        )

        # Should identify pathway dysregulation
        assert 'pathways_dysregulated' in result.__dict__
        assert len(result.quantum_modules_used) > 0

        logger.info(f"✅ Pathway analysis validated")


# ============================================================================
# TEST CLASS 5: EPIDEMIC MODELING
# ============================================================================

class TestEpidemicModeling:
    """Tests for epidemic modeling quantum twin"""

    @pytest.mark.asyncio
    async def test_model_epidemic(self, epidemic_modeling_twin, data_generator):
        """Test epidemic modeling"""
        # Generate epidemic scenario
        scenario = data_generator.generate_epidemic_scenario(
            disease="COVID-19",
            population_size=1000000
        )

        # Model epidemic
        forecast = await epidemic_modeling_twin.model_epidemic(
            disease=scenario['disease'],
            population_size=scenario['population_size']
        )

        # Assertions
        assert forecast is not None
        assert len(forecast.daily_cases) > 0
        assert forecast.peak_day > 0
        assert forecast.total_infected > 0

        logger.info(f"✅ Epidemic modeled: {scenario['disease']}")
        logger.info(f"   Peak day: {forecast.peak_day}")
        logger.info(f"   Total infected: {forecast.total_infected:,}")

    @pytest.mark.asyncio
    async def test_intervention_comparison(self, epidemic_modeling_twin):
        """Test intervention scenario comparison"""
        forecast = await epidemic_modeling_twin.model_epidemic(
            disease="INFLUENZA",
            population_size=500000
        )

        # Check intervention scenarios
        assert len(forecast.intervention_scenarios) > 0

        logger.info(f"✅ Intervention scenarios: {len(forecast.intervention_scenarios)}")


# ============================================================================
# TEST CLASS 6: HOSPITAL OPERATIONS
# ============================================================================

class TestHospitalOperations:
    """Tests for hospital operations quantum twin"""

    @pytest.mark.asyncio
    async def test_optimize_hospital_network(self, hospital_operations_twin, data_generator):
        """Test hospital network optimization"""
        # Generate hospital network
        hospitals, pending_patients = data_generator.generate_hospital_network(
            num_hospitals=8,
            num_pending_patients=50
        )

        # Optimize network
        result = await hospital_operations_twin.optimize_hospital_network(
            hospitals=hospitals,
            pending_patients=pending_patients
        )

        # Assertions
        assert result is not None
        assert len(result.patient_assignments) > 0
        assert result.overall_efficiency > 0.0

        logger.info(f"✅ Hospital network optimized")
        logger.info(f"   Assignments: {len(result.patient_assignments)}")
        logger.info(f"   Efficiency: {result.overall_efficiency:.1%}")

    @pytest.mark.asyncio
    async def test_quantum_speedup(self, hospital_operations_twin, data_generator):
        """Test quantum speedup in optimization"""
        hospitals, patients = data_generator.generate_hospital_network(
            num_hospitals=5,
            num_pending_patients=30
        )

        result = await hospital_operations_twin.optimize_hospital_network(hospitals, patients)

        # Check quantum advantage
        assert result.quantum_speedup > 1.0

        logger.info(f"✅ Quantum speedup: {result.quantum_speedup:.1f}x")


# ============================================================================
# TEST CLASS 7: CLINICAL VALIDATION
# ============================================================================

class TestClinicalValidation:
    """Tests for clinical validation framework"""

    def test_validate_predictions(self, clinical_validator):
        """Test clinical validation metrics"""
        # Generate synthetic predictions
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 10)
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 10)  # 90% accuracy
        y_scores = np.random.random(len(y_true))

        # Validate
        metrics = clinical_validator.clinical_validator.validate_predictions(
            y_true, y_pred, y_scores
        )

        # Assertions
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.sensitivity <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert 0.0 <= metrics.auc_roc <= 1.0

        logger.info(f"✅ Clinical validation metrics computed")
        logger.info(f"   Accuracy: {metrics.accuracy:.1%}")
        logger.info(f"   Sensitivity: {metrics.sensitivity:.1%}")
        logger.info(f"   Specificity: {metrics.specificity:.1%}")

    def test_benchmark_comparison(self, clinical_validator):
        """Test benchmark comparison"""
        comparison = clinical_validator.clinical_validator.compare_to_benchmark(
            quantum_performance=0.92,
            benchmark=ClinicalBenchmark.RADIOLOGIST_ACCURACY
        )

        assert comparison is not None
        assert comparison.quantum_performance == 0.92

        logger.info(f"✅ Benchmark comparison complete")
        logger.info(f"   Improvement: {comparison.improvement_percent:+.1f}%")


# ============================================================================
# TEST CLASS 8: HIPAA COMPLIANCE
# ============================================================================

class TestHIPAACompliance:
    """Tests for HIPAA compliance framework"""

    def test_encrypt_decrypt_phi(self, hipaa_framework):
        """Test PHI encryption and decryption"""
        # Test patient data
        patient_data = {
            'patient_id': 'TEST_001',
            'name': 'Test Patient',
            'age': 65,
            'diagnosis': 'NSCLC'
        }

        # Encrypt
        encrypted = hipaa_framework.encrypt_patient_data(
            patient_data=patient_data,
            phi_categories=[PHICategory.NAME, PHICategory.DIAGNOSIS],
            user_id="doctor_001",
            user_role=AccessLevel.PROVIDER
        )

        assert encrypted is not None
        assert encrypted.encrypted_data is not None

        # Decrypt
        decrypted = hipaa_framework.decrypt_patient_data(
            encrypted_phi=encrypted,
            user_id="doctor_001",
            user_role=AccessLevel.PROVIDER
        )

        assert decrypted['patient_id'] == 'TEST_001'
        assert decrypted['age'] == 65

        logger.info(f"✅ PHI encryption/decryption validated")

    def test_de_identification(self, hipaa_framework):
        """Test de-identification"""
        patient_data = {
            'patient_id': 'TEST_001',
            'name': 'John Doe',
            'age': 55,
            'diagnosis': 'Breast Cancer',
            'email': 'john.doe@example.com'
        }

        de_identified = hipaa_framework.de_identify_for_research(
            patient_data=patient_data,
            researcher_id="researcher_001"
        )

        assert de_identified is not None
        assert len(de_identified.removed_identifiers) > 0
        assert 'name' not in de_identified.de_identified_content
        assert 'email' not in de_identified.de_identified_content

        logger.info(f"✅ De-identification validated")
        logger.info(f"   Removed {len(de_identified.removed_identifiers)} identifiers")


# ============================================================================
# TEST CLASS 9: CONVERSATIONAL AI
# ============================================================================

class TestHealthcareConversationalAI:
    """Tests for healthcare conversational AI"""

    @pytest.mark.asyncio
    async def test_treatment_planning_query(self, healthcare_ai):
        """Test treatment planning query"""
        response = await healthcare_ai.process_query(
            user_message="Create treatment plan for 65-year-old woman with lung cancer",
            user_id="doctor_001",
            user_role=UserRole.PHYSICIAN
        )

        assert response is not None
        assert response.quantum_twin_used == "PersonalizedMedicineQuantumTwin"
        assert response.confidence > 0.0

        logger.info(f"✅ Treatment planning query processed")
        logger.info(f"   Confidence: {response.confidence:.1%}")

    @pytest.mark.asyncio
    async def test_multiple_intents(self, healthcare_ai):
        """Test multiple intent classification"""
        queries = [
            "Design drug for EGFR protein",
            "Analyze chest X-ray",
            "Model COVID-19 outbreak",
            "Optimize patient transfers"
        ]

        for query in queries:
            response = await healthcare_ai.process_query(
                user_message=query,
                user_id="user_001",
                user_role=UserRole.PHYSICIAN
            )

            assert response is not None
            assert response.quantum_twin_used is not None

            logger.info(f"✅ Query: '{query[:30]}...' → {response.quantum_twin_used}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple modules"""

    @pytest.mark.asyncio
    async def test_end_to_end_personalized_medicine(
        self,
        data_generator,
        personalized_medicine_twin,
        genomic_analysis_twin,
        clinical_validator
    ):
        """Test end-to-end personalized medicine workflow"""
        # 1. Generate patient
        patient = data_generator.generate_cancer_patient(cancer_type=CancerType.NSCLC)

        # 2. Analyze genomics
        variants = data_generator.generate_genetic_variants(5, CancerType.NSCLC)
        genomic_result = await genomic_analysis_twin.analyze_genomic_profile(
            patient.patient_id, variants
        )

        # 3. Create treatment plan
        treatment_plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)

        # 4. Validate
        assert genomic_result is not None
        assert treatment_plan is not None
        assert treatment_plan.confidence_score > 0.0

        logger.info(f"✅ End-to-end personalized medicine workflow complete")
        logger.info(f"   Patient: {patient.patient_id}")
        logger.info(f"   Actionable mutations: {len(genomic_result.actionable_mutations)}")
        logger.info(f"   Treatment: {treatment_plan.primary_treatment.therapy_name}")


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    async def test_treatment_planning_performance(self, personalized_medicine_twin, data_generator):
        """Benchmark treatment planning performance"""
        import time

        patient = data_generator.generate_cancer_patient()

        start = time.time()
        plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)
        duration = time.time() - start

        assert plan is not None
        assert duration < 5.0  # Should complete within 5 seconds

        logger.info(f"✅ Treatment planning performance: {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_cohort_processing(self, personalized_medicine_twin, data_generator):
        """Test processing patient cohort"""
        import time

        cohort = data_generator.generate_patient_cohort(num_patients=10)

        start = time.time()
        plans = []
        for patient in cohort:
            plan = await personalized_medicine_twin.create_personalized_treatment_plan(patient)
            plans.append(plan)
        duration = time.time() - start

        assert len(plans) == 10
        assert all(p.confidence_score > 0 for p in plans)

        logger.info(f"✅ Cohort processing: {len(cohort)} patients in {duration:.2f}s")
        logger.info(f"   Average: {duration/len(cohort):.2f}s per patient")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    """Run comprehensive test suite"""
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])
