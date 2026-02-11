#!/usr/bin/env python3
"""
🧪 SYNTHETIC PATIENT DATA GENERATOR
====================================

HIPAA-compliant synthetic patient data generator for clinical testing:
- Realistic patient profiles for all 6 healthcare use cases
- Medical imaging data simulation
- Genomic variant generation
- Epidemic scenario data
- Hospital network data
- Statistically realistic clinical parameters

Data Generation Methods:
    - Demographics: Age, sex distributions matching clinical populations
    - Cancer types: Prevalence-weighted sampling
    - Biomarkers: Realistic ranges from medical literature
    - Genomic mutations: Frequency-based sampling (COSMIC database)
    - Medical images: Synthetic metadata (actual images would require ML generation)
    - Hospital operations: Realistic patient flow and resource constraints

Privacy & Compliance:
    - All data is synthetic (no real patient information)
    - HIPAA-compliant by design (no PHI)
    - Suitable for research, testing, and demonstrations
    - Follows medical literature distributions

Author: Hassan Al-Sahli
Purpose: Generate synthetic data for clinical validation testing
Reference: HEALTHCARE_FOCUS_STRATEGIC_PLAN.md - Week 2 Testing
Implementation: IMPLEMENTATION_TRACKER.md - Week 2
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import random

# Import healthcare modules
try:
    from dt_project.healthcare.personalized_medicine import (
        PatientProfile,
        CancerType
    )
    from dt_project.healthcare.drug_discovery import (
        TargetProtein,
        ProteinClass
    )
    from dt_project.healthcare.medical_imaging import (
        MedicalImage,
        ImageModality,
        AnatomicalRegion
    )
    from dt_project.healthcare.genomic_analysis import (
        GeneticVariant,
        VariantType
    )
    from dt_project.healthcare.hospital_operations import (
        Hospital,
        PendingPatient,
        AcuityLevel,
        SpecialtyType
    )
    HEALTHCARE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Healthcare modules not available: {e}")
    HEALTHCARE_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SyntheticPatientDataGenerator:
    """
    🧪 Synthetic Patient Data Generator

    Generates HIPAA-compliant synthetic patient data for all 6 healthcare use cases:
    1. Personalized Medicine - Cancer patient profiles
    2. Drug Discovery - Target protein data
    3. Medical Imaging - Medical image metadata
    4. Genomic Analysis - Genetic variant data
    5. Epidemic Modeling - Disease outbreak scenarios
    6. Hospital Operations - Hospital network and patient data

    All data is synthetic and statistically realistic based on medical literature.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        """Initialize synthetic data generator"""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Cancer prevalence (per 100,000) - SEER database
        self.cancer_prevalence = {
            CancerType.NSCLC: 0.30,  # 30% of cancers
            CancerType.BREAST: 0.25,
            CancerType.COLORECTAL: 0.15,
            CancerType.PANCREATIC: 0.10,
            CancerType.MELANOMA: 0.10,
            CancerType.PROSTATE: 0.10
        }

        # Common cancer mutations (COSMIC database frequencies)
        self.common_mutations = {
            CancerType.NSCLC: [
                {'gene': 'EGFR', 'variants': ['L858R', 'T790M', 'exon19del'], 'frequency': 0.15},
                {'gene': 'KRAS', 'variants': ['G12C', 'G12V', 'G12D'], 'frequency': 0.25},
                {'gene': 'ALK', 'variants': ['EML4-ALK'], 'frequency': 0.05},
                {'gene': 'TP53', 'variants': ['R175H', 'R248W'], 'frequency': 0.50}
            ],
            CancerType.BREAST: [
                {'gene': 'PIK3CA', 'variants': ['H1047R', 'E545K'], 'frequency': 0.35},
                {'gene': 'TP53', 'variants': ['R175H', 'R248Q'], 'frequency': 0.30},
                {'gene': 'BRCA1', 'variants': ['185delAG', '5382insC'], 'frequency': 0.05},
                {'gene': 'BRCA2', 'variants': ['6174delT'], 'frequency': 0.05}
            ],
            CancerType.COLORECTAL: [
                {'gene': 'APC', 'variants': ['R1450*', 'E1306*'], 'frequency': 0.70},
                {'gene': 'KRAS', 'variants': ['G12D', 'G13D'], 'frequency': 0.40},
                {'gene': 'TP53', 'variants': ['R273H', 'R248W'], 'frequency': 0.60},
                {'gene': 'BRAF', 'variants': ['V600E'], 'frequency': 0.10}
            ],
            CancerType.PANCREATIC: [
                {'gene': 'KRAS', 'variants': ['G12D', 'G12V'], 'frequency': 0.90},
                {'gene': 'TP53', 'variants': ['R175H', 'R248Q'], 'frequency': 0.70},
                {'gene': 'CDKN2A', 'variants': ['p16INK4a'], 'frequency': 0.90},
                {'gene': 'SMAD4', 'variants': ['Q311*'], 'frequency': 0.55}
            ],
            CancerType.MELANOMA: [
                {'gene': 'BRAF', 'variants': ['V600E', 'V600K'], 'frequency': 0.50},
                {'gene': 'NRAS', 'variants': ['Q61R', 'Q61K'], 'frequency': 0.25},
                {'gene': 'NF1', 'variants': ['frameshift'], 'frequency': 0.15},
                {'gene': 'TP53', 'variants': ['R248W'], 'frequency': 0.20}
            ],
            CancerType.PROSTATE: [
                {'gene': 'PTEN', 'variants': ['R130*', 'R173C'], 'frequency': 0.40},
                {'gene': 'TP53', 'variants': ['R175H'], 'frequency': 0.35},
                {'gene': 'SPOP', 'variants': ['F133V'], 'frequency': 0.10},
                {'gene': 'BRCA2', 'variants': ['6174delT'], 'frequency': 0.05}
            ]
        }

        # Biomarker ranges (normal and elevated)
        self.biomarker_ranges = {
            'PD-L1': {'normal': (0.0, 0.30), 'elevated': (0.30, 1.0)},
            'TMB': {'normal': (0, 6), 'elevated': (10, 50)},  # mutations/megabase
            'MSI': {'normal': (0.0, 0.30), 'elevated': (0.70, 1.0)},  # MSI-high
            'HER2': {'normal': (0, 1), 'elevated': (2, 3)},  # IHC score
            'EGFR_expression': {'normal': (0, 30), 'elevated': (50, 100)},  # % positive
            'Ki67': {'normal': (0, 20), 'elevated': (30, 90)}  # % positive
        }

        # Target proteins for drug discovery
        self.target_proteins = [
            {'id': 'EGFR', 'name': 'Epidermal Growth Factor Receptor', 'class': ProteinClass.KINASE},
            {'id': 'BRAF', 'name': 'B-Raf Proto-Oncogene', 'class': ProteinClass.KINASE},
            {'id': 'BCL2', 'name': 'B-cell lymphoma 2', 'class': ProteinClass.PROTEASE},
            {'id': 'PARP1', 'name': 'Poly(ADP-ribose) polymerase 1', 'class': ProteinClass.ENZYME},
            {'id': 'PD1', 'name': 'Programmed cell death protein 1', 'class': ProteinClass.RECEPTOR}
        ]

        logger.info("🧪 Synthetic Patient Data Generator initialized")

    def generate_cancer_patient(
        self,
        cancer_type: Optional[CancerType] = None,
        age_range: tuple = (40, 80),
        include_mutations: bool = True
    ) -> PatientProfile:
        """
        Generate synthetic cancer patient profile

        Args:
            cancer_type: Specific cancer type (random if None)
            age_range: Age range for patient
            include_mutations: Include genomic mutations

        Returns:
            PatientProfile with realistic clinical data
        """
        # Select cancer type based on prevalence
        if cancer_type is None:
            cancer_types = list(self.cancer_prevalence.keys())
            weights = list(self.cancer_prevalence.values())
            cancer_type = np.random.choice(cancer_types, p=weights)

        # Generate demographics
        age = np.random.randint(age_range[0], age_range[1] + 1)

        # Sex distribution (some cancers are sex-specific)
        if cancer_type == CancerType.BREAST:
            sex = 'F' if np.random.random() < 0.99 else 'M'
        elif cancer_type == CancerType.PROSTATE:
            sex = 'M'
        else:
            sex = 'F' if np.random.random() < 0.5 else 'M'

        # Generate genomic mutations
        genomic_mutations = []
        if include_mutations and cancer_type in self.common_mutations:
            for mutation_group in self.common_mutations[cancer_type]:
                # Include mutation based on frequency
                if np.random.random() < mutation_group['frequency']:
                    variant = np.random.choice(mutation_group['variants'])
                    genomic_mutations.append({
                        'gene': mutation_group['gene'],
                        'variant': variant,
                        'type': 'SNV' if len(variant) < 10 else 'INDEL',
                        'allele_frequency': np.random.uniform(0.3, 0.8)
                    })

        # Generate biomarkers
        biomarkers = {}
        for marker, ranges in self.biomarker_ranges.items():
            # 30% chance of elevated biomarker
            is_elevated = np.random.random() < 0.3
            range_to_use = ranges['elevated'] if is_elevated else ranges['normal']
            biomarkers[marker] = np.random.uniform(range_to_use[0], range_to_use[1])

        # Generate imaging studies (just metadata)
        imaging_studies = []
        num_scans = np.random.randint(1, 4)
        for _ in range(num_scans):
            imaging_studies.append({
                'study_id': f"IMG_{uuid.uuid4().hex[:8]}",
                'modality': np.random.choice(['CT', 'MRI', 'PET-CT']),
                'date': (datetime.now() - timedelta(days=np.random.randint(1, 180))).isoformat(),
                'findings': 'Synthetic scan data'
            })

        # Generate stage and tumor grade
        stages = ["I", "II", "IIIA", "IIIB", "IV"]
        grades = ["G1", "G2", "G3"]
        stage = np.random.choice(stages, p=[0.15, 0.25, 0.20, 0.15, 0.25])
        tumor_grade = np.random.choice(grades, p=[0.25, 0.50, 0.25])

        patient = PatientProfile(
            patient_id=f"PT_{uuid.uuid4().hex[:8]}",
            age=age,
            sex=sex,
            diagnosis=cancer_type,
            stage=stage,
            tumor_grade=tumor_grade,
            genomic_mutations=genomic_mutations,
            imaging_studies=imaging_studies,
            biomarkers=biomarkers
        )

        logger.info(f"🧪 Generated patient: {patient.patient_id} - {age}y {sex} {cancer_type.value}")
        logger.info(f"   Mutations: {len(genomic_mutations)}, Biomarkers: {len(biomarkers)}")

        return patient

    def generate_patient_cohort(
        self,
        num_patients: int,
        cancer_type: Optional[CancerType] = None
    ) -> List[PatientProfile]:
        """
        Generate cohort of synthetic cancer patients

        Args:
            num_patients: Number of patients to generate
            cancer_type: Specific cancer type (mixed if None)

        Returns:
            List of PatientProfile objects
        """
        logger.info(f"🧪 Generating cohort of {num_patients} patients...")

        cohort = []
        for i in range(num_patients):
            patient = self.generate_cancer_patient(cancer_type=cancer_type)
            cohort.append(patient)

        logger.info(f"✅ Generated {len(cohort)} synthetic patients")

        return cohort

    def generate_target_protein(
        self,
        protein_id: Optional[str] = None
    ) -> TargetProtein:
        """
        Generate synthetic target protein for drug discovery

        Args:
            protein_id: Specific protein ID (random if None)

        Returns:
            TargetProtein object
        """
        # Select protein
        if protein_id is None:
            protein_data = random.choice(self.target_proteins)
        else:
            protein_data = next(
                (p for p in self.target_proteins if p['id'] == protein_id),
                self.target_proteins[0]
            )

        # Generate synthetic protein sequence (truncated for demo)
        sequence = ''.join(
            np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=50)
        ) + "..."

        # Active site residues
        active_site = sorted(np.random.choice(range(100, 900), size=3, replace=False).tolist())

        # Known mutations
        mutations = [f"{aa}{pos}" for aa, pos in zip(
            np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=3),
            np.random.choice(range(100, 900), size=3)
        )]

        protein = TargetProtein(
            protein_id=protein_data['id'],
            protein_name=protein_data['name'],
            protein_type=protein_data['class'],
            binding_site_residues=active_site,
        )

        logger.info(f"🧪 Generated target protein: {protein.protein_id} ({protein.protein_type.value})")

        return protein

    def generate_medical_image_metadata(
        self,
        modality: Optional[ImageModality] = None,
        region: Optional[AnatomicalRegion] = None
    ) -> MedicalImage:
        """
        Generate synthetic medical image metadata

        Note: Actual image generation would require ML models (GANs, diffusion models)
        This generates realistic metadata for testing purposes

        Args:
            modality: Image modality (random if None)
            region: Anatomical region (random if None)

        Returns:
            MedicalImage object with metadata
        """
        # Select modality
        if modality is None:
            modality = random.choice(list(ImageModality))

        # Select region
        if region is None:
            region = random.choice(list(AnatomicalRegion))

        # Generate acquisition parameters
        study_date = datetime.now() - timedelta(days=np.random.randint(1, 365))

        # Modality-specific parameters
        if modality == ImageModality.CT:
            slice_thickness = np.random.uniform(1.0, 5.0)
            kvp = np.random.choice([80, 100, 120, 140])
        elif modality == ImageModality.MRI:
            slice_thickness = np.random.uniform(2.0, 8.0)
            kvp = None
        else:
            slice_thickness = np.random.uniform(1.0, 10.0)
            kvp = None

        # Clinical indication (pathology present 40% of time)
        has_pathology = np.random.random() < 0.4

        image = MedicalImage(
            image_id=f"IMG_{uuid.uuid4().hex[:12]}",
            modality=modality,
            body_part=region.value if hasattr(region, 'value') else str(region),
            image_array=np.random.rand(64, 64, 1),  # Simplified image array
            resolution=(64, 64, 1),
            patient_age=np.random.randint(20, 80),
            patient_sex=np.random.choice(['M', 'F']),
            acquisition_date=study_date
        )

        logger.info(f"🧪 Generated image: {modality.value} - {region.value} (pathology: {has_pathology})")

        return image

    def generate_genetic_variants(
        self,
        num_variants: int,
        cancer_type: CancerType
    ) -> List[GeneticVariant]:
        """
        Generate synthetic genetic variants

        Args:
            num_variants: Number of variants to generate
            cancer_type: Cancer type (affects mutation spectrum)

        Returns:
            List of GeneticVariant objects
        """
        variants = []

        if cancer_type in self.common_mutations:
            mutation_groups = self.common_mutations[cancer_type]

            for i in range(min(num_variants, len(mutation_groups) * 2)):
                group = random.choice(mutation_groups)
                variant_name = random.choice(group['variants'])

                # Determine variant type
                if 'del' in variant_name or 'ins' in variant_name:
                    var_type = VariantType.INDEL
                elif '-' in variant_name:  # Fusion
                    var_type = VariantType.FUSION
                else:
                    var_type = VariantType.SNV

                variant = GeneticVariant(
                    variant_id=f"VAR_{uuid.uuid4().hex[:8]}",
                    gene=group['gene'],
                    chromosome=f"chr{np.random.randint(1, 23)}",
                    position=np.random.randint(1000000, 200000000),
                    reference='G',
                    alternate='A',
                    variant_type=var_type,
                    variant_allele_frequency=np.random.uniform(0.3, 0.8),
                    depth=np.random.randint(50, 500),
                    quality_score=np.random.uniform(30, 60),
                    consequence=np.random.choice(['missense', 'nonsense', 'frameshift', 'synonymous'])
                )

                variants.append(variant)

        logger.info(f"🧪 Generated {len(variants)} genetic variants for {cancer_type.value}")

        return variants

    def generate_epidemic_scenario(
        self,
        disease: str = "COVID-19",
        population_size: int = 1000000
    ) -> Dict[str, Any]:
        """
        Generate synthetic epidemic scenario data

        Args:
            disease: Disease name
            population_size: Population size

        Returns:
            Dictionary with epidemic parameters
        """
        # Disease-specific parameters
        disease_params = {
            'COVID-19': {'R0': (2.5, 4.0), 'incubation': (3, 7), 'infectious': (7, 14)},
            'INFLUENZA': {'R0': (1.2, 1.8), 'incubation': (1, 3), 'infectious': (5, 7)},
            'MEASLES': {'R0': (12, 18), 'incubation': (8, 12), 'infectious': (4, 7)}
        }

        params = disease_params.get(disease, disease_params['COVID-19'])

        scenario = {
            'disease': disease,
            'population_size': population_size,
            'initial_infected': np.random.randint(10, 100),
            'R0': np.random.uniform(*params['R0']),
            'incubation_period_days': np.random.uniform(*params['incubation']),
            'infectious_period_days': np.random.uniform(*params['infectious']),
            'hospitalization_rate': np.random.uniform(0.05, 0.20),
            'death_rate': np.random.uniform(0.005, 0.02),
            'vaccination_rate': np.random.uniform(0.0, 0.70),
            'simulation_days': 180
        }

        logger.info(f"🧪 Generated epidemic scenario: {disease} (R0={scenario['R0']:.2f})")

        return scenario

    def generate_hospital_network(
        self,
        num_hospitals: int = 8,
        num_pending_patients: int = 50
    ) -> tuple[List[Hospital], List[PendingPatient]]:
        """
        Generate synthetic hospital network data

        Args:
            num_hospitals: Number of hospitals in network
            num_pending_patients: Number of patients needing assignment

        Returns:
            Tuple of (hospitals, pending_patients)
        """
        # Hospital names
        hospital_names = [
            "University Medical Center", "St. Mary's Hospital", "Regional Health Center",
            "Community Hospital", "Memorial Hospital", "Central Medical Center",
            "North Shore Hospital", "Riverside Medical Center"
        ]

        hospitals = []
        for i in range(num_hospitals):
            # Generate random coordinates (within 50km radius)
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, 50)
            lat = 40.7128 + (distance * np.cos(angle)) / 111  # NY latitude as base
            lon = -74.0060 + (distance * np.sin(angle)) / 111

            # Random capacities
            total_beds = np.random.randint(100, 500)
            occupied = np.random.randint(int(total_beds * 0.5), int(total_beds * 0.9))

            icu_beds = int(total_beds * 0.1)  # 10% ICU beds
            available_beds = total_beds - occupied
            available_icu = max(0, icu_beds - int(occupied * 0.1))
            current_occupancy = occupied / total_beds
            icu_occupancy = (icu_beds - available_icu) / icu_beds if icu_beds > 0 else 0.0
            
            hospital = Hospital(
                hospital_id=f"HOSP_{i+1:02d}",
                hospital_name=hospital_names[i] if i < len(hospital_names) else f"Hospital {i+1}",
                location=(lat, lon),
                total_beds=total_beds,
                icu_beds=icu_beds,
                available_beds=available_beds,
                available_icu=available_icu,
                specialties=[
                    spec for spec in SpecialtyType
                    if np.random.random() < 0.7  # 70% chance of having each specialty
                ],
                current_occupancy=current_occupancy,
                icu_occupancy=icu_occupancy,
            )

            hospitals.append(hospital)

        # Generate pending patients
        pending_patients = []
        for i in range(num_pending_patients):
            # Random patient location
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, 50)
            lat = 40.7128 + (distance * np.cos(angle)) / 111
            lon = -74.0060 + (distance * np.sin(angle)) / 111

            patient = PendingPatient(
                patient_id=f"PEND_{i+1:03d}",
                current_location=f"HOSP_{np.random.randint(1, num_hospitals + 1):02d}",
                acuity=random.choice(list(AcuityLevel)),
                specialty_needed=random.choice(list(SpecialtyType)),
                requires_icu=np.random.random() < 0.2,
                requires_ventilator=np.random.random() < 0.1,
                estimated_los_days=np.random.randint(1, 14)
            )

            pending_patients.append(patient)

        logger.info(f"🧪 Generated hospital network: {num_hospitals} hospitals, {num_pending_patients} patients")

        return hospitals, pending_patients


# Convenience functions
def generate_test_patient(cancer_type: Optional[CancerType] = None) -> PatientProfile:
    """Generate single test patient"""
    generator = SyntheticPatientDataGenerator()
    return generator.generate_cancer_patient(cancer_type=cancer_type)


def generate_test_cohort(num_patients: int = 100) -> List[PatientProfile]:
    """Generate test patient cohort"""
    generator = SyntheticPatientDataGenerator()
    return generator.generate_patient_cohort(num_patients)
