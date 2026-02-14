# Test Suite Status Report

**Date:** 2026-02-15
**Commit:** `4683521` (Fix 6 audit issues)
**Python:** 3.9 | **pytest** with asyncio-mode=auto

## Summary

| Metric | Count |
|--------|-------|
| **Collected** | 431 |
| **Passed** | 242 |
| **Failed** | 135 |
| **Skipped** | 42 |
| **Errors** (setup) | 25 |
| **Collection errors** | 0 |

---

## Errors (25) — Fixture / Setup Failures

These tests never ran because a shared fixture or setup step failed.

### `test_healthcare_comprehensive.py` — 15 errors

All caused by `AttributeError: NSCLC` in the `personalized_medicine_twin` fixture.
The `CancerType` enum stub does not define the `NSCLC` member that the fixture expects.

| Test class | Tests affected |
|------------|---------------|
| `TestPersonalizedMedicine` | 3 (treatment plan, cancer types, advantage tracking) |
| `TestDrugDiscovery` | 2 (candidates, properties) |
| `TestMedicalImaging` | 2 (analyze, modalities) |
| `TestGenomicAnalysis` | 2 (profile, pathway) |
| `TestEpidemicModeling` | 1 (model epidemic) |
| `TestHospitalOperations` | 2 (optimize, speedup) |
| `TestIntegration` | 1 (end-to-end personalized medicine) |
| `TestPerformance` | 2 (treatment planning, cohort) |

**Root cause:** `CancerType` enum in `dt_project/healthcare/personalized_medicine.py` stub is missing variant `NSCLC` (and likely `BREAST`). Adding the enum members would unblock all 15 tests.

### `test_e2e_integration.py` — 9 errors

All caused by `assert 404 == 200` in the `created_twin` fixture — the POST to `/api/twins/` returns 404 during test setup.

| Test class | Tests affected |
|------------|---------------|
| `TestConversationTwinCreation` | 2 (existing twin, history) |
| `TestSimulationQuantumBackend` | 1 (simulation results) |
| `TestTwinQuery` | 5 (prediction, optimization, understanding, comparison, explicit type) |
| `TestResponseTimes` | 1 (simulation under 10s) |

**Root cause:** The fixture creates a twin via the API, but the endpoint returns 404. Likely the test's request body doesn't match the current API schema, or the test expects a route that was refactored.

### `test_comprehensive_quantum_platform.py` — 1 error

`AttributeError: AUTOMATIC` in `TestMasterFactory::test_automatic_processing`.

**Root cause:** `ProcessingMode` enum in the `quantum_digital_twin_factory_master` stub is missing the `AUTOMATIC` member.

---

## Failures (135) — By Root Cause Category

### Category A: Missing Methods on Stub Classes (89 failures)

The `dt_project/` shim package provides importable classes with minimal stub implementations. Tests that call methods beyond the stub surface fail at runtime.

| Stub class | Missing method / attribute | Failures |
|------------|---------------------------|----------|
| `AcademicStatisticalValidator` | `validate_performance_claim` | 10 |
| `AcademicStatisticalValidator` | `validate_fidelity_claim` | 3 |
| `AcademicStatisticalValidator` | `validate_sensing_precision`, `validate_optimization_speedup`, `benchmarks` | 3 |
| `TreeTensorNetwork` | `leaf_ids`, `contract_network`, `optimize_bond_dimensions` | 5 |
| `QuantumSensingDigitalTwin` | `theory`, `validate_quantum_advantage`, `n_sensors`, `sensing_history`, `generate_sensing_report` | 10 |
| `QuantumOptimizationDigitalTwin` | `generate_optimization_problem` | 2 |
| `ProvenQuantumAdvantageValidator` | `validate_quantum_advantages` | 2 |
| `QuantumHardwareOrchestrator` | `get_execution_statistics`, `connectors` | 3 |
| `AerSimulatorConnector` | `simulator` | 1 |
| `HIPAAComplianceFramework` | `encrypt_patient_data`, `de_identify_for_research` | 4 |
| `HealthcareConversationalAI` | `process_query` | 2 |
| `ClinicalValidationFramework` | `clinical_validator` | 2 |
| `ClinicalValidator` | `validate_predictions`, `compare_to_benchmark` | 2 |
| `ClinicalBenchmark` | `RADIOLOGIST_ACCURACY` class attribute | 1 |
| `RegulatoryValidator` | `validate_fda_part_11` | 1 |
| `QuantumDigitalTwinFactoryMaster` | `get_factory_statistics` | 1 |
| `IntelligentQuantumMapper` | `complexity_analyzer`, `advantage_predictor` | 2 |
| `QuantumFrameworkComparator` | `measure_performance` | 1 |
| `EpidemicForecast` | `intervention_scenarios` | 1 |
| `SensingResult` | `modality` | 1 |
| `PerformanceBenchmark` | `cern_fidelity` | 1 |
| `MLResult` | `training_losses` | 1 |
| `AlgorithmResult` | `classical_time` | 1 |
| `AthletePerformanceDigitalTwin` | `run_performance_analysis` | 1 |

**Resolution:** Expand stub classes with the expected methods/attributes. These are not collection errors — each missing method is a targeted stub gap.

### Category B: Missing Factory Functions / Imports (16 failures)

Tests import factory functions or specific names that the stubs don't export.

| Module | Missing export | Failures |
|--------|---------------|----------|
| `distributed_quantum_system` | `create_distributed_quantum_system` | 7 |
| `pennylane_quantum_ml` | `create_quantum_ml_classifier` | 2 |
| `nisq_hardware_integration` | `create_nisq_integrator`, `NISQHardwareIntegrator` | 3 |
| `uncertainty_quantification` | `UncertaintyQuantificationFramework`, `VirtualQPU`, `create_uq_framework` | 3 |
| `quantum_sensing_digital_twin` | `create_quantum_sensing_twin` | 1 |
| `neural_quantum_digital_twin` | `create_neural_quantum_twin` | 1 |
| `error_matrix_digital_twin` | `create_error_matrix_twin` | 1 |
| `hipaa_compliance` | `AuditAction` | 1 |

### Category C: Constructor Signature Mismatches (14 failures)

Tests pass keyword arguments that the stub `__init__` doesn't accept.

| Class | Unexpected kwarg | Failures |
|-------|-----------------|----------|
| `create_ttn_for_benchmarking()` | `max_bond_dim` (expects `bond_dimension`) | 8 |
| `ConfigManager` | `config_path` | 5 |
| `QuantumSensingDigitalTwin` | `n_qubits` | 2 |
| `MedicalImagingQuantumTwin` | `modality` | 2 |
| `TreeTensorNetwork` | `physical_indices` | 3 |
| `QuantumSensingDigitalTwin` | `circuit_depth` | 2 |
| `HIPAAComplianceFramework` | `enable_encryption` | 1 |

### Category D: Enum Members Missing (6 failures)

| Enum | Missing member | Failures |
|------|---------------|----------|
| `CancerType` | `NSCLC`, `BREAST` | 6 (healthcare_basic) |
| `QuantumProvider` | `AMAZON`, `PROVIDER` variants | 2 |
| `AlgorithmType` | `GROVER_SEARCH` | 1 |
| `SensingModality` | `FORCE_DETECTION` | 1 |

### Category E: Return Value / Assertion Mismatches (10 failures)

Tests expect specific return formats that stubs don't produce.

| Issue | Failures |
|-------|----------|
| Algorithm `name` field returns snake_case, test expects Title Case | 4 |
| `chat_with_quantum_ai()` doesn't accept `session_id` kwarg | 1 |
| `DID NOT RAISE <class 'ValueError'>` — validation stubs too permissive | 2 |
| Misc assertion mismatches (`assert None == {}`, precision, etc.) | 3 |

---

## Skipped Tests (42) — By Reason

### Archived / Refactored Modules (13 skips)

These modules were intentionally archived or refactored. Tests are marked `skipIf` / `skipUnless`.

| Test file | Reason |
|-----------|--------|
| `test_api_routes_comprehensive.py` | web_interface archived (1) |
| `test_authentication_security.py` | web_interface archived (1) |
| `test_web_interface_core.py` | web_interface archived (1) |
| `test_web_interface.py` | GraphQL compatibility issues (1) |
| `test_database_integration.py` | database_integration refactored (1) |
| `test_quantum_consciousness_bridge.py` | archived as experimental (1) |
| `test_quantum_innovations.py` | archived as experimental (1) |
| `test_quantum_multiverse_network.py` | archived as experimental (1) |
| `test_quantum_digital_twin_core.py` | refactored with different interface (1) |
| `test_quantum_digital_twin_validation.py` | memory-intensive (1) |
| `test_real_quantum_digital_twins.py` | archived (1) |
| `test_working_quantum_digital_twins.py` | archived (1) |

### Missing Optional Dependencies (22 skips)

| Test file | Reason | Skips |
|-----------|--------|-------|
| `test_enhanced_quantum_digital_twin.py` | "Enhanced DT not available" | 18 |
| `test_enhanced_quantum_digital_twin.py` | "Tensor networks not available" | 4 |

### Framework Not Available (6 skips)

| Test file | Reason | Skips |
|-----------|--------|-------|
| `test_framework_comparison.py` | "Qiskit not available" | 2 |
| `test_framework_comparison.py` | "PennyLane not available" | 2 |
| `test_framework_comparison.py` | "Framework/Comprehensive study not available" | 2 |

### External Tool Not Installed (1 skip)

| Test file | Reason |
|-----------|--------|
| `test_e2e_quantum_platform.py` | Playwright not installed |

---

## Passed Tests (242) — By File

| Test file | Passed |
|-----------|--------|
| `test_backend_api.py` | 37 |
| `test_benchmark_api.py` | 16 |
| `test_twin_generation_engine.py` | 27 |
| `test_e2e_integration.py` | 26 |
| `test_error_handling.py` | 24 |
| `test_unified_config.py` | 15 |
| `test_proven_quantum_advantage.py` | 12 |
| `test_quantum_ai_simple.py` | 13 |
| `test_real_quantum_algorithms.py` | 12 |
| `test_academic_validation.py` | 8 |
| `test_comprehensive_quantum_platform.py` | 8 |
| `test_healthcare_comprehensive.py` | 7 |
| `test_quantum_sensing_digital_twin.py` | 4 |
| `test_real_quantum_hardware_integration.py` | 4 |
| `test_framework_comparison.py` | 3 |
| `test_healthcare_basic.py` | 3 |
| `test_independent_study_validation.py` | 3 |
| `test_phase3_comprehensive.py` | 3 |
| `test_config.py` | 2 |
| `test_quantum_ai.py` | 2 |
| `test_tree_tensor_network.py` | 2 |
| Other files | 11 |

---

## Warnings (3)

1. **PydanticDeprecatedSince20** — `backend/models/schemas.py` uses class-based `Config` instead of pydantic v2 `model_config`. This is a pre-existing library migration issue, not introduced by this commit.
2. **DeprecationWarning** — `datetime.utcnow()` warnings: **all eliminated** from project code; any remaining come from third-party libraries.

---

## Recommendations for Future Work

1. **Quick wins (unblock ~21 tests):** Add `NSCLC` and `BREAST` to `CancerType` enum, `AUTOMATIC` to `ProcessingMode` enum.
2. **Medium effort (unblock ~30 tests):** Add missing factory functions (`create_distributed_quantum_system`, `create_nisq_integrator`, etc.) to stubs.
3. **Larger effort (unblock ~89 tests):** Expand stub method implementations to match full test expectations.
4. **Low priority:** Fix constructor signature mismatches (keyword argument names).
5. **Migrate pydantic:** Update `backend/models/schemas.py` to use `model_config` dict instead of inner `Config` class.
