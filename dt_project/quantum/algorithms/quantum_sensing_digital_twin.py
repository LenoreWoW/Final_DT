"""Quantum Sensing Digital Twin stub."""

import enum, numpy as np
from dataclasses import dataclass


class SensingModality(enum.Enum):
    PHASE_ESTIMATION = "phase_estimation"
    AMPLITUDE_ESTIMATION = "amplitude_estimation"
    FREQUENCY_ESTIMATION = "frequency_estimation"


class PrecisionScaling(enum.Enum):
    STANDARD_QUANTUM_LIMIT = "SQL"
    HEISENBERG_LIMIT = "HL"


class ScalingRegime(enum.Enum):
    SQL = "SQL"
    HL = "HL"


@dataclass
class SensingResult:
    measured_value: float
    precision: float
    scaling_regime: ScalingRegime = ScalingRegime.HL
    quantum_fisher_information: float = 100.0

    def cramer_rao_bound(self):
        return 1.0 / np.sqrt(self.quantum_fisher_information) if self.quantum_fisher_information > 0 else float("inf")


class QuantumSensingDigitalTwin:
    def __init__(self, num_qubits=4, modality=None):
        self.num_qubits = num_qubits
        self.modality = modality or SensingModality.PHASE_ESTIMATION

    def perform_sensing(self, true_parameter=0.5, num_shots=1000):
        precision = 1.0 / (num_shots * self.num_qubits)
        measured = true_parameter + np.random.normal(0, precision)
        qfi = float(self.num_qubits ** 2 * num_shots)
        return SensingResult(
            measured_value=float(measured),
            precision=float(precision),
            scaling_regime=ScalingRegime.HL,
            quantum_fisher_information=qfi,
        )


class QuantumSensingTheory:
    def __init__(self):
        self.standard_quantum_limit = 1.0

    def calculate_precision_limit(self, num_measurements, scaling_type=PrecisionScaling.STANDARD_QUANTUM_LIMIT):
        if scaling_type == PrecisionScaling.HEISENBERG_LIMIT:
            return 1.0 / num_measurements
        return 1.0 / np.sqrt(num_measurements)

    def quantum_advantage_factor(self, num_measurements):
        return np.sqrt(num_measurements)
