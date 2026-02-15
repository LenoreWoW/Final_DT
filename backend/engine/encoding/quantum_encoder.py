"""
Quantum Encoding Engine

Encodes extracted system components into quantum representations:
- Entities → Qubits
- States → Amplitudes
- Relationships → Entanglement
- Rules → Quantum Gates
- Constraints → Measurement Conditions

This prepares the system for quantum simulation.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.models.schemas import (
    Entity,
    Relationship,
    Rule,
    Constraint,
    ExtractedSystem,
    AlgorithmType,
)


class EncodingStrategy(str, Enum):
    """Quantum encoding strategies."""
    AMPLITUDE = "amplitude"  # Encode values in amplitudes
    ANGLE = "angle"  # Encode values as rotation angles
    BASIS = "basis"  # Encode in computational basis states
    HYBRID = "hybrid"  # Combination of strategies


class EntanglementStructure(str, Enum):
    """Types of entanglement structures."""
    LINEAR = "linear"  # Chain of entangled qubits
    STAR = "star"  # Central qubit entangled with all others
    FULL = "full"  # All-to-all entanglement
    HIERARCHICAL = "hierarchical"  # Tree-like structure
    CUSTOM = "custom"  # Based on relationship graph


@dataclass
class QubitAllocation:
    """Allocation of qubits to system components."""
    entity_qubits: Dict[str, List[int]] = field(default_factory=dict)
    state_qubits: Dict[str, int] = field(default_factory=dict)
    ancilla_qubits: List[int] = field(default_factory=list)
    total_qubits: int = 0


@dataclass
class GateSequence:
    """A sequence of quantum gates."""
    gates: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0


@dataclass
class QuantumEncoding:
    """Complete quantum encoding of a system."""
    qubit_allocation: QubitAllocation
    initial_state: List[complex]
    gate_sequences: Dict[str, GateSequence]  # rule_id -> gates
    entanglement_map: List[Tuple[int, int]]  # pairs of entangled qubits
    measurement_conditions: Dict[str, List[int]]  # constraint_id -> qubits to measure
    encoding_strategy: EncodingStrategy
    estimated_circuit_depth: int
    estimated_gate_count: int


class QuantumEncoder:
    """
    Encodes extracted systems into quantum representations.
    
    This creates the quantum circuits that will simulate the digital twin.
    """
    
    # Qubits per entity type (can be tuned based on precision needs)
    QUBITS_PER_ENTITY_TYPE = {
        "patient": 4,
        "tumor": 3,
        "drug": 3,
        "athlete": 4,
        "race": 3,
        "unit": 5,
        "terrain": 3,
        "fire": 4,
        "species": 3,
        "stock": 3,
        "portfolio": 4,
        "default": 3,
    }
    
    # Qubits per property (for state encoding)
    QUBITS_PER_PROPERTY = 2
    
    def encode(
        self,
        system: ExtractedSystem,
        strategy: EncodingStrategy = EncodingStrategy.HYBRID,
        max_qubits: int = 50,
    ) -> QuantumEncoding:
        """
        Encode an extracted system into quantum representation.
        
        Args:
            system: The extracted system to encode
            strategy: Encoding strategy to use
            max_qubits: Maximum number of qubits to use
            
        Returns:
            QuantumEncoding with all quantum representations
        """
        # Allocate qubits
        allocation = self._allocate_qubits(system, max_qubits)
        
        # Create initial state
        initial_state = self._create_initial_state(system, allocation, strategy)
        
        # Create gate sequences for rules
        gate_sequences = self._create_gate_sequences(system.rules, allocation)
        
        # Create entanglement map from relationships
        entanglement_map = self._create_entanglement_map(
            system.relationships,
            allocation
        )
        
        # Create measurement conditions from constraints
        measurement_conditions = self._create_measurement_conditions(
            system.constraints,
            allocation
        )
        
        # Estimate circuit metrics
        total_gates = sum(len(seq.gates) for seq in gate_sequences.values())
        max_depth = max((seq.depth for seq in gate_sequences.values()), default=0)
        
        return QuantumEncoding(
            qubit_allocation=allocation,
            initial_state=initial_state,
            gate_sequences=gate_sequences,
            entanglement_map=entanglement_map,
            measurement_conditions=measurement_conditions,
            encoding_strategy=strategy,
            estimated_circuit_depth=max_depth + len(entanglement_map),
            estimated_gate_count=total_gates + len(entanglement_map) * 2,
        )
    
    def _allocate_qubits(
        self,
        system: ExtractedSystem,
        max_qubits: int
    ) -> QubitAllocation:
        """Allocate qubits to entities and properties."""
        allocation = QubitAllocation()
        current_qubit = 0
        
        for entity in system.entities:
            # Get qubit count for this entity type
            entity_type = entity.type.lower()
            base_qubits = self.QUBITS_PER_ENTITY_TYPE.get(
                entity_type,
                self.QUBITS_PER_ENTITY_TYPE["default"]
            )
            
            # Add qubits for properties
            property_qubits = len(entity.properties) * self.QUBITS_PER_PROPERTY
            total_entity_qubits = base_qubits + property_qubits
            
            # Check if we have room
            if current_qubit + total_entity_qubits > max_qubits:
                # Scale down if necessary
                total_entity_qubits = min(
                    total_entity_qubits,
                    max_qubits - current_qubit
                )
                if total_entity_qubits <= 0:
                    break
            
            # Allocate
            qubit_range = list(range(current_qubit, current_qubit + total_entity_qubits))
            allocation.entity_qubits[entity.id] = qubit_range
            current_qubit += total_entity_qubits
            
            # Map properties to specific qubits
            for i, prop in enumerate(entity.properties.keys()):
                prop_qubit_idx = base_qubits + i * self.QUBITS_PER_PROPERTY
                if prop_qubit_idx < len(qubit_range):
                    allocation.state_qubits[f"{entity.id}_{prop}"] = qubit_range[prop_qubit_idx]
        
        # Add ancilla qubits if room
        ancilla_count = min(3, max_qubits - current_qubit)
        if ancilla_count > 0:
            allocation.ancilla_qubits = list(range(current_qubit, current_qubit + ancilla_count))
            current_qubit += ancilla_count
        
        allocation.total_qubits = current_qubit
        
        return allocation
    
    def _create_initial_state(
        self,
        system: ExtractedSystem,
        allocation: QubitAllocation,
        strategy: EncodingStrategy
    ) -> List[complex]:
        """Create the initial quantum state.

        For large qubit counts we store a compact symbolic representation
        (first *MAX_STATEVECTOR_DIM* amplitudes) rather than a full 2^n
        vector, which would be prohibitively large.
        """
        n_qubits = allocation.total_qubits
        if n_qubits == 0:
            return [complex(1, 0)]

        # Cap the statevector dimension to keep encoding fast.  The full
        # simulation uses Qiskit Aer which manages its own statevector
        # internally -- this is only a descriptive representation.
        MAX_STATEVECTOR_DIM = 2 ** min(n_qubits, 12)  # max 4096 entries

        dim = MAX_STATEVECTOR_DIM

        # Create equal superposition (Hadamard on all qubits)
        amplitude = complex(1 / math.sqrt(dim), 0)
        initial_state = [amplitude] * dim

        # Encode entity values if available (amplitude encoding)
        if strategy in [EncodingStrategy.AMPLITUDE, EncodingStrategy.HYBRID]:
            for entity in system.entities:
                for prop, value in entity.properties.items():
                    if value is not None and isinstance(value, (int, float)):
                        # Normalize value to [0, 1]
                        normalized = min(max(float(value) / 100, 0), 1)

                        # Modify amplitudes for corresponding qubits
                        qubit_key = f"{entity.id}_{prop}"
                        if qubit_key in allocation.state_qubits:
                            qubit_idx = allocation.state_qubits[qubit_key]
                            if qubit_idx >= min(n_qubits, 12):
                                continue  # Skip qubits beyond our compact representation
                            # Bias amplitudes based on value
                            for i in range(dim):
                                if (i >> qubit_idx) & 1:  # Qubit is |1>
                                    initial_state[i] *= complex(normalized, 0)
                                else:  # Qubit is |0>
                                    initial_state[i] *= complex(1 - normalized, 0)

            # Renormalize
            norm = math.sqrt(sum(abs(a)**2 for a in initial_state))
            if norm > 0:
                initial_state = [a / norm for a in initial_state]
            else:
                # Edge case: all amplitudes collapsed to zero (e.g. normalized == 0
                # for every qubit).  Fall back to uniform superposition.
                uniform_amp = complex(1 / math.sqrt(dim), 0)
                initial_state = [uniform_amp] * dim

        return initial_state
    
    def _create_gate_sequences(
        self,
        rules: List[Rule],
        allocation: QubitAllocation
    ) -> Dict[str, GateSequence]:
        """Create quantum gate sequences for rules."""
        gate_sequences = {}
        
        for rule in rules:
            gates = []
            depth = 0
            
            # Different gate patterns based on rule type
            if rule.type == "physiology":
                # Rotation gates for continuous dynamics
                for qubit_list in allocation.entity_qubits.values():
                    for qubit in qubit_list[:2]:  # Use first 2 qubits
                        gates.append({
                            "type": "RY",
                            "qubit": qubit,
                            "angle": 0.1,  # Small rotation per time step
                        })
                depth = 1
                
            elif rule.type == "biology":
                # Controlled rotations for biological processes
                qubits = list(allocation.state_qubits.values())
                if len(qubits) >= 2:
                    gates.append({
                        "type": "CRY",
                        "control": qubits[0],
                        "target": qubits[1],
                        "angle": 0.2,
                    })
                    depth = 2
                    
            elif rule.type == "military":
                # Two-qubit gates for interactions
                for entity_id, qubits in allocation.entity_qubits.items():
                    if len(qubits) >= 2:
                        gates.append({
                            "type": "CNOT",
                            "control": qubits[0],
                            "target": qubits[1],
                        })
                        gates.append({
                            "type": "RZ",
                            "qubit": qubits[0],
                            "angle": 0.15,
                        })
                depth = len(allocation.entity_qubits)
                
            elif rule.type == "physics":
                # Time evolution simulation
                for i, (entity_id, qubits) in enumerate(allocation.entity_qubits.items()):
                    for qubit in qubits:
                        gates.append({
                            "type": "RX",
                            "qubit": qubit,
                            "angle": 0.05 * (i + 1),
                        })
                depth = max(len(q) for q in allocation.entity_qubits.values()) if allocation.entity_qubits else 0
                
            else:
                # Generic evolution
                for qubit in range(min(allocation.total_qubits, 10)):
                    gates.append({
                        "type": "RY",
                        "qubit": qubit,
                        "angle": 0.1,
                    })
                depth = 1
            
            gate_sequences[rule.id] = GateSequence(gates=gates, depth=depth)
        
        return gate_sequences
    
    def _create_entanglement_map(
        self,
        relationships: List[Relationship],
        allocation: QubitAllocation
    ) -> List[Tuple[int, int]]:
        """Create entanglement structure from relationships."""
        entanglement = []
        
        for rel in relationships:
            source_qubits = allocation.entity_qubits.get(rel.source_id, [])
            target_qubits = allocation.entity_qubits.get(rel.target_id, [])
            
            if source_qubits and target_qubits:
                # Entangle first qubit of each entity
                entanglement.append((source_qubits[0], target_qubits[0]))
                
                # For strong relationships, add more entanglement
                if rel.strength > 0.7:
                    for sq, tq in zip(source_qubits[1:2], target_qubits[1:2]):
                        entanglement.append((sq, tq))
        
        return entanglement
    
    def _create_measurement_conditions(
        self,
        constraints: List[Constraint],
        allocation: QubitAllocation
    ) -> Dict[str, List[int]]:
        """Create measurement conditions for constraints."""
        conditions = {}
        
        for constraint in constraints:
            # Determine which qubits to measure for this constraint
            qubits_to_measure = []
            
            if constraint.type == "budget":
                # Measure resource-related qubits
                for entity_id, qubits in allocation.entity_qubits.items():
                    if "resource" in entity_id or "cost" in entity_id:
                        qubits_to_measure.extend(qubits[:2])
                        
            elif constraint.type == "time":
                # Measure state qubits
                qubits_to_measure = list(allocation.state_qubits.values())[:3]
                
            else:
                # Measure ancilla qubits for general constraints
                qubits_to_measure = allocation.ancilla_qubits[:2]
            
            if qubits_to_measure:
                conditions[constraint.id] = qubits_to_measure
        
        return conditions
    
    def estimate_resources(self, encoding: QuantumEncoding) -> Dict[str, Any]:
        """Estimate computational resources needed."""
        return {
            "qubits": encoding.qubit_allocation.total_qubits,
            "circuit_depth": encoding.estimated_circuit_depth,
            "gate_count": encoding.estimated_gate_count,
            "entanglement_pairs": len(encoding.entanglement_map),
            "measurement_count": sum(len(m) for m in encoding.measurement_conditions.values()),
            "estimated_shots": 1000,
            "estimated_time_seconds": (
                encoding.estimated_circuit_depth * 0.001 * 1000  # depth * gate_time * shots
            ),
            "classical_equivalent_hours": (
                2 ** encoding.qubit_allocation.total_qubits * 0.000001  # Exponential scaling
            ),
        }


# Singleton instance
encoder = QuantumEncoder()


def encode_system(
    system: ExtractedSystem,
    strategy: EncodingStrategy = EncodingStrategy.HYBRID,
    max_qubits: int = 50,
) -> QuantumEncoding:
    """Convenience function to encode a system."""
    return encoder.encode(system, strategy, max_qubits)

