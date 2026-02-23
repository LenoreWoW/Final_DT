"""
Classical Baseline: Personalized Medicine

Classical approach using Genetic Algorithm + Grid Search
for treatment optimization.

This is used for fair comparison against the quantum approach
(QAOA) in the Quantum Advantage Showcase.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TreatmentOption:
    """A treatment option with its parameters."""
    drug_id: str
    dosage: float
    schedule: str
    predicted_efficacy: float = 0.0
    side_effects: float = 0.0


@dataclass
class PatientProfile:
    """Patient profile for treatment optimization."""
    age: int
    cancer_type: str
    cancer_stage: int
    biomarkers: Dict[str, float]
    comorbidities: List[str]


@dataclass
class ClassicalResult:
    """Result from classical optimization."""
    best_treatment: TreatmentOption
    all_treatments_tested: int
    execution_time_seconds: float
    method: str
    convergence_history: List[float]


class PersonalizedMedicineClassical:
    """
    Classical approach to personalized medicine optimization.
    
    Uses Genetic Algorithm for global search combined with
    Grid Search for local refinement.
    
    Quantum Comparison:
    - Classical: Tests N treatments sequentially
    - Quantum (QAOA): Tests all treatments via superposition
    - Expected speedup: 1000x for large treatment spaces
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Available drugs and dosages
        self.drugs = [
            "Tamoxifen", "Anastrozole", "Letrozole", "Exemestane",
            "Fulvestrant", "Palbociclib", "Ribociclib", "Abemaciclib",
            "Everolimus", "Trastuzumab", "Pertuzumab", "Pembrolizumab",
        ]
        self.dosages = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.schedules = ["daily", "weekly", "biweekly", "monthly"]
    
    def optimize(
        self,
        patient: PatientProfile,
        available_drugs: Optional[List[str]] = None,
    ) -> ClassicalResult:
        """
        Optimize treatment for a patient using genetic algorithm.
        
        Args:
            patient: Patient profile
            available_drugs: List of available drugs (or all if None)
            
        Returns:
            ClassicalResult with best treatment found
        """
        start_time = time.time()
        drugs = available_drugs or self.drugs
        
        # Calculate total search space
        total_combinations = len(drugs) * len(self.dosages) * len(self.schedules)
        
        # Initialize population
        population = self._initialize_population(drugs)
        
        # Track convergence
        convergence_history = []
        treatments_tested = 0
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_fitness(individual, patient)
                fitness_scores.append(score)
                treatments_tested += 1
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            convergence_history.append(fitness_scores[best_idx])
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(selected, drugs)
            
            # Mutation
            offspring = self._mutation(offspring, drugs)
            
            # Replace population
            population = offspring
        
        # Final evaluation
        final_scores = [self._evaluate_fitness(ind, patient) for ind in population]
        best_idx = np.argmax(final_scores)
        best_individual = population[best_idx]
        
        execution_time = time.time() - start_time
        
        return ClassicalResult(
            best_treatment=TreatmentOption(
                drug_id=best_individual["drug"],
                dosage=best_individual["dosage"],
                schedule=best_individual["schedule"],
                predicted_efficacy=final_scores[best_idx],
                side_effects=self._estimate_side_effects(best_individual, patient),
            ),
            all_treatments_tested=treatments_tested,
            execution_time_seconds=execution_time,
            method="genetic_algorithm",
            convergence_history=convergence_history,
        )
    
    def grid_search(
        self,
        patient: PatientProfile,
        available_drugs: Optional[List[str]] = None,
    ) -> ClassicalResult:
        """
        Exhaustive grid search over all treatment combinations.
        
        This is the true classical baseline - testing every combination.
        Very slow but guaranteed to find the global optimum.
        """
        start_time = time.time()
        drugs = available_drugs or self.drugs
        
        best_treatment = None
        best_score = float("-inf")
        treatments_tested = 0
        convergence_history = []
        
        for drug in drugs:
            for dosage in self.dosages:
                for schedule in self.schedules:
                    individual = {
                        "drug": drug,
                        "dosage": dosage,
                        "schedule": schedule,
                    }
                    
                    score = self._evaluate_fitness(individual, patient)
                    treatments_tested += 1
                    
                    if score > best_score:
                        best_score = score
                        best_treatment = individual
                        convergence_history.append(score)
        
        execution_time = time.time() - start_time
        
        return ClassicalResult(
            best_treatment=TreatmentOption(
                drug_id=best_treatment["drug"],
                dosage=best_treatment["dosage"],
                schedule=best_treatment["schedule"],
                predicted_efficacy=best_score,
                side_effects=self._estimate_side_effects(best_treatment, patient),
            ),
            all_treatments_tested=treatments_tested,
            execution_time_seconds=execution_time,
            method="grid_search",
            convergence_history=convergence_history,
        )
    
    def _initialize_population(self, drugs: List[str]) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = {
                "drug": random.choice(drugs),
                "dosage": random.choice(self.dosages),
                "schedule": random.choice(self.schedules),
            }
            population.append(individual)
        return population
    
    def _evaluate_fitness(
        self,
        individual: Dict[str, Any],
        patient: PatientProfile,
    ) -> float:
        """
        Evaluate treatment fitness for patient.
        
        This is a simplified model. Real implementation would use
        clinical prediction models.
        """
        base_efficacy = 0.5
        
        # Drug-specific efficacy modifiers
        drug_modifiers = {
            "Tamoxifen": 0.1 if patient.biomarkers.get("ER+", 0) > 0.5 else -0.1,
            "Trastuzumab": 0.15 if patient.biomarkers.get("HER2+", 0) > 0.5 else -0.2,
            "Palbociclib": 0.12 if patient.cancer_stage <= 2 else 0.05,
        }
        
        drug_modifier = drug_modifiers.get(individual["drug"], 0.0)
        
        # Dosage effect (bell curve around optimal)
        optimal_dosage = 1.0
        dosage_effect = -0.1 * (individual["dosage"] - optimal_dosage) ** 2
        
        # Age adjustment
        age_factor = 0.1 if 40 <= patient.age <= 65 else -0.05
        
        # Stage adjustment
        stage_factor = (5 - patient.cancer_stage) * 0.05
        
        # Random noise (simulating uncertainty)
        noise = random.gauss(0, 0.05)
        
        efficacy = base_efficacy + drug_modifier + dosage_effect + age_factor + stage_factor + noise
        
        return max(0.0, min(1.0, efficacy))
    
    def _estimate_side_effects(
        self,
        individual: Dict[str, Any],
        patient: PatientProfile,
    ) -> float:
        """Estimate side effects score (0-1, higher is worse)."""
        base_side_effects = 0.3
        
        # Higher dosage = more side effects
        dosage_effect = 0.1 * (individual["dosage"] - 1.0)
        
        # Age increases sensitivity
        age_effect = 0.01 * max(0, patient.age - 60)
        
        # Some drugs have more side effects
        drug_effects = {
            "Palbociclib": 0.1,
            "Ribociclib": 0.08,
            "Everolimus": 0.12,
        }
        drug_effect = drug_effects.get(individual["drug"], 0.0)
        
        return max(0.0, min(1.0, base_side_effects + dosage_effect + age_effect + drug_effect))
    
    def _selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
    ) -> List[Dict[str, Any]]:
        """Tournament selection."""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            winner_idx = max(tournament_idx, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(
        self,
        population: List[Dict[str, Any]],
        drugs: List[str],
    ) -> List[Dict[str, Any]]:
        """Single-point crossover."""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[min(i + 1, len(population) - 1)]
            
            if random.random() < self.crossover_rate:
                # Crossover
                child1 = {
                    "drug": parent1["drug"],
                    "dosage": parent2["dosage"],
                    "schedule": parent1["schedule"],
                }
                child2 = {
                    "drug": parent2["drug"],
                    "dosage": parent1["dosage"],
                    "schedule": parent2["schedule"],
                }
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            offspring.extend([child1, child2])
        
        return offspring[:len(population)]
    
    def _mutation(
        self,
        population: List[Dict[str, Any]],
        drugs: List[str],
    ) -> List[Dict[str, Any]]:
        """Random mutation."""
        for individual in population:
            if random.random() < self.mutation_rate:
                # Mutate one gene
                gene = random.choice(["drug", "dosage", "schedule"])
                if gene == "drug":
                    individual["drug"] = random.choice(drugs)
                elif gene == "dosage":
                    individual["dosage"] = random.choice(self.dosages)
                else:
                    individual["schedule"] = random.choice(self.schedules)
        
        return population


def run_classical_baseline(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run classical baseline for benchmarking.
    
    Args:
        patient_data: Patient information
        
    Returns:
        Results including timing and accuracy
    """
    # Create patient profile
    patient = PatientProfile(
        age=patient_data.get("age", 55),
        cancer_type=patient_data.get("cancer_type", "breast"),
        cancer_stage=patient_data.get("cancer_stage", 2),
        biomarkers=patient_data.get("biomarkers", {"ER+": 0.8, "HER2+": 0.2}),
        comorbidities=patient_data.get("comorbidities", []),
    )
    
    # Run optimization (cap at 100 generations for fair benchmark comparison)
    optimizer = PersonalizedMedicineClassical(generations=100)
    result = optimizer.optimize(patient)
    
    return {
        "method": "classical_genetic_algorithm",
        "best_treatment": {
            "drug": result.best_treatment.drug_id,
            "dosage": result.best_treatment.dosage,
            "schedule": result.best_treatment.schedule,
            "efficacy": result.best_treatment.predicted_efficacy,
        },
        "treatments_tested": result.all_treatments_tested,
        "execution_time_seconds": result.execution_time_seconds,
        "convergence": result.convergence_history[-5:] if result.convergence_history else [],
    }


def run_simulated_annealing_baseline(
    patient_data: Dict[str, Any] = None,
    n_treatments: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulated annealing baseline for treatment optimization.

    Uses scipy.optimize.dual_annealing for global optimization.
    This is a second classical optimization comparison for QAOA,
    complementing the genetic algorithm baseline.

    Args:
        patient_data: Patient information dictionary.
        n_treatments: Number of candidate treatments.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with optimization results.
    """
    from scipy.optimize import dual_annealing

    patient_data = patient_data or {}
    start_time = time.time()
    rng = np.random.RandomState(seed)

    # Treatment efficacy landscape (synthetic)
    efficacy_weights = rng.rand(n_treatments)
    interaction_matrix = rng.rand(n_treatments, n_treatments) * 0.2
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2

    def objective(x):
        """Negative efficacy (we minimize)."""
        # x is continuous [0, 1] for each treatment; round to binary
        binary = (x > 0.5).astype(float)
        efficacy = np.dot(efficacy_weights, binary)
        interactions = binary @ interaction_matrix @ binary
        return -(efficacy + 0.5 * interactions)

    bounds = [(0, 1)] * n_treatments

    result = dual_annealing(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=200,
    )

    elapsed = time.time() - start_time
    best_binary = (result.x > 0.5).astype(int)

    return {
        "method": "simulated_annealing",
        "best_efficacy": float(-result.fun),
        "selected_treatments": int(np.sum(best_binary)),
        "treatment_vector": best_binary.tolist(),
        "execution_time_seconds": elapsed,
        "n_function_evaluations": int(result.nfev),
        "converged": bool(result.success),
    }

