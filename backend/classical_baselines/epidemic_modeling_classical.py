"""
Epidemic Modeling Classical Baseline

Agent-based SIR epidemic model with spatial dynamics and interventions.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AgentState(Enum):
    """Agent health state"""
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DEAD = 3


@dataclass
class Agent:
    """Individual agent in the simulation"""
    id: int
    state: AgentState
    position: np.ndarray  # (x, y)
    age: int
    infection_day: int = -1
    recovery_day: int = -1


@dataclass
class Intervention:
    """Public health intervention"""
    name: str
    start_day: int
    effectiveness: float  # 0.0 to 1.0
    compliance_rate: float  # 0.0 to 1.0


class EpidemicModelingClassical:
    """Classical agent-based epidemic simulation"""

    def __init__(self,
                 population_size: int = 10000,
                 initial_infected: int = 10,
                 grid_size: Tuple[int, int] = (100, 100),
                 transmission_rate: float = 0.3,
                 recovery_days: int = 14,
                 mortality_rate: float = 0.02,
                 contact_radius: float = 5.0):
        """
        Initialize epidemic model

        Args:
            population_size: Number of agents
            initial_infected: Initial infected count
            grid_size: Spatial grid dimensions
            transmission_rate: Probability of transmission per contact
            recovery_days: Days to recovery
            mortality_rate: Probability of death
            contact_radius: Distance for potential contact
        """
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.grid_size = grid_size
        self.transmission_rate = transmission_rate
        self.recovery_days = recovery_days
        self.mortality_rate = mortality_rate
        self.contact_radius = contact_radius

        self.agents: List[Agent] = []
        self.current_day = 0

        # Statistics tracking
        self.history = {
            'susceptible': [],
            'infected': [],
            'recovered': [],
            'dead': []
        }

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population of agents"""
        self.agents = []

        for i in range(self.population_size):
            # Random position in grid
            position = np.array([
                np.random.uniform(0, self.grid_size[0]),
                np.random.uniform(0, self.grid_size[1])
            ])

            # Age distribution (simplified)
            age = int(np.random.exponential(35)) + 18
            age = min(age, 90)

            # Initial state
            if i < self.initial_infected:
                state = AgentState.INFECTED
                infection_day = 0
            else:
                state = AgentState.SUSCEPTIBLE
                infection_day = -1

            agent = Agent(
                id=i,
                state=state,
                position=position,
                age=age,
                infection_day=infection_day
            )

            self.agents.append(agent)

    def _get_nearby_agents(self, agent: Agent) -> List[Agent]:
        """Find agents within contact radius"""
        nearby = []

        for other in self.agents:
            if other.id == agent.id:
                continue

            distance = np.linalg.norm(agent.position - other.position)

            if distance <= self.contact_radius:
                nearby.append(other)

        return nearby

    def _attempt_transmission(self,
                            infected: Agent,
                            susceptible: Agent,
                            intervention_factor: float = 1.0) -> bool:
        """
        Attempt disease transmission

        Args:
            infected: Infected agent
            susceptible: Susceptible agent
            intervention_factor: Multiplier for intervention effectiveness

        Returns:
            True if transmission occurred
        """
        # Effective transmission rate with interventions
        effective_rate = self.transmission_rate * intervention_factor

        # Age-based susceptibility (older = more susceptible)
        age_factor = 1.0 + (susceptible.age - 40) / 100
        age_factor = max(0.5, min(2.0, age_factor))

        # Transmission probability
        transmission_prob = effective_rate * age_factor

        return np.random.random() < transmission_prob

    def _move_agents(self):
        """Move agents randomly (simplified mobility)"""
        for agent in self.agents:
            if agent.state != AgentState.DEAD:
                # Random walk
                direction = np.random.randn(2)
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                # Movement distance (infected move less)
                if agent.state == AgentState.INFECTED:
                    distance = np.random.uniform(0, 2)
                else:
                    distance = np.random.uniform(0, 5)

                # Update position
                new_position = agent.position + direction * distance

                # Boundary conditions (toroidal)
                new_position[0] = new_position[0] % self.grid_size[0]
                new_position[1] = new_position[1] % self.grid_size[1]

                agent.position = new_position

    def _update_infections(self, intervention_factor: float = 1.0):
        """Process new infections"""
        infected_agents = [a for a in self.agents if a.state == AgentState.INFECTED]
        new_infections = []

        for infected in infected_agents:
            # Find nearby susceptible agents
            nearby = self._get_nearby_agents(infected)
            susceptible_nearby = [
                a for a in nearby
                if a.state == AgentState.SUSCEPTIBLE
            ]

            # Attempt transmission
            for susceptible in susceptible_nearby:
                if self._attempt_transmission(infected, susceptible, intervention_factor):
                    new_infections.append(susceptible)

        # Apply new infections
        for agent in new_infections:
            agent.state = AgentState.INFECTED
            agent.infection_day = self.current_day

    def _update_recoveries(self):
        """Process recoveries and deaths"""
        infected_agents = [a for a in self.agents if a.state == AgentState.INFECTED]

        for agent in infected_agents:
            days_infected = self.current_day - agent.infection_day

            if days_infected >= self.recovery_days:
                # Determine outcome
                # Age-adjusted mortality
                age_mortality_factor = 1.0 + (agent.age - 40) / 50
                age_mortality_factor = max(0.5, min(5.0, age_mortality_factor))

                death_prob = self.mortality_rate * age_mortality_factor

                if np.random.random() < death_prob:
                    agent.state = AgentState.DEAD
                else:
                    agent.state = AgentState.RECOVERED
                    agent.recovery_day = self.current_day

    def _record_statistics(self):
        """Record current state statistics"""
        counts = {
            AgentState.SUSCEPTIBLE: 0,
            AgentState.INFECTED: 0,
            AgentState.RECOVERED: 0,
            AgentState.DEAD: 0
        }

        for agent in self.agents:
            counts[agent.state] += 1

        self.history['susceptible'].append(counts[AgentState.SUSCEPTIBLE])
        self.history['infected'].append(counts[AgentState.INFECTED])
        self.history['recovered'].append(counts[AgentState.RECOVERED])
        self.history['dead'].append(counts[AgentState.DEAD])

    def simulate(self,
                days: int = 100,
                interventions: Optional[List[Intervention]] = None) -> Dict:
        """
        Run epidemic simulation

        Args:
            days: Number of days to simulate
            interventions: List of interventions to apply

        Returns:
            Simulation results
        """
        start_time = time.time()

        if interventions is None:
            interventions = []

        self.current_day = 0
        self.history = {
            'susceptible': [],
            'infected': [],
            'recovered': [],
            'dead': []
        }

        # Run simulation
        for day in range(days):
            self.current_day = day

            # Calculate intervention factor
            intervention_factor = 1.0
            for intervention in interventions:
                if day >= intervention.start_day:
                    reduction = intervention.effectiveness * intervention.compliance_rate
                    intervention_factor *= (1.0 - reduction)

            # Simulation steps
            self._move_agents()
            self._update_infections(intervention_factor)
            self._update_recoveries()
            self._record_statistics()

            # Early stopping if epidemic ends
            if self.history['infected'][-1] == 0:
                break

        elapsed_time = time.time() - start_time

        # Calculate summary statistics
        peak_infections = max(self.history['infected'])
        peak_day = self.history['infected'].index(peak_infections)
        total_infected = (
            self.history['infected'][-1] +
            self.history['recovered'][-1] +
            self.history['dead'][-1]
        )
        attack_rate = total_infected / self.population_size

        return {
            'simulation_days': len(self.history['susceptible']),
            'simulation_time': elapsed_time,
            'population_size': self.population_size,
            'peak_infections': peak_infections,
            'peak_day': peak_day,
            'total_infected': total_infected,
            'total_recovered': self.history['recovered'][-1],
            'total_deaths': self.history['dead'][-1],
            'attack_rate': attack_rate * 100,
            'mortality_rate': (self.history['dead'][-1] / total_infected * 100) if total_infected > 0 else 0,
            'final_susceptible': self.history['susceptible'][-1],
            'history': self.history,
            'method': 'Agent-Based SIR Model'
        }

    def evaluate_intervention(self,
                            intervention: Intervention,
                            days: int = 100) -> Dict:
        """
        Evaluate the effect of an intervention

        Args:
            intervention: Intervention to test
            days: Simulation duration

        Returns:
            Comparison results
        """
        # Baseline (no intervention)
        self._initialize_population()
        baseline = self.simulate(days=days, interventions=[])

        # With intervention
        self._initialize_population()
        with_intervention = self.simulate(days=days, interventions=[intervention])

        # Calculate improvement
        lives_saved = baseline['total_deaths'] - with_intervention['total_deaths']
        infections_prevented = baseline['total_infected'] - with_intervention['total_infected']

        return {
            'intervention': intervention.name,
            'baseline': baseline,
            'with_intervention': with_intervention,
            'lives_saved': lives_saved,
            'infections_prevented': infections_prevented,
            'effectiveness_score': (lives_saved / baseline['total_deaths'] * 100) if baseline['total_deaths'] > 0 else 0
        }


def run_epidemic_modeling_classical(population_size: int = 10000,
                                   simulation_days: int = 100) -> Dict:
    """
    Run classical epidemic modeling benchmark

    Args:
        population_size: Size of population
        simulation_days: Days to simulate

    Returns:
        Benchmark results
    """
    model = EpidemicModelingClassical(
        population_size=population_size,
        initial_infected=10,
        transmission_rate=0.3,
        recovery_days=14,
        mortality_rate=0.02,
        contact_radius=5.0
    )

    # Test intervention: social distancing
    intervention = Intervention(
        name="Social Distancing",
        start_day=20,
        effectiveness=0.5,
        compliance_rate=0.7
    )

    results = model.evaluate_intervention(
        intervention=intervention,
        days=simulation_days
    )

    return results


if __name__ == '__main__':
    # Test the classical epidemic modeling
    print("Testing Classical Epidemic Modeling...")
    print("=" * 60)

    results = run_epidemic_modeling_classical(
        population_size=1000,
        simulation_days=100
    )

    print(f"\nPopulation Size: {results['baseline']['population_size']}")
    print(f"Simulation Time: {results['baseline']['simulation_time']:.2f} seconds")

    print("\n=== BASELINE (No Intervention) ===")
    baseline = results['baseline']
    print(f"Peak Infections: {baseline['peak_infections']} (Day {baseline['peak_day']})")
    print(f"Total Infected: {baseline['total_infected']}")
    print(f"Total Deaths: {baseline['total_deaths']}")
    print(f"Attack Rate: {baseline['attack_rate']:.1f}%")
    print(f"Mortality Rate: {baseline['mortality_rate']:.1f}%")

    print(f"\n=== WITH INTERVENTION ({results['intervention']}) ===")
    intervention = results['with_intervention']
    print(f"Peak Infections: {intervention['peak_infections']} (Day {intervention['peak_day']})")
    print(f"Total Infected: {intervention['total_infected']}")
    print(f"Total Deaths: {intervention['total_deaths']}")
    print(f"Attack Rate: {intervention['attack_rate']:.1f}%")
    print(f"Mortality Rate: {intervention['mortality_rate']:.1f}%")

    print("\n=== INTERVENTION IMPACT ===")
    print(f"Lives Saved: {results['lives_saved']}")
    print(f"Infections Prevented: {results['infections_prevented']}")
    print(f"Effectiveness Score: {results['effectiveness_score']:.1f}%")
