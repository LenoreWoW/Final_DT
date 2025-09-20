"""
GraphQL schema and resolvers for Quantum Trail API.
"""

import graphene
from graphene import ObjectType, String, Int, Float, Boolean, List, Field, Mutation, InputObjectType, DateTime, JSONString
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from flask_graphql import GraphQLView
from datetime import datetime as dt
import json
import asyncio
from typing import Dict, Any, Optional

# Import models
from dt_project.models import (
    SimulationRun as SimulationRunModel,
    QuantumState as QuantumStateModel,
    AthleteProfile as AthleteProfileModel,
    PerformancePrediction as PerformancePredictionModel,
    MilitaryMission as MilitaryMissionModel
)

# Import quantum modules
from dt_project.quantum.quantum_digital_twin import QuantumDigitalTwin, QuantumTwinType
from dt_project.quantum.quantum_optimization import create_quantum_optimizer

# GraphQL Types
class SimulationRun(SQLAlchemyObjectType):
    """GraphQL type for simulation runs."""
    class Meta:
        model = SimulationRunModel
        interfaces = (graphene.relay.Node,)

class QuantumState(SQLAlchemyObjectType):
    """GraphQL type for quantum states."""
    class Meta:
        model = QuantumStateModel
        interfaces = (graphene.relay.Node,)
    
    # Custom fields
    entanglement_measure = graphene.Float()
    state_vector_json = graphene.String()
    
    def resolve_state_vector_json(self, info):
        """Convert state vector to JSON string."""
        if self.state_vector:
            return json.dumps(self.state_vector)
        return None

class AthleteProfile(SQLAlchemyObjectType):
    """GraphQL type for athlete profiles."""
    class Meta:
        model = AthleteProfileModel
        interfaces = (graphene.relay.Node,)
    
    # Custom computed fields
    overall_fitness = graphene.Float()
    injury_risk_category = graphene.String()
    
    def resolve_overall_fitness(self, info):
        """Calculate overall fitness score."""
        if self.fitness_level and self.endurance_level and self.strength_level:
            return (self.fitness_level + self.endurance_level + self.strength_level) / 3
        return 0.0
    
    def resolve_injury_risk_category(self, info):
        """Categorize injury risk."""
        if self.injury_risk is None:
            return "Unknown"
        elif self.injury_risk < 0.3:
            return "Low"
        elif self.injury_risk < 0.7:
            return "Medium"
        else:
            return "High"

class PerformancePrediction(SQLAlchemyObjectType):
    """GraphQL type for performance predictions."""
    class Meta:
        model = PerformancePredictionModel
        interfaces = (graphene.relay.Node,)

class MilitaryMission(SQLAlchemyObjectType):
    """GraphQL type for military missions."""
    class Meta:
        model = MilitaryMissionModel
        interfaces = (graphene.relay.Node,)

# Custom GraphQL Types
class QuantumCircuit(ObjectType):
    """GraphQL type for quantum circuits."""
    n_qubits = Int()
    gates = List(JSONString)
    depth = Int()
    circuit_type = String()

class QuantumMeasurementResult(ObjectType):
    """GraphQL type for quantum measurement results."""
    outcome = String()
    probability = Float()
    counts = JSONString()
    expectation_value = Float()
    variance = Float()

class OptimizationResult(ObjectType):
    """GraphQL type for optimization results."""
    optimal_value = Float()
    optimal_parameters = List(Float)
    convergence_history = List(Float)
    quantum_advantage = Float()
    execution_time = Float()
    metadata = JSONString()

class SystemStatus(ObjectType):
    """GraphQL type for system status."""
    status = String()
    quantum_backends_available = List(String)
    active_simulations = Int()
    total_simulations = Int()
    uptime_seconds = Float()
    version = String()

# Input Types
class SimulationInput(InputObjectType):
    """Input type for creating simulations."""
    simulation_type = String(required=True)
    parameters = JSONString(required=True)
    use_quantum = Boolean(default_value=True)
    quantum_backend = String(default_value="simulator")

class AthleteInput(InputObjectType):
    """Input type for athlete data."""
    name = String(required=True)
    age = Int(required=True)
    sport = String(required=True)
    fitness_level = Float()
    endurance_level = Float()
    strength_level = Float()
    training_hours = Float()

class QuantumCircuitInput(InputObjectType):
    """Input type for quantum circuits."""
    n_qubits = Int(required=True)
    gates = JSONString(required=True)
    shots = Int(default_value=1024)
    backend = String(default_value="simulator")

class OptimizationInput(InputObjectType):
    """Input type for optimization problems."""
    problem_type = String(required=True)
    parameters = JSONString(required=True)
    algorithm = String(default_value="vqe")
    n_qubits = Int(default_value=4)
    max_iterations = Int(default_value=100)

# Mutations
class CreateSimulation(Mutation):
    """Mutation to create a new simulation."""
    class Arguments:
        input = SimulationInput(required=True)
    
    simulation = Field(SimulationRun)
    success = Boolean()
    message = String()
    
    def mutate(self, info, input):
        try:
            # Create simulation in database
            simulation = SimulationRunModel(
                simulation_id=f"sim_{dt.utcnow().timestamp()}",
                simulation_type=input.simulation_type,
                status="pending",
                input_params=json.loads(input.parameters)
            )
            
            # Add to database session
            db_session = info.context.get('db_session')
            if db_session:
                db_session.add(simulation)
                db_session.commit()
            
            return CreateSimulation(
                simulation=simulation,
                success=True,
                message="Simulation created successfully"
            )
        except Exception as e:
            return CreateSimulation(
                simulation=None,
                success=False,
                message=str(e)
            )

class CreateAthleteProfile(Mutation):
    """Mutation to create athlete profile."""
    class Arguments:
        input = AthleteInput(required=True)
    
    athlete = Field(AthleteProfile)
    success = Boolean()
    message = String()
    
    def mutate(self, info, input):
        try:
            athlete = AthleteProfileModel(
                athlete_id=f"athlete_{dt.utcnow().timestamp()}",
                name=input.name,
                age=input.age,
                sport=input.sport,
                fitness_level=input.fitness_level,
                endurance_level=input.endurance_level,
                strength_level=input.strength_level,
                training_hours=input.training_hours
            )
            
            db_session = info.context.get('db_session')
            if db_session:
                db_session.add(athlete)
                db_session.commit()
            
            return CreateAthleteProfile(
                athlete=athlete,
                success=True,
                message="Athlete profile created successfully"
            )
        except Exception as e:
            return CreateAthleteProfile(
                athlete=None,
                success=False,
                message=str(e)
            )

class ExecuteQuantumCircuit(Mutation):
    """Mutation to execute quantum circuit."""
    class Arguments:
        input = QuantumCircuitInput(required=True)
    
    result = Field(QuantumMeasurementResult)
    success = Boolean()
    message = String()
    execution_time = Float()
    
    async def mutate(self, info, input):
        try:
            # Parse circuit data
            gates = json.loads(input.gates)
            circuit_data = {
                'n_qubits': input.n_qubits,
                'gates': gates
            }
            
            # Get quantum processor from context
            processor = info.context.get('quantum_processor')
            if not processor:
                raise Exception("Quantum processor not available")
            
            # Submit job
            job_result = await processor.submit_job(
                circuit_data=circuit_data,
                backend_preference=input.backend,
                shots=input.shots
            )
            
            # Wait for result
            result = await processor.get_result(job_result['job_id'])
            
            # Convert to GraphQL result
            measurement_result = QuantumMeasurementResult(
                counts=json.dumps(result.counts),
                execution_time=result.execution_time
            )
            
            return ExecuteQuantumCircuit(
                result=measurement_result,
                success=True,
                message="Circuit executed successfully",
                execution_time=result.execution_time
            )
            
        except Exception as e:
            return ExecuteQuantumCircuit(
                result=None,
                success=False,
                message=str(e),
                execution_time=0
            )

class RunOptimization(Mutation):
    """Mutation to run quantum optimization."""
    class Arguments:
        input = OptimizationInput(required=True)
    
    result = Field(OptimizationResult)
    success = Boolean()
    message = String()
    
    async def mutate(self, info, input):
        try:
            # Create optimizer
            optimizer = create_quantum_optimizer(
                algorithm=input.algorithm,
                n_qubits=input.n_qubits
            )
            
            # Parse parameters
            params = json.loads(input.parameters)
            
            # Run optimization based on problem type
            if input.problem_type == "maxcut":
                edges = params.get('edges', [])
                result = await optimizer.optimize_maxcut(edges)
            elif input.problem_type == "portfolio":
                assets = params.get('assets', [])
                risk_tolerance = params.get('risk_tolerance', 0.5)
                result = await optimizer.optimize_portfolio(assets, risk_tolerance)
            else:
                raise Exception(f"Unknown problem type: {input.problem_type}")
            
            # Convert to GraphQL result
            opt_result = OptimizationResult(
                optimal_value=result.optimal_value,
                optimal_parameters=result.optimal_parameters.tolist(),
                convergence_history=result.convergence_history,
                quantum_advantage=result.quantum_advantage,
                execution_time=result.execution_time,
                metadata=json.dumps(result.metadata)
            )
            
            return RunOptimization(
                result=opt_result,
                success=True,
                message="Optimization completed successfully"
            )
            
        except Exception as e:
            return RunOptimization(
                result=None,
                success=False,
                message=str(e)
            )

# Queries
class Query(ObjectType):
    """Root query type."""
    
    # Node queries for Relay
    node = graphene.relay.Node.Field()
    
    # List queries
    all_simulations = SQLAlchemyConnectionField(SimulationRun)
    all_quantum_states = SQLAlchemyConnectionField(QuantumState)
    all_athletes = SQLAlchemyConnectionField(AthleteProfile)
    all_predictions = SQLAlchemyConnectionField(PerformancePrediction)
    all_missions = SQLAlchemyConnectionField(MilitaryMission)
    
    # Single item queries
    simulation = Field(SimulationRun, simulation_id=String(required=True))
    athlete = Field(AthleteProfile, athlete_id=String(required=True))
    quantum_state = Field(QuantumState, entity_id=String(required=True))
    
    # System queries
    system_status = Field(SystemStatus)
    available_backends = List(String)
    
    def resolve_simulation(self, info, simulation_id):
        """Get simulation by ID."""
        db_session = info.context.get('db_session')
        if db_session:
            return db_session.query(SimulationRunModel).filter_by(
                simulation_id=simulation_id
            ).first()
        return None
    
    def resolve_athlete(self, info, athlete_id):
        """Get athlete by ID."""
        db_session = info.context.get('db_session')
        if db_session:
            return db_session.query(AthleteProfileModel).filter_by(
                athlete_id=athlete_id
            ).first()
        return None
    
    def resolve_quantum_state(self, info, entity_id):
        """Get quantum state by entity ID."""
        db_session = info.context.get('db_session')
        if db_session:
            return db_session.query(QuantumStateModel).filter_by(
                entity_id=entity_id
            ).order_by(QuantumStateModel.created_at.desc()).first()
        return None
    
    def resolve_system_status(self, info):
        """Get system status."""
        return SystemStatus(
            status="operational",
            quantum_backends_available=["simulator", "ibm_quantum"],
            active_simulations=5,
            total_simulations=100,
            uptime_seconds=3600.0,
            version="1.0.0"
        )
    
    def resolve_available_backends(self, info):
        """Get available quantum backends."""
        processor = info.context.get('quantum_processor')
        if processor:
            return list(processor.backends.keys())
        return ["simulator"]

# Root Mutation
class Mutations(ObjectType):
    """Root mutation type."""
    create_simulation = CreateSimulation.Field()
    create_athlete_profile = CreateAthleteProfile.Field()
    execute_quantum_circuit = ExecuteQuantumCircuit.Field()
    run_optimization = RunOptimization.Field()

# Create schema
schema = graphene.Schema(query=Query, mutation=Mutations)

def create_graphql_view(app, db_session, quantum_processor=None):
    """Create GraphQL view for Flask app."""
    
    # Context function to pass dependencies
    def get_context():
        return {
            'db_session': db_session,
            'quantum_processor': quantum_processor,
            'app': app
        }
    
    # Add GraphQL endpoint
    app.add_url_rule(
        '/graphql',
        view_func=GraphQLView.as_view(
            'graphql',
            schema=schema,
            graphiql=app.config.get('DEBUG', False),  # GraphiQL interface in debug mode
            get_context=get_context
        )
    )
    
    return schema