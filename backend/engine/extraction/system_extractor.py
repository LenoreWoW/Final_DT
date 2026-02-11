"""
System Extraction Module

Extracts system components from natural language descriptions:
- Entities (people, objects, resources)
- States (properties, attributes)
- Relationships (interactions, dependencies)
- Rules (physics, logic, dynamics)
- Constraints (limits, boundaries)
- Goals (what user wants to know/optimize)

This is the first step in the Universal Twin Generation pipeline.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.models.schemas import (
    Entity,
    Relationship,
    Rule,
    Constraint,
    ExtractedSystem,
)


class DomainType(str, Enum):
    """Detected domain types."""
    HEALTHCARE = "healthcare"
    SPORTS = "sports"
    MILITARY = "military"
    ENVIRONMENT = "environment"
    FINANCE = "finance"
    LOGISTICS = "logistics"
    MANUFACTURING = "manufacturing"
    SOCIAL = "social"
    SCIENCE = "science"
    GENERAL = "general"


class GoalType(str, Enum):
    """Types of goals users want to achieve."""
    OPTIMIZE = "optimize"  # Find the best X
    PREDICT = "predict"  # What will happen
    UNDERSTAND = "understand"  # Why does X occur
    EXPLORE = "explore"  # Show possibilities
    COMPARE = "compare"  # A vs B
    DETECT = "detect"  # Find anomalies


@dataclass
class ExtractionResult:
    """Result of system extraction."""
    system: ExtractedSystem
    confidence: float
    missing_info: List[str]
    suggestions: List[str]
    domain_confidence: Dict[str, float] = field(default_factory=dict)


class SystemExtractor:
    """
    Extracts system components from natural language.
    
    Uses pattern matching and keyword analysis. In production,
    this would be enhanced with LLM-based extraction.
    """
    
    # Domain detection keywords
    DOMAIN_KEYWORDS: Dict[DomainType, List[str]] = {
        DomainType.HEALTHCARE: [
            "patient", "treatment", "diagnosis", "drug", "hospital", "cancer",
            "tumor", "disease", "symptom", "therapy", "medicine", "doctor",
            "nurse", "surgery", "clinical", "health", "medical", "oncology",
            "genomic", "epidemic", "virus", "infection", "vaccine"
        ],
        DomainType.SPORTS: [
            "athlete", "marathon", "race", "training", "performance", "stamina",
            "pace", "runner", "player", "team", "game", "match", "score",
            "fitness", "workout", "exercise", "competition", "championship",
            "olympics", "sports", "football", "basketball", "soccer"
        ],
        DomainType.MILITARY: [
            "battalion", "troops", "defense", "operation", "combat",
            "soldier", "army", "navy", "mission", "enemy", "strategic",
            "tactical", "weapon", "ammunition", "headquarters", "base",
            "reconnaissance", "intelligence", "deployment"
        ],
        DomainType.ENVIRONMENT: [
            "wildfire", "ecosystem", "species", "population", "climate",
            "wildlife", "forest", "ocean", "temperature", "carbon",
            "pollution", "conservation", "habitat", "biodiversity",
            "extinction", "sustainable", "renewable", "weather", 
            "acres", "spread", "tanker", "firefighter"
        ],
        DomainType.FINANCE: [
            "stock", "portfolio", "investment", "market", "trading", "asset",
            "risk", "return", "dividend", "bond", "equity", "hedge",
            "derivative", "currency", "cryptocurrency", "bitcoin", "bank"
        ],
        DomainType.LOGISTICS: [
            "supply", "shipping", "warehouse", "delivery", "route", "fleet",
            "inventory", "distribution", "transport", "cargo", "freight",
            "logistics", "chain", "procurement", "vendor"
        ],
        DomainType.MANUFACTURING: [
            "factory", "production", "assembly", "quality", "defect",
            "machine", "equipment", "maintenance", "output", "efficiency",
            "automation", "robotics", "lean", "throughput"
        ],
        DomainType.SOCIAL: [
            "user", "customer", "behavior", "network", "community",
            "engagement", "viral", "influence", "trend", "sentiment",
            "opinion", "election", "vote", "poll"
        ],
        DomainType.SCIENCE: [
            "molecule", "atom", "particle", "quantum", "experiment",
            "hypothesis", "reaction", "chemical", "physics", "biology",
            "research", "laboratory", "simulation", "model"
        ],
    }
    
    # Entity patterns per domain
    ENTITY_PATTERNS: Dict[DomainType, List[Tuple[str, str, List[str]]]] = {
        DomainType.HEALTHCARE: [
            (r"patient", "patient", ["age", "condition", "vitals", "history"]),
            (r"tumor|cancer", "tumor", ["size", "stage", "type", "location"]),
            (r"drug|medication", "drug", ["dosage", "frequency", "interactions"]),
            (r"hospital", "hospital", ["beds", "capacity", "departments"]),
            (r"treatment", "treatment", ["type", "duration", "efficacy"]),
        ],
        DomainType.SPORTS: [
            (r"athlete|runner|player", "athlete", ["stamina", "pace", "energy", "fitness"]),
            (r"marathon|race", "race", ["distance", "elevation", "temperature"]),
            (r"team", "team", ["players", "strategy", "formation"]),
            (r"game|match", "game", ["duration", "rules", "score"]),
        ],
        DomainType.MILITARY: [
            (r"battalion|brigade|division", "unit", ["strength", "morale", "position", "equipment"]),
            (r"troops|soldiers", "personnel", ["count", "training", "fatigue"]),
            (r"terrain|area|region", "terrain", ["type", "elevation", "cover"]),
            (r"enemy|adversary", "enemy", ["strength", "position", "capabilities"]),
        ],
        DomainType.ENVIRONMENT: [
            (r"fire|wildfire", "fire", ["intensity", "spread_rate", "area"]),
            (r"species|animal|plant", "species", ["population", "habitat", "status"]),
            (r"forest|ecosystem", "ecosystem", ["health", "biodiversity", "area"]),
            (r"water|river|ocean", "water_body", ["volume", "quality", "flow"]),
        ],
        DomainType.FINANCE: [
            (r"stock|share", "stock", ["price", "volume", "volatility"]),
            (r"portfolio", "portfolio", ["value", "allocation", "risk"]),
            (r"market", "market", ["trend", "sentiment", "volume"]),
        ],
        DomainType.MANUFACTURING: [
            (r"factory|plant", "factory", ["capacity", "output", "efficiency"]),
            (r"robot|machine", "machine", ["speed", "accuracy", "uptime"]),
            (r"assembly.line|production.line", "assembly_line", ["throughput", "cycle_time", "defect_rate"]),
            (r"product|item|part", "product", ["quantity", "quality", "cost"]),
        ],
        DomainType.LOGISTICS: [
            (r"warehouse", "warehouse", ["capacity", "location", "inventory"]),
            (r"fleet|truck|vehicle", "vehicle", ["capacity", "speed", "fuel"]),
            (r"route|path", "route", ["distance", "time", "cost"]),
            (r"shipment|delivery", "shipment", ["weight", "destination", "deadline"]),
        ],
        DomainType.SCIENCE: [
            (r"molecule|compound", "molecule", ["formula", "mass", "bonds"]),
            (r"particle|atom", "particle", ["mass", "charge", "spin"]),
            (r"experiment", "experiment", ["hypothesis", "variables", "observations"]),
            (r"reaction|process", "reaction", ["rate", "temperature", "yield"]),
        ],
        DomainType.SOCIAL: [
            (r"user|customer|person", "user", ["age", "preferences", "behavior"]),
            (r"network|community", "network", ["size", "density", "activity"]),
            (r"campaign|promotion", "campaign", ["budget", "reach", "conversion"]),
        ],
    }
    
    # Goal patterns
    GOAL_PATTERNS: List[Tuple[str, GoalType]] = [
        (r"(optimize|maximize|minimize|best\b|reduce\b|improve\b|increase\b)", GoalType.OPTIMIZE),
        (r"(predict|forecast|project|expect|what will happen|what happens)", GoalType.PREDICT),
        (r"(why|understand|explain|cause)", GoalType.UNDERSTAND),
        (r"(show|explore|possibilities|scenarios)", GoalType.EXPLORE),
        (r"(compare|versus|vs|difference|better)", GoalType.COMPARE),
        (r"(detect|find|identify|anomaly|outlier)", GoalType.DETECT),
    ]
    
    # Relationship patterns
    RELATIONSHIP_PATTERNS: List[Tuple[str, str]] = [
        (r"(\w+) affects (\w+)", "affects"),
        (r"(\w+) depends on (\w+)", "depends_on"),
        (r"(\w+) competes with (\w+)", "competes_with"),
        (r"(\w+) supplies (\w+)", "supplies"),
        (r"(\w+) interacts with (\w+)", "interacts_with"),
        (r"(\w+) influences (\w+)", "influences"),
    ]
    
    # Constraint patterns
    CONSTRAINT_PATTERNS: List[Tuple[str, str]] = [
        (r"budget (?:is |of )?\$?([\d,]+)", "budget"),
        (r"must (?:finish |complete )?(?:in |within )?([\d]+)\s*(hours?|days?|weeks?|months?)", "time"),
        (r"(no more than|at most|maximum|max) ([\d]+)", "maximum"),
        (r"(at least|minimum|min) ([\d]+)", "minimum"),
        (r"cannot exceed ([\d]+)", "limit"),
        (r"(?:under|below|less than) ([\d]+)", "maximum"),
        (r"(?:over|above|more than|greater than) ([\d]+)", "minimum"),
    ]
    
    def extract(self, message: str, existing_system: Optional[ExtractedSystem] = None) -> ExtractionResult:
        """
        Extract system components from a natural language message.
        
        Args:
            message: The user's description
            existing_system: Previously extracted system to build upon
            
        Returns:
            ExtractionResult with extracted system and metadata
        """
        message_lower = message.lower()
        
        # Start with existing system or create new
        if existing_system:
            system = ExtractedSystem(**existing_system.model_dump())
        else:
            system = ExtractedSystem()
        
        # Detect domain
        domain, domain_confidence = self._detect_domain(message_lower)
        if not system.domain:
            system.domain = domain.value
        
        # Extract entities
        new_entities = self._extract_entities(message, message_lower, domain)
        for entity in new_entities:
            if not any(e.id == entity.id for e in system.entities):
                system.entities.append(entity)
        
        # Extract relationships
        new_relationships = self._extract_relationships(message_lower, system.entities)
        for rel in new_relationships:
            if not any(r.source_id == rel.source_id and r.target_id == rel.target_id 
                       for r in system.relationships):
                system.relationships.append(rel)
        
        # Extract rules
        new_rules = self._extract_rules(message, domain)
        for rule in new_rules:
            if not any(r.id == rule.id for r in system.rules):
                system.rules.append(rule)
        
        # Extract constraints
        new_constraints = self._extract_constraints(message_lower)
        for constraint in new_constraints:
            if not any(c.id == constraint.id for c in system.constraints):
                system.constraints.append(constraint)
        
        # Extract goal
        if not system.goal:
            system.goal = self._extract_goal(message_lower)
        
        # Determine missing information and suggestions
        missing_info = self._get_missing_info(system)
        suggestions = self._generate_suggestions(system, domain)
        confidence = self._calculate_confidence(system)
        
        return ExtractionResult(
            system=system,
            confidence=confidence,
            missing_info=missing_info,
            suggestions=suggestions,
            domain_confidence=domain_confidence,
        )
    
    def _detect_domain(self, message: str) -> Tuple[DomainType, Dict[str, float]]:
        """Detect the domain from message content."""
        scores: Dict[str, float] = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in message)
            scores[domain.value] = count / len(keywords) if keywords else 0
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        # Find best match
        best_domain = max(scores.items(), key=lambda x: x[1])
        
        if best_domain[1] > 0.1:
            return DomainType(best_domain[0]), scores
        return DomainType.GENERAL, scores
    
    def _extract_entities(
        self, 
        message: str, 
        message_lower: str,
        domain: DomainType
    ) -> List[Entity]:
        """Extract entities from message."""
        entities = []
        
        # Get patterns for detected domain
        patterns = self.ENTITY_PATTERNS.get(domain, [])
        
        for pattern, entity_type, properties in patterns:
            if re.search(pattern, message_lower):
                # Extract specific instance if mentioned
                entity_id = f"{entity_type}_{len(entities)}"
                entities.append(Entity(
                    id=entity_id,
                    name=entity_type.replace("_", " ").title(),
                    type=entity_type,
                    properties={prop: None for prop in properties},
                ))
        
        # Fallback: if no domain-specific entities found, extract nouns after "with" or "of"
        if not entities:
            # Pattern: "N <nouns>" e.g. "10 robots", "5 assembly lines"
            noun_matches = re.findall(r'(\d+)\s+([a-z][a-z\s]{2,20}?)(?:\s+and|\s*,|\s*\.|\s+to\b|\s+with\b|\s+for\b|$)', message_lower)
            for count, noun in noun_matches:
                noun = noun.strip()
                if noun and len(noun) > 2:
                    eid = f"{noun.replace(' ', '_')}_{len(entities)}"
                    entities.append(Entity(
                        id=eid,
                        name=noun.title(),
                        type=noun.replace(" ", "_"),
                        properties={"count": float(count)},
                    ))
            # Also look for "a/an/the <noun>" patterns
            if not entities:
                simple_nouns = re.findall(r'(?:a|an|the)\s+([a-z][a-z\s]{2,20}?)(?:\s+with|\s+that|\s+for|\s*,|\s*\.)', message_lower)
                for noun in simple_nouns[:3]:
                    noun = noun.strip()
                    if noun:
                        eid = f"{noun.replace(' ', '_')}_{len(entities)}"
                        entities.append(Entity(
                            id=eid,
                            name=noun.title(),
                            type=noun.replace(" ", "_"),
                            properties={},
                        ))

        # Extract numbers and associate with entities
        numbers = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(\w+)', message)
        for value, unit in numbers:
            for entity in entities:
                if unit.lower() in entity.properties or any(
                    unit.lower() in prop.lower() for prop in entity.properties
                ):
                    # Found a matching property
                    for prop in entity.properties:
                        if unit.lower() in prop.lower():
                            entity.properties[prop] = float(value.replace(",", ""))
        
        return entities
    
    def _extract_relationships(
        self,
        message: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        # Use pattern matching for explicit relationships
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            matches = re.findall(pattern, message)
            for match in matches:
                if len(match) >= 2:
                    source_name, target_name = match[0], match[1]
                    # Find matching entities
                    source = next((e for e in entities if source_name in e.name.lower()), None)
                    target = next((e for e in entities if target_name in e.name.lower()), None)
                    if source and target:
                        relationships.append(Relationship(
                            source_id=source.id,
                            target_id=target.id,
                            type=rel_type,
                            strength=1.0,
                        ))
        
        # Add implicit relationships based on domain knowledge
        if len(entities) >= 2:
            # Add basic interaction relationship between first two entities
            relationships.append(Relationship(
                source_id=entities[0].id,
                target_id=entities[1].id,
                type="interacts_with",
                strength=0.5,  # Lower confidence for inferred
            ))
        
        return relationships
    
    def _extract_rules(self, message: str, domain: DomainType) -> List[Rule]:
        """Extract rules governing the system."""
        rules = []
        
        # Domain-specific default rules
        domain_rules = {
            DomainType.SPORTS: [
                Rule(id="fatigue", name="Fatigue Accumulation", 
                     description="Fatigue increases with exertion",
                     formula="fatigue += exertion * time", type="physiology"),
            ],
            DomainType.HEALTHCARE: [
                Rule(id="treatment_response", name="Treatment Response",
                     description="Treatment efficacy depends on patient factors",
                     type="biology"),
            ],
            DomainType.MILITARY: [
                Rule(id="combat_power", name="Combat Power",
                     description="Combat effectiveness depends on morale and strength",
                     type="military"),
            ],
            DomainType.ENVIRONMENT: [
                Rule(id="spread_dynamics", name="Spread Dynamics",
                     description="Fire/disease spreads based on conditions",
                     type="physics"),
            ],
        }
        
        # Add domain default rules
        if domain in domain_rules:
            rules.extend(domain_rules[domain])
        
        # Extract explicit rules from message
        rule_patterns = [
            r"when (.+) then (.+)",
            r"if (.+) then (.+)",
            r"(.+) causes (.+)",
            r"(.+) leads to (.+)",
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, message.lower())
            for i, match in enumerate(matches):
                if len(match) >= 2:
                    rules.append(Rule(
                        id=f"custom_rule_{i}",
                        name=f"Custom Rule {i+1}",
                        description=f"When {match[0]}, then {match[1]}",
                        type="custom",
                    ))
        
        return rules
    
    def _extract_constraints(self, message: str) -> List[Constraint]:
        """Extract constraints from the message."""
        constraints = []
        
        for pattern, constraint_type in self.CONSTRAINT_PATTERNS:
            matches = re.findall(pattern, message)
            for match in matches:
                if isinstance(match, tuple):
                    value = match[-1] if len(match) > 1 else match[0]
                else:
                    value = match
                    
                constraints.append(Constraint(
                    id=f"constraint_{constraint_type}_{len(constraints)}",
                    name=f"{constraint_type.title()} Constraint",
                    type=constraint_type,
                    value=value,
                ))
        
        return constraints
    
    def _extract_goal(self, message: str) -> Optional[str]:
        """Extract the user's goal from the message."""
        for pattern, goal_type in self.GOAL_PATTERNS:
            if re.search(pattern, message):
                return goal_type.value
        return None
    
    def _get_missing_info(self, system: ExtractedSystem) -> List[str]:
        """Determine what information is still needed.

        Only entities and goal are required to proceed.
        Constraints and entity_properties are optional enhancements.
        """
        missing = []

        if not system.entities:
            missing.append("entities")
        if not system.goal:
            missing.append("goal")

        return missing
    
    def _generate_suggestions(
        self,
        system: ExtractedSystem,
        domain: DomainType
    ) -> List[str]:
        """Generate suggestions based on extracted system."""
        suggestions = []
        
        if not system.entities:
            suggestions.append("Describe the main components of your system")
        
        if not system.goal:
            suggestions.append("Tell me what you want to optimize, predict, or understand")
        
        # Domain-specific suggestions
        domain_suggestions = {
            DomainType.HEALTHCARE: [
                "What treatment options are you considering?",
                "What patient factors are important?",
            ],
            DomainType.SPORTS: [
                "What metrics matter for performance?",
                "What constraints do you have (time, budget)?",
            ],
            DomainType.MILITARY: [
                "What is the operational objective?",
                "What resources are available?",
            ],
        }
        
        if domain in domain_suggestions and len(system.entities) > 0:
            suggestions.extend(domain_suggestions[domain][:1])
        
        return suggestions
    
    def _calculate_confidence(self, system: ExtractedSystem) -> float:
        """Calculate confidence score for the extraction."""
        score = 0.0
        
        # Entities (max 0.3)
        if system.entities:
            score += min(len(system.entities) * 0.1, 0.3)
        
        # Goal (0.2)
        if system.goal:
            score += 0.2
        
        # Relationships (max 0.2)
        if system.relationships:
            score += min(len(system.relationships) * 0.1, 0.2)
        
        # Rules (max 0.15)
        if system.rules:
            score += min(len(system.rules) * 0.05, 0.15)
        
        # Constraints (max 0.15)
        if system.constraints:
            score += min(len(system.constraints) * 0.05, 0.15)
        
        return min(score, 1.0)


# Singleton instance
extractor = SystemExtractor()


def extract_system(message: str, existing: Optional[ExtractedSystem] = None) -> ExtractionResult:
    """Convenience function to extract system from message."""
    return extractor.extract(message, existing)

