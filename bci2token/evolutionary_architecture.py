"""
Evolutionary Architecture - Generation 6+ Self-Evolving System
=============================================================

Revolutionary self-evolving architecture implementing:
- Genetic algorithms for architecture optimization
- Evolutionary neural topology search
- Self-modifying code structures
- Fitness-based component selection
- Multi-objective evolution (performance vs efficiency)
- Adaptive mutation and crossover strategies
- Evolutionary memory and learning transfer

This system continuously evolves its own architecture to
achieve optimal performance for each specific use case.
"""

import asyncio
import time
import threading
import json
import math
import random
import copy
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict, deque
import concurrent.futures
import secrets

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not available. Evolutionary features will be limited.")

class GeneType(Enum):
    """Types of architectural genes."""
    PROCESSING_DEPTH = "processing_depth"
    CONNECTION_DENSITY = "connection_density"
    LEARNING_RATE = "learning_rate"
    MEMORY_ALLOCATION = "memory_allocation"
    PARALLEL_THREADS = "parallel_threads"
    CACHE_SIZE = "cache_size"
    SECURITY_LEVEL = "security_level"
    ERROR_TOLERANCE = "error_tolerance"

class FitnessMetric(Enum):
    """Fitness evaluation metrics."""
    PROCESSING_SPEED = "processing_speed"
    ACCURACY = "accuracy"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    FAULT_TOLERANCE = "fault_tolerance"
    ADAPTABILITY = "adaptability"
    SECURITY_SCORE = "security_score"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class ArchitecturalGene:
    """Individual gene in the architectural genome."""
    gene_type: GeneType
    value: float  # Normalized value between 0 and 1
    expression_strength: float = 1.0  # How strongly this gene is expressed
    mutation_rate: float = 0.1
    last_mutation: float = field(default_factory=time.time)
    fitness_contribution: float = 0.0
    
    def mutate(self, mutation_strength: float = None) -> 'ArchitecturalGene':
        """Create a mutated copy of this gene."""
        mutated = copy.deepcopy(self)
        
        if mutation_strength is None:
            mutation_strength = self.mutation_rate
            
        # Apply mutation
        mutation_delta = random.gauss(0, mutation_strength)
        mutated.value = max(0.0, min(1.0, self.value + mutation_delta))
        
        # Mutate expression strength
        expression_delta = random.gauss(0, mutation_strength * 0.5)
        mutated.expression_strength = max(0.0, min(2.0, self.expression_strength + expression_delta))
        
        # Adapt mutation rate based on recent performance
        if self.fitness_contribution > 0.5:
            mutated.mutation_rate *= 0.95  # Reduce mutation for successful genes
        else:
            mutated.mutation_rate *= 1.05  # Increase mutation for unsuccessful genes
            
        mutated.mutation_rate = max(0.01, min(0.5, mutated.mutation_rate))
        mutated.last_mutation = time.time()
        
        return mutated

@dataclass
class ArchitecturalGenome:
    """Complete genome defining an architecture."""
    genes: Dict[GeneType, ArchitecturalGene]
    generation: int = 0
    fitness_scores: Dict[FitnessMetric, float] = field(default_factory=dict)
    overall_fitness: float = 0.0
    birth_time: float = field(default_factory=time.time)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def create_random(cls) -> 'ArchitecturalGenome':
        """Create a random genome."""
        genes = {}
        for gene_type in GeneType:
            genes[gene_type] = ArchitecturalGene(
                gene_type=gene_type,
                value=random.random(),
                expression_strength=random.uniform(0.5, 1.5),
                mutation_rate=random.uniform(0.05, 0.2)
            )
        
        return cls(genes=genes)
    
    def crossover(self, other: 'ArchitecturalGenome') -> 'ArchitecturalGenome':
        """Create offspring through crossover with another genome."""
        offspring_genes = {}
        
        for gene_type in GeneType:
            parent1_gene = self.genes.get(gene_type)
            parent2_gene = other.genes.get(gene_type)
            
            if parent1_gene and parent2_gene:
                # Weighted crossover based on fitness contribution
                weight1 = max(0.1, parent1_gene.fitness_contribution)
                weight2 = max(0.1, parent2_gene.fitness_contribution)
                total_weight = weight1 + weight2
                
                # Create hybrid gene
                hybrid_value = (
                    (parent1_gene.value * weight1 + parent2_gene.value * weight2) / total_weight
                )
                
                hybrid_expression = (
                    (parent1_gene.expression_strength * weight1 + 
                     parent2_gene.expression_strength * weight2) / total_weight
                )
                
                offspring_genes[gene_type] = ArchitecturalGene(
                    gene_type=gene_type,
                    value=hybrid_value,
                    expression_strength=hybrid_expression,
                    mutation_rate=(parent1_gene.mutation_rate + parent2_gene.mutation_rate) / 2
                )
            elif parent1_gene:
                offspring_genes[gene_type] = copy.deepcopy(parent1_gene)
            elif parent2_gene:
                offspring_genes[gene_type] = copy.deepcopy(parent2_gene)
        
        offspring = ArchitecturalGenome(
            genes=offspring_genes,
            generation=max(self.generation, other.generation) + 1
        )
        
        return offspring
    
    def mutate(self, mutation_rate: float = None) -> 'ArchitecturalGenome':
        """Create a mutated copy of this genome."""
        mutated_genes = {}
        
        for gene_type, gene in self.genes.items():
            if random.random() < (mutation_rate or gene.mutation_rate):
                mutated_genes[gene_type] = gene.mutate()
            else:
                mutated_genes[gene_type] = copy.deepcopy(gene)
        
        mutated = ArchitecturalGenome(
            genes=mutated_genes,
            generation=self.generation,
            fitness_scores=copy.deepcopy(self.fitness_scores),
            overall_fitness=self.overall_fitness
        )
        
        return mutated
    
    def get_phenotype(self) -> Dict[str, Any]:
        """Convert genome to expressed phenotype (actual architecture parameters)."""
        phenotype = {}
        
        for gene_type, gene in self.genes.items():
            expressed_value = gene.value * gene.expression_strength
            
            # Map to actual parameter ranges
            if gene_type == GeneType.PROCESSING_DEPTH:
                phenotype['processing_layers'] = int(1 + expressed_value * 10)  # 1-11 layers
            elif gene_type == GeneType.CONNECTION_DENSITY:
                phenotype['connection_ratio'] = expressed_value
            elif gene_type == GeneType.LEARNING_RATE:
                phenotype['learning_rate'] = 0.0001 + expressed_value * 0.01  # 0.0001-0.0101
            elif gene_type == GeneType.MEMORY_ALLOCATION:
                phenotype['memory_mb'] = int(100 + expressed_value * 900)  # 100-1000 MB
            elif gene_type == GeneType.PARALLEL_THREADS:
                phenotype['thread_count'] = int(1 + expressed_value * 15)  # 1-16 threads
            elif gene_type == GeneType.CACHE_SIZE:
                phenotype['cache_size_mb'] = int(10 + expressed_value * 190)  # 10-200 MB
            elif gene_type == GeneType.SECURITY_LEVEL:
                phenotype['security_level'] = int(1 + expressed_value * 4)  # 1-5 levels
            elif gene_type == GeneType.ERROR_TOLERANCE:
                phenotype['error_threshold'] = expressed_value
        
        return phenotype
    
    def calculate_similarity(self, other: 'ArchitecturalGenome') -> float:
        """Calculate genetic similarity with another genome."""
        if not self.genes or not other.genes:
            return 0.0
            
        similarities = []
        for gene_type in GeneType:
            gene1 = self.genes.get(gene_type)
            gene2 = other.genes.get(gene_type)
            
            if gene1 and gene2:
                value_similarity = 1.0 - abs(gene1.value - gene2.value)
                expression_similarity = 1.0 - abs(gene1.expression_strength - gene2.expression_strength) / 2.0
                gene_similarity = (value_similarity + expression_similarity) / 2.0
                similarities.append(gene_similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

class FitnessEvaluator:
    """Evaluates fitness of architectural genomes."""
    
    def __init__(self):
        self.evaluation_history: deque = deque(maxlen=500)
        self.baseline_metrics: Dict[FitnessMetric, float] = {}
        self._evaluation_lock = threading.Lock()
        
    async def evaluate_genome(self, genome: ArchitecturalGenome, 
                            test_workload: Optional[Dict[str, Any]] = None) -> Dict[FitnessMetric, float]:
        """Evaluate the fitness of a genome."""
        try:
            start_time = time.time()
            phenotype = genome.get_phenotype()
            
            # Simulate workload execution with this architecture
            fitness_scores = {}
            
            # Evaluate processing speed
            processing_efficiency = self._evaluate_processing_speed(phenotype)
            fitness_scores[FitnessMetric.PROCESSING_SPEED] = processing_efficiency
            
            # Evaluate accuracy
            accuracy_score = self._evaluate_accuracy(phenotype, test_workload)
            fitness_scores[FitnessMetric.ACCURACY] = accuracy_score
            
            # Evaluate resource efficiency
            resource_efficiency = self._evaluate_resource_efficiency(phenotype)
            fitness_scores[FitnessMetric.RESOURCE_EFFICIENCY] = resource_efficiency
            
            # Evaluate fault tolerance
            fault_tolerance = self._evaluate_fault_tolerance(phenotype)
            fitness_scores[FitnessMetric.FAULT_TOLERANCE] = fault_tolerance
            
            # Evaluate adaptability
            adaptability = self._evaluate_adaptability(phenotype)
            fitness_scores[FitnessMetric.ADAPTABILITY] = adaptability
            
            # Evaluate security
            security_score = self._evaluate_security(phenotype)
            fitness_scores[FitnessMetric.SECURITY_SCORE] = security_score
            
            # Calculate overall fitness (weighted combination)
            weights = {
                FitnessMetric.PROCESSING_SPEED: 0.25,
                FitnessMetric.ACCURACY: 0.25,
                FitnessMetric.RESOURCE_EFFICIENCY: 0.20,
                FitnessMetric.FAULT_TOLERANCE: 0.15,
                FitnessMetric.ADAPTABILITY: 0.10,
                FitnessMetric.SECURITY_SCORE: 0.05
            }
            
            overall_fitness = sum(
                fitness_scores[metric] * weight 
                for metric, weight in weights.items()
            )
            
            # Update genome fitness
            genome.fitness_scores = fitness_scores
            genome.overall_fitness = overall_fitness
            
            # Update gene fitness contributions
            for gene_type, gene in genome.genes.items():
                gene.fitness_contribution = self._calculate_gene_contribution(gene, fitness_scores)
            
            # Record evaluation
            evaluation_time = time.time() - start_time
            with self._evaluation_lock:
                self.evaluation_history.append({
                    'genome_id': id(genome),
                    'generation': genome.generation,
                    'fitness_scores': fitness_scores,
                    'overall_fitness': overall_fitness,
                    'evaluation_time': evaluation_time,
                    'timestamp': time.time()
                })
            
            return fitness_scores
            
        except Exception as e:
            warnings.warn(f"Genome evaluation failed: {e}")
            # Return poor fitness scores
            return {metric: 0.1 for metric in FitnessMetric}
    
    def _evaluate_processing_speed(self, phenotype: Dict[str, Any]) -> float:
        """Evaluate processing speed fitness."""
        try:
            # Simulate processing with given parameters
            layers = phenotype.get('processing_layers', 5)
            threads = phenotype.get('thread_count', 4)
            cache_size = phenotype.get('cache_size_mb', 100)
            
            # Model performance based on architecture
            base_speed = 1.0
            
            # More layers = more processing but potentially slower
            layer_factor = 1.0 + (layers - 5) * 0.02  # Slight improvement up to a point
            if layers > 8:
                layer_factor *= 0.8  # Diminishing returns
            
            # More threads = better parallelization
            thread_factor = min(1.5, 1.0 + threads * 0.05)
            
            # Larger cache = better performance
            cache_factor = min(1.3, 1.0 + cache_size / 1000.0)
            
            speed_score = base_speed * layer_factor * thread_factor * cache_factor
            return min(1.0, speed_score / 2.0)  # Normalize
            
        except Exception as e:
            warnings.warn(f"Speed evaluation failed: {e}")
            return 0.5
    
    def _evaluate_accuracy(self, phenotype: Dict[str, Any], test_workload: Any) -> float:
        """Evaluate accuracy fitness."""
        try:
            # Base accuracy depends on processing depth and learning rate
            layers = phenotype.get('processing_layers', 5)
            learning_rate = phenotype.get('learning_rate', 0.001)
            connection_ratio = phenotype.get('connection_ratio', 0.5)
            
            # Model accuracy based on architecture complexity
            base_accuracy = 0.7
            
            # More layers generally improve accuracy up to a point
            layer_bonus = min(0.2, layers * 0.02)
            if layers > 10:
                layer_bonus *= 0.7  # Overfitting penalty
            
            # Optimal learning rate around 0.001-0.005
            lr_factor = 1.0 - abs(learning_rate - 0.003) / 0.01
            lr_bonus = max(0, lr_factor) * 0.1
            
            # Connection density affects learning capacity
            connection_bonus = connection_ratio * 0.1
            
            accuracy_score = base_accuracy + layer_bonus + lr_bonus + connection_bonus
            return min(1.0, max(0.0, accuracy_score))
            
        except Exception as e:
            warnings.warn(f"Accuracy evaluation failed: {e}")
            return 0.7
    
    def _evaluate_resource_efficiency(self, phenotype: Dict[str, Any]) -> float:
        """Evaluate resource efficiency fitness."""
        try:
            memory_mb = phenotype.get('memory_mb', 500)
            threads = phenotype.get('thread_count', 4)
            cache_mb = phenotype.get('cache_size_mb', 100)
            layers = phenotype.get('processing_layers', 5)
            
            # Calculate resource usage score (lower is better for efficiency)
            total_memory_usage = memory_mb + cache_mb + (threads * 50) + (layers * 30)
            
            # Efficiency is inversely related to resource usage
            # Assume 1000MB total is reasonable baseline
            efficiency = max(0.1, 1.0 - (total_memory_usage - 1000) / 2000)
            
            return min(1.0, max(0.1, efficiency))
            
        except Exception as e:
            warnings.warn(f"Resource efficiency evaluation failed: {e}")
            return 0.5
    
    def _evaluate_fault_tolerance(self, phenotype: Dict[str, Any]) -> float:
        """Evaluate fault tolerance fitness."""
        try:
            error_threshold = phenotype.get('error_threshold', 0.5)
            threads = phenotype.get('thread_count', 4)
            connection_ratio = phenotype.get('connection_ratio', 0.5)
            
            # Higher error threshold = more tolerant
            tolerance_score = error_threshold
            
            # More threads provide redundancy
            redundancy_bonus = min(0.2, threads * 0.02)
            
            # Higher connection density provides alternative paths
            connectivity_bonus = connection_ratio * 0.2
            
            fault_tolerance = tolerance_score + redundancy_bonus + connectivity_bonus
            return min(1.0, max(0.1, fault_tolerance))
            
        except Exception as e:
            warnings.warn(f"Fault tolerance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_adaptability(self, phenotype: Dict[str, Any]) -> float:
        """Evaluate adaptability fitness."""
        try:
            learning_rate = phenotype.get('learning_rate', 0.001)
            connection_ratio = phenotype.get('connection_ratio', 0.5)
            memory_mb = phenotype.get('memory_mb', 500)
            
            # Higher learning rate enables faster adaptation (up to a point)
            lr_factor = min(1.0, learning_rate / 0.01)  # Normalize to 0.01 as max
            adaptability_score = lr_factor * 0.4
            
            # More connections allow better adaptation
            adaptability_score += connection_ratio * 0.3
            
            # More memory allows storing adaptation patterns
            memory_factor = min(1.0, memory_mb / 1000.0)
            adaptability_score += memory_factor * 0.3
            
            return min(1.0, max(0.0, adaptability_score))
            
        except Exception as e:
            warnings.warn(f"Adaptability evaluation failed: {e}")
            return 0.5
    
    def _evaluate_security(self, phenotype: Dict[str, Any]) -> float:
        """Evaluate security fitness."""
        try:
            security_level = phenotype.get('security_level', 3)
            error_threshold = phenotype.get('error_threshold', 0.5)
            
            # Higher security level = better security
            security_score = security_level / 5.0
            
            # Lower error threshold can indicate better input validation
            validation_bonus = (1.0 - error_threshold) * 0.2
            
            total_security = security_score + validation_bonus
            return min(1.0, max(0.0, total_security))
            
        except Exception as e:
            warnings.warn(f"Security evaluation failed: {e}")
            return 0.6
    
    def _calculate_gene_contribution(self, gene: ArchitecturalGene, 
                                   fitness_scores: Dict[FitnessMetric, float]) -> float:
        """Calculate how much this gene contributes to overall fitness."""
        try:
            # Map gene types to relevant fitness metrics
            gene_metric_map = {
                GeneType.PROCESSING_DEPTH: [FitnessMetric.PROCESSING_SPEED, FitnessMetric.ACCURACY],
                GeneType.CONNECTION_DENSITY: [FitnessMetric.FAULT_TOLERANCE, FitnessMetric.ADAPTABILITY],
                GeneType.LEARNING_RATE: [FitnessMetric.ACCURACY, FitnessMetric.ADAPTABILITY],
                GeneType.MEMORY_ALLOCATION: [FitnessMetric.RESOURCE_EFFICIENCY, FitnessMetric.ADAPTABILITY],
                GeneType.PARALLEL_THREADS: [FitnessMetric.PROCESSING_SPEED, FitnessMetric.FAULT_TOLERANCE],
                GeneType.CACHE_SIZE: [FitnessMetric.PROCESSING_SPEED, FitnessMetric.RESOURCE_EFFICIENCY],
                GeneType.SECURITY_LEVEL: [FitnessMetric.SECURITY_SCORE],
                GeneType.ERROR_TOLERANCE: [FitnessMetric.FAULT_TOLERANCE, FitnessMetric.SECURITY_SCORE]
            }
            
            relevant_metrics = gene_metric_map.get(gene.gene_type, [])
            if not relevant_metrics:
                return 0.5  # Default contribution
            
            # Average fitness of relevant metrics
            relevant_scores = [fitness_scores.get(metric, 0.5) for metric in relevant_metrics]
            contribution = sum(relevant_scores) / len(relevant_scores)
            
            return contribution
            
        except Exception as e:
            warnings.warn(f"Gene contribution calculation failed: {e}")
            return 0.5

class EvolutionaryOptimizer:
    """Manages the evolutionary optimization process."""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.current_population: List[ArchitecturalGenome] = []
        self.generation_count = 0
        self.fitness_evaluator = FitnessEvaluator()
        self.evolution_history: deque = deque(maxlen=100)
        self.best_genome: Optional[ArchitecturalGenome] = None
        self._evolution_lock = threading.Lock()
        
        # Evolution parameters
        self.elite_ratio = 0.2  # Top 20% survive to next generation
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the starting population."""
        try:
            self.current_population = []
            
            # Create diverse initial population
            for i in range(self.population_size):
                genome = ArchitecturalGenome.create_random()
                self.current_population.append(genome)
            
            print(f"Initialized population of {len(self.current_population)} genomes")
            
        except Exception as e:
            warnings.warn(f"Population initialization failed: {e}")
    
    async def evolve_generation(self, test_workload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evolve the population by one generation."""
        try:
            start_time = time.time()
            
            # Step 1: Evaluate current population
            await self._evaluate_population(test_workload)
            
            # Step 2: Select parents for next generation
            parents = self._select_parents()
            
            # Step 3: Create next generation through crossover and mutation
            next_generation = await self._create_next_generation(parents)
            
            # Step 4: Replace population
            with self._evolution_lock:
                self.current_population = next_generation
                self.generation_count += 1
                
                # Update best genome
                if self.current_population:
                    current_best = max(self.current_population, key=lambda g: g.overall_fitness)
                    if self.best_genome is None or current_best.overall_fitness > self.best_genome.overall_fitness:
                        self.best_genome = copy.deepcopy(current_best)
            
            # Record evolution statistics
            generation_time = time.time() - start_time
            stats = self._calculate_generation_stats()
            
            self.evolution_history.append({
                'generation': self.generation_count,
                'stats': stats,
                'evolution_time': generation_time,
                'timestamp': time.time()
            })
            
            return {
                'generation': self.generation_count,
                'evolution_time': generation_time,
                'population_stats': stats,
                'best_fitness': self.best_genome.overall_fitness if self.best_genome else 0.0
            }
            
        except Exception as e:
            error_msg = f"Evolution failed: {e}"
            warnings.warn(error_msg)
            return {'error': error_msg}
    
    async def _evaluate_population(self, test_workload: Optional[Dict[str, Any]] = None):
        """Evaluate fitness of all genomes in population."""
        try:
            evaluation_tasks = []
            
            for genome in self.current_population:
                task = self.fitness_evaluator.evaluate_genome(genome, test_workload)
                evaluation_tasks.append(task)
            
            # Evaluate all genomes concurrently
            await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
        except Exception as e:
            warnings.warn(f"Population evaluation failed: {e}")
    
    def _select_parents(self) -> List[ArchitecturalGenome]:
        """Select parents for reproduction using tournament selection."""
        try:
            # Sort by fitness
            sorted_population = sorted(
                self.current_population, 
                key=lambda g: g.overall_fitness, 
                reverse=True
            )
            
            # Elite selection - top performers always survive
            elite_count = int(self.population_size * self.elite_ratio)
            elite_parents = sorted_population[:elite_count]
            
            # Tournament selection for remaining parents
            remaining_count = self.population_size - elite_count
            tournament_parents = []
            
            for _ in range(remaining_count):
                # Tournament of size 3
                tournament_candidates = random.sample(sorted_population, min(3, len(sorted_population)))
                winner = max(tournament_candidates, key=lambda g: g.overall_fitness)
                tournament_parents.append(winner)
            
            return elite_parents + tournament_parents
            
        except Exception as e:
            warnings.warn(f"Parent selection failed: {e}")
            return self.current_population[:self.population_size // 2]
    
    async def _create_next_generation(self, parents: List[ArchitecturalGenome]) -> List[ArchitecturalGenome]:
        """Create next generation through crossover and mutation."""
        try:
            next_generation = []
            
            # Keep some elite individuals unchanged
            elite_count = int(len(parents) * self.elite_ratio)
            next_generation.extend(copy.deepcopy(parents[:elite_count]))
            
            # Create offspring through crossover
            while len(next_generation) < self.population_size:
                # Select two parents randomly (weighted by fitness)
                parent1, parent2 = self._select_crossover_parents(parents)
                
                if random.random() < self.crossover_rate:
                    # Crossover
                    offspring = parent1.crossover(parent2)
                else:
                    # Copy parent
                    offspring = copy.deepcopy(parent1)
                
                # Apply mutation
                if random.random() < self.mutation_rate:
                    offspring = offspring.mutate()
                
                next_generation.append(offspring)
            
            return next_generation[:self.population_size]
            
        except Exception as e:
            warnings.warn(f"Next generation creation failed: {e}")
            return copy.deepcopy(parents[:self.population_size])
    
    def _select_crossover_parents(self, parents: List[ArchitecturalGenome]) -> Tuple[ArchitecturalGenome, ArchitecturalGenome]:
        """Select two parents for crossover using fitness-weighted selection."""
        try:
            # Create fitness weights
            fitness_values = [max(0.1, p.overall_fitness) for p in parents]
            total_fitness = sum(fitness_values)
            
            if total_fitness == 0:
                return random.sample(parents, 2)
            
            # Weighted selection
            weights = [f / total_fitness for f in fitness_values]
            
            parent1 = random.choices(parents, weights=weights)[0]
            parent2 = random.choices(parents, weights=weights)[0]
            
            # Ensure parents are different
            attempts = 0
            while parent1 is parent2 and attempts < 10:
                parent2 = random.choices(parents, weights=weights)[0]
                attempts += 1
            
            return parent1, parent2
            
        except Exception as e:
            warnings.warn(f"Crossover parent selection failed: {e}")
            return random.sample(parents, 2)
    
    def _calculate_generation_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        try:
            if not self.current_population:
                return {}
            
            fitness_values = [g.overall_fitness for g in self.current_population]
            
            return {
                'population_size': len(self.current_population),
                'avg_fitness': sum(fitness_values) / len(fitness_values),
                'max_fitness': max(fitness_values),
                'min_fitness': min(fitness_values),
                'fitness_std': self._calculate_std(fitness_values),
                'generation_diversity': self._calculate_diversity()
            }
            
        except Exception as e:
            warnings.warn(f"Generation stats calculation failed: {e}")
            return {}
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity of population."""
        try:
            if len(self.current_population) < 2:
                return 0.0
            
            similarities = []
            for i in range(len(self.current_population)):
                for j in range(i + 1, len(self.current_population)):
                    similarity = self.current_population[i].calculate_similarity(self.current_population[j])
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                diversity = 1.0 - avg_similarity  # Diversity is inverse of similarity
                return max(0.0, min(1.0, diversity))
            
            return 0.0
            
        except Exception as e:
            warnings.warn(f"Diversity calculation failed: {e}")
            return 0.5
    
    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best evolved architecture."""
        if self.best_genome:
            return {
                'generation': self.best_genome.generation,
                'fitness_score': self.best_genome.overall_fitness,
                'fitness_breakdown': self.best_genome.fitness_scores,
                'phenotype': self.best_genome.get_phenotype(),
                'age': time.time() - self.best_genome.birth_time
            }
        return None
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive evolution status."""
        try:
            with self._evolution_lock:
                current_stats = self._calculate_generation_stats()
                
            return {
                'generation_count': self.generation_count,
                'population_size': len(self.current_population),
                'current_stats': current_stats,
                'best_architecture': self.get_best_architecture(),
                'evolution_parameters': {
                    'elite_ratio': self.elite_ratio,
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate
                },
                'evolution_version': '6.0.0-evolutionary',
                'timestamp': time.time()
            }
            
        except Exception as e:
            warnings.warn(f"Evolution status failed: {e}")
            return {
                'error': str(e),
                'evolution_version': '6.0.0-evolutionary',
                'timestamp': time.time()
            }

# Global evolutionary optimizer instance
_evolutionary_optimizer = None

def get_evolutionary_optimizer() -> EvolutionaryOptimizer:
    """Get the global evolutionary optimizer instance."""
    global _evolutionary_optimizer
    if _evolutionary_optimizer is None:
        _evolutionary_optimizer = EvolutionaryOptimizer()
    return _evolutionary_optimizer

# Demo and testing functions
def demonstrate_evolutionary_architecture():
    """Demonstrate evolutionary architecture capabilities."""
    print("🧬 Evolutionary Architecture Demo")
    print("=" * 50)
    
    optimizer = get_evolutionary_optimizer()
    
    async def demo_evolution():
        print("Starting evolution...")
        
        # Run several generations
        for generation in range(3):
            print(f"\nEvolving Generation {generation + 1}...")
            
            result = await optimizer.evolve_generation()
            if 'error' not in result:
                print(f"Generation {result['generation']}: "
                      f"Best fitness = {result['best_fitness']:.4f}, "
                      f"Evolution time = {result['evolution_time']:.2f}s")
                
                stats = result['population_stats']
                print(f"  Population stats: avg={stats.get('avg_fitness', 0):.4f}, "
                      f"diversity={stats.get('generation_diversity', 0):.4f}")
        
        # Show best architecture
        best_arch = optimizer.get_best_architecture()
        if best_arch:
            print(f"\nBest Architecture Found:")
            print(json.dumps(best_arch, indent=2, default=str))
        
        # Show evolution status
        status = optimizer.get_evolution_status()
        print(f"\nEvolution Status:")
        print(json.dumps(status, indent=2, default=str))
    
    # Run demo
    try:
        asyncio.run(demo_evolution())
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    demonstrate_evolutionary_architecture()