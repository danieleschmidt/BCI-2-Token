"""
Neural Mesh Network - Generation 6+ Distributed Intelligence
==========================================================

Revolutionary distributed neural processing implementing:
- Mesh topology with self-organizing nodes
- Distributed consciousness across multiple processing units
- Swarm intelligence for collective decision making
- Fault-tolerant neural redundancy
- Dynamic load balancing across neural processors
- Emergent intelligence from distributed components

This creates a living, breathing neural network that adapts
and evolves its own topology for optimal performance.
"""

import asyncio
import time
import threading
import json
import math
import random
import hashlib
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
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
    warnings.warn("NumPy not available. Neural mesh features will be limited.")

class NodeType(Enum):
    """Types of neural mesh nodes."""
    PREPROCESSOR = "preprocessor"
    ANALYZER = "analyzer"
    DECODER = "decoder"
    AGGREGATOR = "aggregator"
    COORDINATOR = "coordinator"
    MEMORY = "memory"
    SECURITY = "security"

class NodeState(Enum):
    """Neural node states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    IDLE = "idle"
    OVERLOADED = "overloaded"
    FAILING = "failing"
    OFFLINE = "offline"

class ConnectionStrength(Enum):
    """Connection strength between nodes."""
    WEAK = 0.1
    MODERATE = 0.5
    STRONG = 0.8
    CRITICAL = 1.0

@dataclass
class NeuralNode:
    """Individual neural processing node."""
    node_id: str
    node_type: NodeType
    position: Tuple[float, float, float]  # 3D position in mesh
    state: NodeState = NodeState.INITIALIZING
    processing_capacity: float = 1.0
    current_load: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    specializations: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        self.creation_time = time.time()
        self.total_processed = 0
        self.success_rate = 1.0
        
    def calculate_distance_to(self, other: 'NeuralNode') -> float:
        """Calculate 3D distance to another node."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def update_load(self, new_load: float):
        """Update processing load and state."""
        self.current_load = new_load
        
        if new_load > 0.9:
            self.state = NodeState.OVERLOADED
        elif new_load > 0.7:
            self.state = NodeState.PROCESSING
        elif new_load > 0.1:
            self.state = NodeState.ACTIVE
        else:
            self.state = NodeState.IDLE
    
    def add_performance_record(self, processing_time: float, success: bool):
        """Add performance record."""
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success,
            'load': self.current_load
        })
        
        # Update success rate
        if self.performance_history:
            successes = sum(1 for r in self.performance_history if r['success'])
            self.success_rate = successes / len(self.performance_history)
    
    def get_efficiency_score(self) -> float:
        """Calculate node efficiency score."""
        base_score = self.success_rate
        load_penalty = min(0.3, self.current_load * 0.3)  # Penalty for high load
        uptime_bonus = min(0.1, (time.time() - self.creation_time) / 86400 * 0.1)  # Daily bonus
        
        return max(0.0, base_score - load_penalty + uptime_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'position': self.position,
            'state': self.state.value,
            'processing_capacity': self.processing_capacity,
            'current_load': self.current_load,
            'connections': self.connections,
            'specializations': self.specializations,
            'last_heartbeat': self.last_heartbeat,
            'success_rate': self.success_rate,
            'efficiency_score': self.get_efficiency_score(),
            'total_processed': self.total_processed
        }

class SwarmIntelligence:
    """Collective intelligence system for the neural mesh."""
    
    def __init__(self):
        self.collective_memory: Dict[str, Any] = {}
        self.decision_history: deque = deque(maxlen=500)
        self.learning_patterns: Dict[str, List[float]] = defaultdict(list)
        self.consensus_threshold = 0.7
        self._swarm_lock = threading.Lock()
        
    def propose_decision(self, decision_context: Dict[str, Any], 
                        participating_nodes: List[NeuralNode]) -> Dict[str, Any]:
        """Propose a decision using swarm intelligence."""
        try:
            decision_id = secrets.token_hex(8)
            
            # Collect node opinions based on their specializations and performance
            node_opinions = []
            for node in participating_nodes:
                if node.state in [NodeState.ACTIVE, NodeState.PROCESSING]:
                    opinion_weight = node.get_efficiency_score()
                    
                    # Generate opinion based on node specialization
                    opinion = self._generate_node_opinion(node, decision_context)
                    
                    node_opinions.append({
                        'node_id': node.node_id,
                        'opinion': opinion,
                        'weight': opinion_weight,
                        'confidence': min(1.0, node.success_rate * opinion_weight)
                    })
            
            # Aggregate opinions using weighted voting
            if node_opinions:
                total_weight = sum(op['weight'] for op in node_opinions)
                weighted_decision = sum(op['opinion'] * op['weight'] for op in node_opinions) / total_weight
                average_confidence = sum(op['confidence'] for op in node_opinions) / len(node_opinions)
            else:
                weighted_decision = 0.5  # Neutral decision
                average_confidence = 0.5
                
            # Determine consensus level
            consensus_level = self._calculate_consensus(node_opinions)
            
            decision_result = {
                'decision_id': decision_id,
                'decision_value': weighted_decision,
                'confidence': average_confidence,
                'consensus_level': consensus_level,
                'participating_nodes': len(node_opinions),
                'timestamp': time.time(),
                'context': decision_context
            }
            
            # Store in collective memory and history
            with self._swarm_lock:
                self.collective_memory[decision_id] = decision_result
                self.decision_history.append(decision_result)
                
                # Update learning patterns
                context_key = decision_context.get('type', 'general')
                self.learning_patterns[context_key].append(weighted_decision)
                
            return decision_result
            
        except Exception as e:
            warnings.warn(f"Swarm decision failed: {e}")
            return {
                'decision_id': 'error',
                'decision_value': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_node_opinion(self, node: NeuralNode, context: Dict[str, Any]) -> float:
        """Generate a node's opinion on a decision."""
        try:
            base_opinion = 0.5  # Neutral starting point
            
            # Adjust based on node type and specializations
            if node.node_type == NodeType.ANALYZER and 'analysis' in context:
                base_opinion += 0.2
            elif node.node_type == NodeType.SECURITY and 'security' in context:
                base_opinion += 0.3
            elif node.node_type == NodeType.COORDINATOR and 'coordination' in context:
                base_opinion += 0.2
                
            # Adjust based on current load (overloaded nodes are more conservative)
            if node.current_load > 0.8:
                base_opinion -= 0.1
            elif node.current_load < 0.3:
                base_opinion += 0.1
                
            # Add some randomness for diversity
            opinion_noise = random.uniform(-0.1, 0.1)
            final_opinion = max(0.0, min(1.0, base_opinion + opinion_noise))
            
            return final_opinion
            
        except Exception as e:
            warnings.warn(f"Node opinion generation failed: {e}")
            return 0.5
    
    def _calculate_consensus(self, opinions: List[Dict[str, Any]]) -> float:
        """Calculate consensus level among node opinions."""
        if not opinions:
            return 0.0
            
        try:
            opinion_values = [op['opinion'] for op in opinions]
            mean_opinion = sum(opinion_values) / len(opinion_values)
            
            # Calculate variance to determine consensus
            variance = sum((op - mean_opinion) ** 2 for op in opinion_values) / len(opinion_values)
            consensus = math.exp(-variance * 4)  # High consensus when low variance
            
            return min(1.0, max(0.0, consensus))
            
        except Exception as e:
            warnings.warn(f"Consensus calculation failed: {e}")
            return 0.0

class MeshTopologyManager:
    """Manages the topology and connections of the neural mesh."""
    
    def __init__(self):
        self.nodes: Dict[str, NeuralNode] = {}
        self.connection_matrix: Dict[Tuple[str, str], float] = {}
        self.topology_version = 0
        self._topology_lock = threading.Lock()
        
    def add_node(self, node_type: NodeType, position: Optional[Tuple[float, float, float]] = None,
                specializations: Optional[List[str]] = None) -> NeuralNode:
        """Add a new node to the mesh."""
        try:
            node_id = f"{node_type.value}_{secrets.token_hex(6)}"
            
            if position is None:
                # Generate random position in 3D space
                position = (
                    random.uniform(-10.0, 10.0),
                    random.uniform(-10.0, 10.0),
                    random.uniform(-10.0, 10.0)
                )
                
            if specializations is None:
                specializations = []
                
            new_node = NeuralNode(
                node_id=node_id,
                node_type=node_type,
                position=position,
                specializations=specializations
            )
            
            with self._topology_lock:
                self.nodes[node_id] = new_node
                self._establish_initial_connections(new_node)
                self.topology_version += 1
                
            return new_node
            
        except Exception as e:
            warnings.warn(f"Node addition failed: {e}")
            raise
    
    def _establish_initial_connections(self, new_node: NeuralNode):
        """Establish initial connections for a new node."""
        try:
            # Connect to nearby nodes and complementary types
            for existing_id, existing_node in self.nodes.items():
                if existing_id == new_node.node_id:
                    continue
                    
                # Calculate connection strength based on distance and compatibility
                distance = new_node.calculate_distance_to(existing_node)
                type_compatibility = self._calculate_type_compatibility(new_node, existing_node)
                
                # Distance-based connection (closer = stronger)
                distance_factor = math.exp(-distance / 5.0)  # Exponential decay
                
                # Final connection strength
                connection_strength = min(1.0, distance_factor * type_compatibility)
                
                if connection_strength > 0.1:  # Only create meaningful connections
                    new_node.connections[existing_id] = connection_strength
                    existing_node.connections[new_node.node_id] = connection_strength
                    
                    # Store in connection matrix
                    self.connection_matrix[(new_node.node_id, existing_id)] = connection_strength
                    self.connection_matrix[(existing_id, new_node.node_id)] = connection_strength
                    
        except Exception as e:
            warnings.warn(f"Initial connections failed: {e}")
    
    def _calculate_type_compatibility(self, node1: NeuralNode, node2: NeuralNode) -> float:
        """Calculate compatibility between node types."""
        # Define complementary relationships
        complementary_pairs = {
            (NodeType.PREPROCESSOR, NodeType.ANALYZER): 0.9,
            (NodeType.ANALYZER, NodeType.DECODER): 0.8,
            (NodeType.DECODER, NodeType.AGGREGATOR): 0.8,
            (NodeType.AGGREGATOR, NodeType.COORDINATOR): 0.7,
            (NodeType.SECURITY, NodeType.COORDINATOR): 0.8,
            (NodeType.MEMORY, NodeType.ANALYZER): 0.7,
            (NodeType.MEMORY, NodeType.AGGREGATOR): 0.7,
        }
        
        # Check both directions
        pair1 = (node1.node_type, node2.node_type)
        pair2 = (node2.node_type, node1.node_type)
        
        if pair1 in complementary_pairs:
            return complementary_pairs[pair1]
        elif pair2 in complementary_pairs:
            return complementary_pairs[pair2]
        else:
            return 0.5  # Default compatibility
    
    def optimize_topology(self):
        """Optimize the mesh topology based on performance."""
        try:
            with self._topology_lock:
                # Remove weak performing connections
                connections_to_remove = []
                
                for (node1_id, node2_id), strength in self.connection_matrix.items():
                    node1 = self.nodes.get(node1_id)
                    node2 = self.nodes.get(node2_id)
                    
                    if node1 and node2:
                        # Calculate performance-based connection value
                        combined_efficiency = (node1.get_efficiency_score() + node2.get_efficiency_score()) / 2
                        
                        # Remove weak connections between low-performing nodes
                        if strength < 0.3 and combined_efficiency < 0.5:
                            connections_to_remove.append((node1_id, node2_id))
                
                # Remove identified weak connections
                for node1_id, node2_id in connections_to_remove:
                    if node1_id in self.nodes and node2_id in self.nodes:
                        self.nodes[node1_id].connections.pop(node2_id, None)
                        self.nodes[node2_id].connections.pop(node1_id, None)
                        self.connection_matrix.pop((node1_id, node2_id), None)
                        self.connection_matrix.pop((node2_id, node1_id), None)
                
                # Add new connections between high-performing nodes
                high_performers = [
                    (node_id, node) for node_id, node in self.nodes.items()
                    if node.get_efficiency_score() > 0.7
                ]
                
                for i, (id1, node1) in enumerate(high_performers):
                    for id2, node2 in high_performers[i+1:]:
                        if id2 not in node1.connections and len(node1.connections) < 5:
                            # Create new connection between high performers
                            new_strength = min(0.8, (node1.get_efficiency_score() + node2.get_efficiency_score()) / 2)
                            node1.connections[id2] = new_strength
                            node2.connections[id1] = new_strength
                            self.connection_matrix[(id1, id2)] = new_strength
                            self.connection_matrix[(id2, id1)] = new_strength
                
                self.topology_version += 1
                
        except Exception as e:
            warnings.warn(f"Topology optimization failed: {e}")
    
    def get_topology_stats(self) -> Dict[str, Any]:
        """Get topology statistics."""
        try:
            with self._topology_lock:
                total_nodes = len(self.nodes)
                total_connections = len(self.connection_matrix) // 2  # Undirected graph
                
                node_type_distribution = defaultdict(int)
                state_distribution = defaultdict(int)
                efficiency_scores = []
                
                for node in self.nodes.values():
                    node_type_distribution[node.node_type.value] += 1
                    state_distribution[node.state.value] += 1
                    efficiency_scores.append(node.get_efficiency_score())
                
                avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
                
                return {
                    'total_nodes': total_nodes,
                    'total_connections': total_connections,
                    'topology_version': self.topology_version,
                    'node_type_distribution': dict(node_type_distribution),
                    'state_distribution': dict(state_distribution),
                    'average_efficiency': avg_efficiency,
                    'connectivity_ratio': total_connections / max(1, total_nodes * (total_nodes - 1) / 2),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            warnings.warn(f"Topology stats failed: {e}")
            return {'error': str(e)}

class NeuralMeshNetwork:
    """Main neural mesh network coordinator."""
    
    def __init__(self):
        self.topology_manager = MeshTopologyManager()
        self.swarm_intelligence = SwarmIntelligence()
        self.processing_queue = asyncio.Queue()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._network_lock = threading.Lock()
        
        # Initialize default topology
        self._create_initial_topology()
        
        # Start background optimization
        self._optimization_task = None
        
    def _create_initial_topology(self):
        """Create initial mesh topology."""
        try:
            # Create core nodes
            coordinator = self.topology_manager.add_node(
                NodeType.COORDINATOR, 
                position=(0, 0, 0),
                specializations=['coordination', 'load_balancing']
            )
            
            # Create preprocessing nodes
            for i in range(2):
                self.topology_manager.add_node(
                    NodeType.PREPROCESSOR,
                    position=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-2, 2)),
                    specializations=['signal_filtering', 'artifact_removal']
                )
            
            # Create analyzer nodes
            for i in range(3):
                self.topology_manager.add_node(
                    NodeType.ANALYZER,
                    position=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-2, 2)),
                    specializations=['pattern_recognition', 'feature_extraction']
                )
            
            # Create decoder nodes
            for i in range(2):
                self.topology_manager.add_node(
                    NodeType.DECODER,
                    position=(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-2, 2)),
                    specializations=['neural_decoding', 'token_generation']
                )
            
            # Create support nodes
            self.topology_manager.add_node(
                NodeType.MEMORY,
                position=(random.uniform(-3, 3), random.uniform(-3, 3), 3),
                specializations=['data_storage', 'pattern_memory']
            )
            
            self.topology_manager.add_node(
                NodeType.SECURITY,
                position=(random.uniform(-3, 3), random.uniform(-3, 3), -3),
                specializations=['security_monitoring', 'anomaly_detection']
            )
            
        except Exception as e:
            warnings.warn(f"Initial topology creation failed: {e}")
    
    async def process_distributed_signal(self, neural_signals: Any, 
                                       session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process neural signals through the distributed mesh."""
        try:
            if session_id is None:
                session_id = secrets.token_hex(8)
                
            start_time = time.time()
            
            # Step 1: Select optimal processing path through mesh
            processing_path = self._select_processing_path(neural_signals)
            
            # Step 2: Distribute processing across selected nodes
            processing_results = []
            
            for node_id in processing_path:
                node = self.topology_manager.nodes.get(node_id)
                if node and node.state in [NodeState.ACTIVE, NodeState.IDLE]:
                    result = await self._process_at_node(node, neural_signals, session_id)
                    processing_results.append(result)
                    
                    # Update node load
                    node.update_load(min(1.0, node.current_load + 0.1))
            
            # Step 3: Use swarm intelligence for final decision
            decision_context = {
                'type': 'signal_processing',
                'session_id': session_id,
                'processing_path_length': len(processing_path),
                'signal_complexity': self._estimate_signal_complexity(neural_signals)
            }
            
            participating_nodes = [
                self.topology_manager.nodes[node_id] 
                for node_id in processing_path[:5]  # Limit to top 5 for efficiency
                if node_id in self.topology_manager.nodes
            ]
            
            swarm_decision = self.swarm_intelligence.propose_decision(
                decision_context, participating_nodes
            )
            
            # Step 4: Aggregate results
            processing_time = time.time() - start_time
            
            result = {
                'session_id': session_id,
                'processing_path': processing_path,
                'nodes_utilized': len(processing_results),
                'swarm_decision': swarm_decision,
                'processing_time_ms': processing_time * 1000,
                'mesh_efficiency': self._calculate_mesh_efficiency(),
                'status': 'success'
            }
            
            # Update session tracking
            with self._network_lock:
                self.active_sessions[session_id] = result
                
            return result
            
        except Exception as e:
            error_msg = f"Distributed processing failed: {e}"
            warnings.warn(error_msg)
            return {
                'session_id': session_id or 'unknown',
                'error': error_msg,
                'status': 'error'
            }
    
    def _select_processing_path(self, neural_signals: Any) -> List[str]:
        """Select optimal processing path through mesh."""
        try:
            # Start with coordinator if available
            coordinators = [
                node_id for node_id, node in self.topology_manager.nodes.items()
                if node.node_type == NodeType.COORDINATOR and node.state == NodeState.ACTIVE
            ]
            
            if coordinators:
                path = [coordinators[0]]
            else:
                path = []
            
            # Add preprocessors
            preprocessors = [
                node_id for node_id, node in self.topology_manager.nodes.items()
                if node.node_type == NodeType.PREPROCESSOR 
                and node.state in [NodeState.ACTIVE, NodeState.IDLE]
                and node.current_load < 0.8
            ]
            path.extend(sorted(preprocessors, key=lambda x: self.topology_manager.nodes[x].current_load)[:2])
            
            # Add analyzers
            analyzers = [
                node_id for node_id, node in self.topology_manager.nodes.items()
                if node.node_type == NodeType.ANALYZER
                and node.state in [NodeState.ACTIVE, NodeState.IDLE]
                and node.current_load < 0.8
            ]
            path.extend(sorted(analyzers, key=lambda x: self.topology_manager.nodes[x].get_efficiency_score(), reverse=True)[:2])
            
            # Add decoder
            decoders = [
                node_id for node_id, node in self.topology_manager.nodes.items()
                if node.node_type == NodeType.DECODER
                and node.state in [NodeState.ACTIVE, NodeState.IDLE]
                and node.current_load < 0.9
            ]
            if decoders:
                best_decoder = max(decoders, key=lambda x: self.topology_manager.nodes[x].get_efficiency_score())
                path.append(best_decoder)
            
            return path
            
        except Exception as e:
            warnings.warn(f"Path selection failed: {e}")
            # Return any available nodes
            return list(self.topology_manager.nodes.keys())[:3]
    
    async def _process_at_node(self, node: NeuralNode, signals: Any, session_id: str) -> Dict[str, Any]:
        """Process signals at a specific node."""
        try:
            start_time = time.time()
            
            # Simulate node-specific processing
            processing_delay = random.uniform(0.01, 0.05)  # 10-50ms
            await asyncio.sleep(processing_delay)
            
            # Simulate processing success/failure based on node state
            success_probability = {
                NodeState.ACTIVE: 0.95,
                NodeState.IDLE: 0.98,
                NodeState.PROCESSING: 0.85,
                NodeState.OVERLOADED: 0.60
            }.get(node.state, 0.50)
            
            success = random.random() < success_probability
            processing_time = time.time() - start_time
            
            # Update node performance
            node.add_performance_record(processing_time, success)
            node.total_processed += 1
            node.last_heartbeat = time.time()
            
            return {
                'node_id': node.node_id,
                'node_type': node.node_type.value,
                'processing_time': processing_time,
                'success': success,
                'session_id': session_id
            }
            
        except Exception as e:
            warnings.warn(f"Node processing failed: {e}")
            return {
                'node_id': node.node_id,
                'success': False,
                'error': str(e)
            }
    
    def _estimate_signal_complexity(self, signals: Any) -> float:
        """Estimate signal complexity."""
        try:
            if HAS_NUMPY and hasattr(signals, '__len__'):
                complexity = min(1.0, len(signals) / 1000.0)  # Normalize by expected size
                return complexity
            return 0.5  # Default complexity
        except:
            return 0.5
    
    def _calculate_mesh_efficiency(self) -> float:
        """Calculate overall mesh efficiency."""
        try:
            if not self.topology_manager.nodes:
                return 0.0
                
            efficiency_scores = [node.get_efficiency_score() for node in self.topology_manager.nodes.values()]
            return sum(efficiency_scores) / len(efficiency_scores)
            
        except Exception as e:
            warnings.warn(f"Mesh efficiency calculation failed: {e}")
            return 0.0
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        try:
            topology_stats = self.topology_manager.get_topology_stats()
            
            with self._network_lock:
                active_sessions_count = len(self.active_sessions)
                
            mesh_efficiency = self._calculate_mesh_efficiency()
            
            # Calculate network health
            healthy_nodes = sum(1 for node in self.topology_manager.nodes.values() 
                              if node.state in [NodeState.ACTIVE, NodeState.IDLE, NodeState.PROCESSING])
            network_health = healthy_nodes / max(1, len(self.topology_manager.nodes))
            
            return {
                'network_health': network_health,
                'mesh_efficiency': mesh_efficiency,
                'active_sessions': active_sessions_count,
                'topology_stats': topology_stats,
                'swarm_decisions_made': len(self.swarm_intelligence.decision_history),
                'network_version': '6.0.0-neural-mesh',
                'timestamp': time.time()
            }
            
        except Exception as e:
            warnings.warn(f"Network status failed: {e}")
            return {
                'error': str(e),
                'network_version': '6.0.0-neural-mesh',
                'timestamp': time.time()
            }
    
    async def optimize_network(self):
        """Optimize network topology and performance."""
        try:
            # Optimize topology
            self.topology_manager.optimize_topology()
            
            # Balance loads across nodes
            await self._balance_node_loads()
            
            # Cleanup old sessions
            current_time = time.time()
            with self._network_lock:
                old_sessions = [
                    sid for sid, session in self.active_sessions.items()
                    if current_time - session.get('timestamp', current_time) > 3600  # 1 hour
                ]
                for sid in old_sessions:
                    del self.active_sessions[sid]
            
        except Exception as e:
            warnings.warn(f"Network optimization failed: {e}")
    
    async def _balance_node_loads(self):
        """Balance processing loads across nodes."""
        try:
            # Gradually reduce load on all nodes
            for node in self.topology_manager.nodes.values():
                if node.current_load > 0:
                    node.update_load(max(0, node.current_load - 0.05))  # 5% reduction
                    
            # Update node states based on new loads
            for node in self.topology_manager.nodes.values():
                if node.state == NodeState.OVERLOADED and node.current_load < 0.8:
                    node.state = NodeState.ACTIVE
                elif node.state == NodeState.PROCESSING and node.current_load < 0.1:
                    node.state = NodeState.IDLE
                    
        except Exception as e:
            warnings.warn(f"Load balancing failed: {e}")

# Global neural mesh network instance
_neural_mesh_network = None

def get_neural_mesh_network() -> NeuralMeshNetwork:
    """Get the global neural mesh network instance."""
    global _neural_mesh_network
    if _neural_mesh_network is None:
        _neural_mesh_network = NeuralMeshNetwork()
    return _neural_mesh_network

# Demo and testing functions
def demonstrate_neural_mesh():
    """Demonstrate neural mesh network capabilities."""
    print("🕸️ Neural Mesh Network Demo")
    print("=" * 50)
    
    network = get_neural_mesh_network()
    
    # Simulate neural signals
    if HAS_NUMPY:
        test_signals = np.random.randn(128, 256)  # 128 channels, 256 timepoints
    else:
        test_signals = [[random.gauss(0, 1) for _ in range(256)] for _ in range(128)]
    
    async def demo_session():
        # Process signals through mesh
        result = await network.process_distributed_signal(test_signals)
        print(f"Processing Result: {json.dumps(result, indent=2, default=str)}")
        
        # Get network status
        status = network.get_network_status()
        print(f"\nNetwork Status: {json.dumps(status, indent=2, default=str)}")
        
        # Optimize network
        await network.optimize_network()
        print("\n🔧 Network optimization completed")
    
    # Run demo
    try:
        asyncio.run(demo_session())
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    demonstrate_neural_mesh()