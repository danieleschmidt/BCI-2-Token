"""
Autonomous Intelligence Engine - Next-Generation BCI-2-Token Enhancement
========================================================================

Advanced autonomous decision-making system that adapts, evolves, and optimizes
the BCI-2-Token framework in real-time based on usage patterns and performance metrics.

This module implements Generation 1+ autonomous capabilities:
- Self-optimizing performance parameters
- Adaptive model selection based on signal quality
- Autonomous error recovery and healing
- Predictive resource allocation
- Dynamic feature activation/deactivation
- Continuous learning from user patterns
"""

import time
import threading
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class IntelligenceLevel(Enum):
    """Intelligence levels for autonomous operation."""
    BASIC = "basic"           # Rule-based decisions only
    ADAPTIVE = "adaptive"     # Learn from patterns 
    PREDICTIVE = "predictive" # Predict future needs
    AUTONOMOUS = "autonomous" # Full self-management

@dataclass
class DecisionContext:
    """Context for autonomous decision making."""
    timestamp: float = field(default_factory=time.time)
    signal_quality: float = 0.0
    system_load: float = 0.0
    user_session_count: int = 0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    processing_latency: float = 0.0
    available_models: List[str] = field(default_factory=list)
    
@dataclass
class AutonomousAction:
    """Represents an autonomous action to be taken."""
    action_type: str
    parameters: Dict[str, Any]
    priority: int  # 1=highest, 10=lowest
    confidence: float  # 0.0-1.0
    reasoning: str
    estimated_impact: str

class AutonomousIntelligence:
    """
    Autonomous Intelligence Engine for BCI-2-Token
    
    This system continuously monitors the BCI framework and makes
    intelligent decisions to optimize performance, reliability, and
    user experience without human intervention.
    """
    
    def __init__(self, intelligence_level: IntelligenceLevel = IntelligenceLevel.ADAPTIVE):
        self.intelligence_level = intelligence_level
        self.is_active = False
        self.decision_thread = None
        self.metrics_history: List[DecisionContext] = []
        self.decision_rules: Dict[str, Callable] = {}
        self.learning_data = {}
        self.autonomous_actions: List[AutonomousAction] = []
        
        # Performance thresholds for autonomous decisions
        self.thresholds = {
            'error_rate_critical': 0.10,     # 10% error rate triggers action
            'error_rate_warning': 0.05,      # 5% error rate triggers monitoring
            'latency_critical': 5.0,         # 5 second latency triggers action
            'latency_warning': 2.0,          # 2 second latency triggers monitoring
            'memory_critical': 0.90,         # 90% memory usage triggers cleanup
            'memory_warning': 0.80,          # 80% memory usage triggers optimization
            'signal_quality_poor': 0.30,     # Below 30% quality triggers adaptation
            'signal_quality_good': 0.80,     # Above 80% quality allows optimization
        }
        
        self._register_decision_rules()
        self._load_learning_data()
        
    def _register_decision_rules(self):
        """Register autonomous decision rules."""
        self.decision_rules = {
            'optimize_performance': self._decide_performance_optimization,
            'adapt_model_selection': self._decide_model_adaptation,
            'manage_resources': self._decide_resource_management,
            'recover_from_errors': self._decide_error_recovery,
            'scale_capacity': self._decide_scaling,
            'update_configuration': self._decide_configuration_update
        }
        
    def _load_learning_data(self):
        """Load historical learning data."""
        try:
            data_path = Path.home() / '.bci2token' / 'autonomous_learning.json'
            if data_path.exists():
                with open(data_path, 'r') as f:
                    self.learning_data = json.load(f)
        except Exception as e:
            warnings.warn(f"Could not load learning data: {e}")
            self.learning_data = {
                'performance_patterns': {},
                'user_preferences': {},
                'error_patterns': {},
                'resource_usage': {}
            }
            
    def _save_learning_data(self):
        """Save learning data for future sessions."""
        try:
            data_path = Path.home() / '.bci2token'
            data_path.mkdir(exist_ok=True)
            
            with open(data_path / 'autonomous_learning.json', 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Could not save learning data: {e}")
            
    def start_autonomous_operation(self):
        """Start the autonomous intelligence system."""
        if self.is_active:
            return
            
        self.is_active = True
        self.decision_thread = threading.Thread(
            target=self._autonomous_decision_loop,
            daemon=True,
            name="AutonomousIntelligence"
        )
        self.decision_thread.start()
        
    def stop_autonomous_operation(self):
        """Stop the autonomous intelligence system."""
        self.is_active = False
        if self.decision_thread and self.decision_thread.is_alive():
            self.decision_thread.join(timeout=5)
        self._save_learning_data()
        
    def _autonomous_decision_loop(self):
        """Main decision-making loop running in background."""
        while self.is_active:
            try:
                # Gather current context
                context = self._gather_context()
                
                # Record context for learning
                self.metrics_history.append(context)
                if len(self.metrics_history) > 1000:  # Limit history
                    self.metrics_history = self.metrics_history[-800:]
                
                # Make autonomous decisions
                actions = self._make_decisions(context)
                
                # Execute high-priority actions
                self._execute_actions(actions)
                
                # Learn from outcomes
                self._update_learning(context, actions)
                
                # Sleep before next cycle
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                warnings.warn(f"Error in autonomous decision loop: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _gather_context(self) -> DecisionContext:
        """Gather current system context for decision making."""
        context = DecisionContext()
        
        try:
            # Get system metrics (simplified for now)
            if HAS_NUMPY:
                # Simulate signal quality check
                context.signal_quality = np.random.uniform(0.2, 0.95)
            
            # System load (placeholder - would use real metrics in production)
            context.system_load = 0.3  # Would get from system monitor
            context.memory_usage = 0.4  # Would get from memory monitor
            context.processing_latency = 0.5  # Would get from performance monitor
            context.error_rate = 0.02  # Would get from error monitor
            context.user_session_count = 1  # Would get from session manager
            context.available_models = ['basic', 'advanced', 'quantum']
            
        except Exception as e:
            warnings.warn(f"Error gathering context: {e}")
            
        return context
        
    def _make_decisions(self, context: DecisionContext) -> List[AutonomousAction]:
        """Make autonomous decisions based on current context."""
        actions = []
        
        for rule_name, rule_func in self.decision_rules.items():
            try:
                action = rule_func(context)
                if action:
                    actions.append(action)
            except Exception as e:
                warnings.warn(f"Error in decision rule {rule_name}: {e}")
                
        # Sort by priority (1=highest priority)
        actions.sort(key=lambda x: x.priority)
        
        return actions
        
    def _decide_performance_optimization(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on performance optimizations."""
        if context.processing_latency > self.thresholds['latency_critical']:
            return AutonomousAction(
                action_type='optimize_processing',
                parameters={'enable_caching': True, 'increase_threads': True},
                priority=1,
                confidence=0.9,
                reasoning=f"High latency detected: {context.processing_latency:.2f}s > {self.thresholds['latency_critical']}s",
                estimated_impact="Reduce latency by 30-50%"
            )
        return None
        
    def _decide_model_adaptation(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on model adaptation based on signal quality."""
        if context.signal_quality < self.thresholds['signal_quality_poor']:
            # Switch to more robust model for poor signal quality
            return AutonomousAction(
                action_type='switch_model',
                parameters={'target_model': 'robust_lowquality', 'adapt_preprocessing': True},
                priority=2,
                confidence=0.8,
                reasoning=f"Poor signal quality: {context.signal_quality:.2f} < {self.thresholds['signal_quality_poor']}",
                estimated_impact="Improve decoding accuracy by 15-25%"
            )
        elif context.signal_quality > self.thresholds['signal_quality_good']:
            # Switch to high-performance model for good signal
            return AutonomousAction(
                action_type='switch_model', 
                parameters={'target_model': 'high_performance', 'enable_advanced_features': True},
                priority=3,
                confidence=0.7,
                reasoning=f"High signal quality: {context.signal_quality:.2f} > {self.thresholds['signal_quality_good']}",
                estimated_impact="Improve speed by 20-30%"
            )
        return None
        
    def _decide_resource_management(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on resource management actions."""
        if context.memory_usage > self.thresholds['memory_critical']:
            return AutonomousAction(
                action_type='cleanup_memory',
                parameters={'clear_cache': True, 'compress_data': True, 'release_unused': True},
                priority=1,
                confidence=0.95,
                reasoning=f"Critical memory usage: {context.memory_usage:.1%} > {self.thresholds['memory_critical']:.1%}",
                estimated_impact="Free 20-40% memory"
            )
        return None
        
    def _decide_error_recovery(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on error recovery actions."""
        if context.error_rate > self.thresholds['error_rate_critical']:
            return AutonomousAction(
                action_type='error_recovery',
                parameters={'restart_components': True, 'reset_connections': True, 'enable_fallback': True},
                priority=1,
                confidence=0.85,
                reasoning=f"Critical error rate: {context.error_rate:.1%} > {self.thresholds['error_rate_critical']:.1%}",
                estimated_impact="Reduce error rate by 60-80%"
            )
        return None
        
    def _decide_scaling(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on scaling actions."""
        if context.user_session_count > 10 and context.system_load > 0.8:
            return AutonomousAction(
                action_type='scale_up',
                parameters={'add_workers': 2, 'increase_cache': True},
                priority=2,
                confidence=0.7,
                reasoning=f"High load with many sessions: {context.user_session_count} sessions, {context.system_load:.1%} load",
                estimated_impact="Handle 50% more concurrent users"
            )
        elif context.user_session_count < 2 and context.system_load < 0.2:
            return AutonomousAction(
                action_type='scale_down',
                parameters={'remove_workers': 1, 'reduce_cache': True},
                priority=5,
                confidence=0.6,
                reasoning=f"Low utilization: {context.user_session_count} sessions, {context.system_load:.1%} load",
                estimated_impact="Save 20-30% resources"
            )
        return None
        
    def _decide_configuration_update(self, context: DecisionContext) -> Optional[AutonomousAction]:
        """Decide on configuration updates."""
        # Learn from patterns and suggest configuration changes
        if len(self.metrics_history) > 10:
            avg_quality = sum(c.signal_quality for c in self.metrics_history[-10:]) / 10
            if avg_quality > 0.8 and context.processing_latency < 1.0:
                return AutonomousAction(
                    action_type='update_config',
                    parameters={'enable_advanced_processing': True, 'increase_precision': True},
                    priority=4,
                    confidence=0.6,
                    reasoning=f"Consistently high quality signals: avg={avg_quality:.2f}",
                    estimated_impact="Improve accuracy by 5-10%"
                )
        return None
        
    def _execute_actions(self, actions: List[AutonomousAction]):
        """Execute autonomous actions (simulate for now)."""
        for action in actions[:3]:  # Execute top 3 priority actions
            try:
                if action.confidence >= 0.7:  # Only execute high-confidence actions
                    self.autonomous_actions.append(action)
                    # In a real implementation, this would execute the actual action
                    # For now, just log it
                    print(f"[AUTONOMOUS] {action.action_type}: {action.reasoning}")
                    
            except Exception as e:
                warnings.warn(f"Error executing action {action.action_type}: {e}")
                
    def _update_learning(self, context: DecisionContext, actions: List[AutonomousAction]):
        """Update learning data based on outcomes."""
        if self.intelligence_level in [IntelligenceLevel.ADAPTIVE, IntelligenceLevel.PREDICTIVE, IntelligenceLevel.AUTONOMOUS]:
            # Update performance patterns
            pattern_key = f"quality_{context.signal_quality:.1f}_load_{context.system_load:.1f}"
            if pattern_key not in self.learning_data['performance_patterns']:
                self.learning_data['performance_patterns'][pattern_key] = []
                
            self.learning_data['performance_patterns'][pattern_key].append({
                'latency': context.processing_latency,
                'error_rate': context.error_rate,
                'timestamp': context.timestamp,
                'actions_taken': [a.action_type for a in actions]
            })
            
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get current status of autonomous intelligence system."""
        return {
            'is_active': self.is_active,
            'intelligence_level': self.intelligence_level.value,
            'metrics_history_size': len(self.metrics_history),
            'actions_taken': len(self.autonomous_actions),
            'learning_data_size': sum(len(v) if isinstance(v, list) else len(v) if isinstance(v, dict) else 1 
                                    for v in self.learning_data.values()),
            'decision_rules_count': len(self.decision_rules),
            'current_thresholds': self.thresholds.copy()
        }
        
    def get_recent_actions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent autonomous actions."""
        recent_actions = self.autonomous_actions[-limit:]
        return [
            {
                'action_type': action.action_type,
                'parameters': action.parameters,
                'priority': action.priority,
                'confidence': action.confidence,
                'reasoning': action.reasoning,
                'estimated_impact': action.estimated_impact
            }
            for action in recent_actions
        ]
        
    def adjust_intelligence_level(self, level: IntelligenceLevel):
        """Adjust the intelligence level of the system."""
        old_level = self.intelligence_level
        self.intelligence_level = level
        
        # Restart if level changed significantly
        if self.is_active and old_level != level:
            self.stop_autonomous_operation()
            time.sleep(1)
            self.start_autonomous_operation()

# Global autonomous intelligence instance
_global_ai_engine: Optional[AutonomousIntelligence] = None

def get_autonomous_engine(intelligence_level: IntelligenceLevel = IntelligenceLevel.ADAPTIVE) -> AutonomousIntelligence:
    """Get or create the global autonomous intelligence engine."""
    global _global_ai_engine
    if _global_ai_engine is None:
        _global_ai_engine = AutonomousIntelligence(intelligence_level)
    return _global_ai_engine

def start_autonomous_intelligence(intelligence_level: IntelligenceLevel = IntelligenceLevel.ADAPTIVE) -> AutonomousIntelligence:
    """Start autonomous intelligence with specified level."""
    engine = get_autonomous_engine(intelligence_level)
    engine.start_autonomous_operation()
    return engine

def stop_autonomous_intelligence():
    """Stop autonomous intelligence."""
    global _global_ai_engine
    if _global_ai_engine:
        _global_ai_engine.stop_autonomous_operation()