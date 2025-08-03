"""
Agent Orchestrator for Coordinating AI-Powered Development Workflow
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .base_agent import BaseAgent, AgentContext, Task, TaskPriority, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStage:
    """Represents a stage in the development workflow"""
    name: str
    agents: List[str]
    dependencies: Set[str] = field(default_factory=set)
    parallel: bool = False
    timeout_minutes: int = 30
    required_outputs: List[str] = field(default_factory=list)


class AgentOrchestrator:
    """Orchestrates multiple AI agents in the development workflow"""
    
    def __init__(self, context: AgentContext):
        self.context = context
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_stages: List[WorkflowStage] = []
        self.current_stage: Optional[WorkflowStage] = None
        self.stage_index = 0
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Communication hub
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.workflow_state: Dict[str, Any] = {}
        
        # Metrics and monitoring
        self.stage_metrics: List[Dict[str, Any]] = []
        self.global_metrics: Dict[str, Any] = {}
        
        self._setup_default_workflow()
        
    def _setup_default_workflow(self) -> None:
        """Setup the default BCI development workflow"""
        self.workflow_stages = [
            WorkflowStage(
                name="requirements_analysis",
                agents=["requirements"],
                timeout_minutes=15,
                required_outputs=["requirements_spec", "user_stories"]
            ),
            WorkflowStage(
                name="architecture_design", 
                agents=["architecture"],
                dependencies={"requirements_analysis"},
                timeout_minutes=20,
                required_outputs=["system_architecture", "component_design"]
            ),
            WorkflowStage(
                name="implementation",
                agents=["implementation"],
                dependencies={"architecture_design"},
                timeout_minutes=60,
                required_outputs=["source_code", "api_interfaces"]
            ),
            WorkflowStage(
                name="quality_assurance",
                agents=["testing", "security"],
                dependencies={"implementation"},
                parallel=True,
                timeout_minutes=45,
                required_outputs=["test_results", "security_report"]
            ),
            WorkflowStage(
                name="optimization",
                agents=["performance"],
                dependencies={"quality_assurance"},
                timeout_minutes=30,
                required_outputs=["performance_report", "optimization_recommendations"]
            ),
            WorkflowStage(
                name="documentation",
                agents=["documentation"],
                dependencies={"optimization"},
                timeout_minutes=25,
                required_outputs=["api_docs", "user_guide", "deployment_guide"]
            ),
            WorkflowStage(
                name="deployment",
                agents=["deployment"],
                dependencies={"documentation"},
                timeout_minutes=40,
                required_outputs=["deployment_config", "ci_cd_pipeline"]
            )
        ]
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def create_custom_workflow(self, stages: List[WorkflowStage]) -> None:
        """Create a custom workflow with specified stages"""
        self.workflow_stages = stages
        self.stage_index = 0
        logger.info(f"Created custom workflow with {len(stages)} stages")
    
    async def start_workflow(
        self, 
        initial_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start the complete development workflow"""
        self.started_at = datetime.now()
        self.workflow_state = initial_requirements or {}
        
        logger.info("Starting AI-powered development workflow")
        
        try:
            # Start all agent task processors
            agent_tasks = [
                asyncio.create_task(agent.process_tasks())
                for agent in self.agents.values()
            ]
            
            # Start message broker
            message_task = asyncio.create_task(self._message_broker())
            
            # Execute workflow stages
            workflow_task = asyncio.create_task(self._execute_workflow())
            
            # Wait for workflow completion
            await workflow_task
            
            # Cancel agent tasks
            for task in agent_tasks:
                task.cancel()
            message_task.cancel()
            
            self.completed_at = datetime.now()
            
            return await self._generate_workflow_report()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_workflow(self) -> None:
        """Execute all workflow stages in sequence"""
        for self.stage_index, stage in enumerate(self.workflow_stages):
            self.current_stage = stage
            logger.info(f"Starting stage: {stage.name}")
            
            stage_start = datetime.now()
            
            try:
                await self._execute_stage(stage)
                
                stage_duration = (datetime.now() - stage_start).total_seconds()
                self.stage_metrics.append({
                    "stage": stage.name,
                    "duration_seconds": stage_duration,
                    "status": "completed",
                    "agents": stage.agents
                })
                
                logger.info(
                    f"Completed stage: {stage.name} "
                    f"in {stage_duration:.1f} seconds"
                )
                
            except asyncio.TimeoutError:
                logger.error(f"Stage {stage.name} timed out")
                self.stage_metrics.append({
                    "stage": stage.name,
                    "duration_seconds": stage.timeout_minutes * 60,
                    "status": "timeout",
                    "agents": stage.agents
                })
                raise
            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                self.stage_metrics.append({
                    "stage": stage.name,
                    "duration_seconds": (datetime.now() - stage_start).total_seconds(),
                    "status": "error",
                    "error": str(e),
                    "agents": stage.agents
                })
                raise
    
    async def _execute_stage(self, stage: WorkflowStage) -> None:
        """Execute a single workflow stage"""
        # Check dependencies
        if not self._check_stage_dependencies(stage):
            raise RuntimeError(f"Stage {stage.name} dependencies not satisfied")
        
        # Create tasks for agents in this stage
        stage_tasks = []
        for agent_name in stage.agents:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not registered")
            
            agent = self.agents[agent_name]
            task = Task(
                name=f"{stage.name}_{agent_name}",
                description=f"Execute {stage.name} using {agent_name} agent",
                priority=TaskPriority.HIGH,
                metadata={"stage": stage.name, "workflow_state": self.workflow_state}
            )
            
            await agent.add_task(task)
            stage_tasks.append((agent, task))
        
        # Wait for completion or timeout
        timeout = timedelta(minutes=stage.timeout_minutes)
        deadline = datetime.now() + timeout
        
        completed_agents = set()
        
        while len(completed_agents) < len(stage.agents):
            if datetime.now() > deadline:
                raise asyncio.TimeoutError(f"Stage {stage.name} timed out")
            
            # Check agent completion status
            for agent, task in stage_tasks:
                if agent.name not in completed_agents:
                    if task in agent.completed_tasks:
                        completed_agents.add(agent.name)
                        logger.info(
                            f"Agent {agent.name} completed stage {stage.name}"
                        )
                        
                        # Update workflow state with results
                        if task.result:
                            self.workflow_state.update(task.result)
            
            await asyncio.sleep(1.0)  # Check every second
        
        # Validate required outputs
        await self._validate_stage_outputs(stage)
    
    def _check_stage_dependencies(self, stage: WorkflowStage) -> bool:
        """Check if stage dependencies are satisfied"""
        completed_stages = {
            metric["stage"] for metric in self.stage_metrics 
            if metric["status"] == "completed"
        }
        return stage.dependencies.issubset(completed_stages)
    
    async def _validate_stage_outputs(self, stage: WorkflowStage) -> None:
        """Validate that stage produced required outputs"""
        missing_outputs = []
        
        for required_output in stage.required_outputs:
            if required_output not in self.workflow_state:
                missing_outputs.append(required_output)
        
        if missing_outputs:
            raise ValueError(
                f"Stage {stage.name} missing required outputs: {missing_outputs}"
            )
    
    async def _message_broker(self) -> None:
        """Broker messages between agents"""
        while True:
            try:
                # Collect messages from all agents
                for agent in self.agents.values():
                    try:
                        message = agent.output_queue.get_nowait()
                        await self._route_message(message)
                    except asyncio.QueueEmpty:
                        continue
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Message broker error: {e}")
                await asyncio.sleep(1.0)
    
    async def _route_message(self, message: Dict[str, Any]) -> None:
        """Route message to appropriate recipient"""
        if "to" in message and message["to"] in self.agents:
            target_agent = self.agents[message["to"]]
            await target_agent.input_queue.put(message)
        else:
            # Broadcast or handle workflow-level message
            await self.message_queue.put(message)
    
    async def _generate_workflow_report(self) -> Dict[str, Any]:
        """Generate comprehensive workflow execution report"""
        total_duration = None
        if self.started_at and self.completed_at:
            total_duration = (self.completed_at - self.started_at).total_seconds()
        
        # Collect agent metrics
        agent_metrics = {
            name: agent.get_metrics() 
            for name, agent in self.agents.items()
        }
        
        # Calculate success rate
        completed_stages = sum(
            1 for metric in self.stage_metrics 
            if metric["status"] == "completed"
        )
        success_rate = completed_stages / len(self.workflow_stages) if self.workflow_stages else 0
        
        return {
            "workflow_summary": {
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "total_duration_seconds": total_duration,
                "total_stages": len(self.workflow_stages),
                "completed_stages": completed_stages,
                "success_rate": success_rate,
                "status": "completed" if success_rate == 1.0 else "failed"
            },
            "stage_metrics": self.stage_metrics,
            "agent_metrics": agent_metrics,
            "workflow_state": self.workflow_state,
            "context": {
                "project_root": self.context.project_root,
                "current_branch": self.context.current_branch
            }
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "current_stage": self.current_stage.name if self.current_stage else None,
            "stage_progress": f"{self.stage_index + 1}/{len(self.workflow_stages)}",
            "agent_status": {
                name: agent.get_status() 
                for name, agent in self.agents.items()
            },
            "workflow_state": self.workflow_state,
            "started_at": self.started_at.isoformat() if self.started_at else None
        }
    
    async def pause_workflow(self) -> None:
        """Pause the current workflow execution"""
        logger.info("Pausing workflow execution")
        # Implementation for pausing agents and workflow
    
    async def resume_workflow(self) -> None:
        """Resume paused workflow execution"""
        logger.info("Resuming workflow execution")
        # Implementation for resuming agents and workflow
    
    async def abort_workflow(self, reason: str) -> None:
        """Abort the current workflow execution"""
        logger.warning(f"Aborting workflow: {reason}")
        # Implementation for graceful workflow termination