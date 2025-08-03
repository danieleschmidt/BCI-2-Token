"""
Base Agent for AI-Powered Development Coordination
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a development task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AgentContext:
    """Shared context between agents"""
    project_root: str
    current_branch: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    code_quality_metrics: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs: Any) -> None:
        """Update context with new data"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BaseAgent(ABC):
    """Base class for all development agents"""
    
    def __init__(
        self,
        name: str,
        description: str,
        context: AgentContext,
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.context = context
        self.dependencies = dependencies or []
        self.status = AgentStatus.IDLE
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.error_count = 0
        self.max_retries = 3
        
        # Communication channels
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info(f"Initialized agent: {self.name}")
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data and generate insights"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task"""
        pass
    
    @abstractmethod
    async def validate_output(self, output: Any) -> bool:
        """Validate the output of a task"""
        pass
    
    async def add_task(self, task: Task) -> None:
        """Add a task to the agent's queue"""
        self.tasks.append(task)
        await self.input_queue.put(task)
        logger.info(f"Agent {self.name} received task: {task.name}")
    
    async def process_tasks(self) -> None:
        """Main task processing loop"""
        while True:
            try:
                # Wait for task or timeout
                task = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )
                
                await self._execute_task_with_retry(task)
                
            except asyncio.TimeoutError:
                # No tasks, continue
                continue
            except Exception as e:
                logger.error(f"Agent {self.name} error: {e}")
                self.status = AgentStatus.ERROR
    
    async def _execute_task_with_retry(self, task: Task) -> None:
        """Execute task with retry logic"""
        self.status = AgentStatus.WORKING
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Check dependencies
                if not await self._check_dependencies(task):
                    self.status = AgentStatus.BLOCKED
                    await asyncio.sleep(1.0)  # Wait and retry
                    continue
                
                # Execute the task
                result = await self.execute_task(task)
                
                # Validate output
                if await self.validate_output(result):
                    task.result = result
                    task.completed_at = datetime.now()
                    self.completed_tasks.append(task)
                    self.tasks.remove(task)
                    
                    # Notify other agents
                    await self.output_queue.put({
                        'agent': self.name,
                        'task': task,
                        'result': result
                    })
                    
                    self.status = AgentStatus.COMPLETED
                    logger.info(f"Agent {self.name} completed task: {task.name}")
                    return
                else:
                    raise ValueError("Output validation failed")
                    
            except Exception as e:
                retry_count += 1
                task.error = str(e)
                logger.warning(
                    f"Agent {self.name} task {task.name} failed "
                    f"(attempt {retry_count}): {e}"
                )
                
                if retry_count > self.max_retries:
                    self.error_count += 1
                    self.status = AgentStatus.ERROR
                    logger.error(
                        f"Agent {self.name} failed task {task.name} "
                        f"after {self.max_retries} attempts"
                    )
                    return
                
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        completed_task_ids = {t.id for t in self.completed_tasks}
        return task.dependencies.issubset(completed_task_ids)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "status": self.status.value,
            "pending_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "error_count": self.error_count,
            "description": self.description
        }
    
    async def communicate_with_agent(
        self, 
        target_agent: str, 
        message: Dict[str, Any]
    ) -> None:
        """Send message to another agent"""
        await self.output_queue.put({
            'from': self.name,
            'to': target_agent,
            'message': message
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        total_tasks = len(self.completed_tasks) + len(self.tasks)
        completion_rate = (
            len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0
        )
        
        avg_completion_time = None
        if self.completed_tasks:
            completion_times = [
                (task.completed_at - task.created_at).total_seconds()
                for task in self.completed_tasks
                if task.completed_at and task.created_at
            ]
            if completion_times:
                avg_completion_time = sum(completion_times) / len(completion_times)
        
        return {
            "agent": self.name,
            "total_tasks": total_tasks,
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len(self.tasks),
            "completion_rate": completion_rate,
            "error_count": self.error_count,
            "avg_completion_time": avg_completion_time,
            "status": self.status.value
        }