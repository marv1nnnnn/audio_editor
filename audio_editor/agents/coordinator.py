"""
Coordinator for the multi-agent audio processing system.
"""
import os
import time
import asyncio
import logfire
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
import inspect

from pydantic_ai.usage import Usage, UsageLimits

import audio_tools
from .models import (
    AudioPlan, AudioInput, ExecutionResult, ToolDefinition,
    PlanStep, StepStatus
)
from .dependencies import (
    AudioProcessingContext, PlannerDependencies, ExecutorDependencies
)
from .planner import planner_agent, PlannerResponse
from .executor import executor_agent, CodeGenerationResponse
from .mcp import MCPCodeExecutor


class AudioProcessingCoordinator:
    """Coordinates the multi-agent audio processing system."""
    
    def __init__(self, working_dir: str, model_name: str = "gemini-2.0-flash"):
        """Initialize the coordinator.
        
        Args:
            working_dir: Working directory for audio processing
            model_name: Name of the LLM model to use
        """
        self.working_dir = os.path.abspath(working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.context = AudioProcessingContext.create(
            workspace_dir=Path(self.working_dir),
            model_name=model_name
        )
        
        self.tool_definitions = self._create_tool_definitions()
        self.mcp = MCPCodeExecutor(self.working_dir)
        self.usage = Usage()
        self.usage_limits = UsageLimits(request_limit=25)
        
        logfire.info(f"AudioProcessingCoordinator initialized with {len(self.tool_definitions)} tools.")
    
    def _create_tool_definitions(self) -> List[ToolDefinition]:
        """Create tool definitions from available audio tools."""
        definitions = []
        
        for name, func in self.context.available_tools.items():
            doc = inspect.getdoc(func) or "No description available."
            # Extract first line of docstring if multiline
            first_line_doc = doc.splitlines()[0].strip()
            
            try:
                sig = str(inspect.signature(func))
                # Clean signature slightly for readability
                sig = sig.replace("NoneType", "None")
            except ValueError:
                sig = "(...)"  # Fallback
                
            definitions.append(ToolDefinition(
                name=name,
                description=first_line_doc,
                signature=sig,
                docstring=doc
            ))
            
        return definitions
    
    async def process_audio(
        self, 
        task_description: str, 
        audio_file_path: str,
        audio_transcript: str = ""
    ) -> str:
        """Process audio using the multi-agent system.
        
        Args:
            task_description: Description of the task to perform
            audio_file_path: Path to the audio file to process
            audio_transcript: Transcript of the audio file (if any)
            
        Returns:
            Path to the final processed audio file
        """
        with logfire.span("process_audio", task=task_description, file=audio_file_path):
            start_time = time.time()
            
            # Copy input file to working directory
            input_basename = os.path.basename(audio_file_path)
            working_input = os.path.join(self.working_dir, input_basename)
            
            # Copy file if not already in working directory
            if os.path.abspath(audio_file_path) != os.path.abspath(working_input):
                import shutil
                shutil.copy(audio_file_path, working_input)
                logfire.info(f"Copied {audio_file_path} to {working_input}")
            
            # Create initial audio input
            audio_input = AudioInput(
                transcript=audio_transcript or f"Audio file: {input_basename}",
                timestamp=time.time()
            )
            
            # Initialize the planner dependencies
            planner_deps = PlannerDependencies(
                context=self.context,
                task_description=task_description,
                tool_definitions=self.tool_definitions,
                audio_input=audio_input
            )
            
            # Generate initial plan
            planner_result = await planner_agent.run(
                f"Create an initial plan for the task: {task_description}",
                deps=planner_deps,
                usage=self.usage,
                usage_limits=self.usage_limits
            )
            
            plan = await self._call_planner_tool(
                "generate_initial_plan",
                {
                    "task_description": task_description,
                    "current_audio_path": working_input
                },
                planner_result
            )
            
            # Process until complete
            max_iterations = 25
            iteration = 0
            
            while iteration < max_iterations and not plan.is_complete:
                iteration += 1
                logfire.info(f"Starting iteration {iteration}/{max_iterations}")
                
                # Find next step
                next_step_index = self._find_next_step(plan)
                if next_step_index is None:
                    logfire.warning("No next step found, marking plan as complete")
                    plan.is_complete = True
                    break
                
                # Execute the step using the executor and MCP
                plan, execution_result = await self._execute_step(plan, next_step_index)
                
                # Update the plan based on execution result
                planner_deps = PlannerDependencies(
                    context=self.context,
                    task_description=task_description,
                    tool_definitions=self.tool_definitions,
                    audio_input=audio_input,
                    current_plan=plan,
                    execution_result=execution_result
                )
                
                planner_result = await planner_agent.run(
                    f"Update the plan after executing step {next_step_index + 1}",
                    deps=planner_deps,
                    usage=self.usage,
                    usage_limits=self.usage_limits
                )
                
                planner_response = await self._call_planner_tool(
                    "update_plan_after_execution",
                    {
                        "plan": plan,
                        "step_index": next_step_index,
                        "execution_result": execution_result.model_dump()
                    },
                    planner_result
                )
                
                # Update plan from planner response
                plan = planner_response.updated_plan
                
                # Set a checkpoint if specified
                if planner_response.checkpoint_index is not None:
                    if planner_response.checkpoint_index not in plan.checkpoint_indices:
                        plan.checkpoint_indices.append(planner_response.checkpoint_index)
                        logfire.info(f"Set checkpoint at step {planner_response.checkpoint_index + 1}")
                
                # Handle replanning if needed
                if planner_response.replanning_needed and plan.checkpoint_indices:
                    last_checkpoint = max(plan.checkpoint_indices)
                    
                    planner_result = await planner_agent.run(
                        f"Replan from checkpoint at step {last_checkpoint + 1}",
                        deps=planner_deps,
                        usage=self.usage,
                        usage_limits=self.usage_limits
                    )
                    
                    planner_response = await self._call_planner_tool(
                        "replan_from_checkpoint",
                        {
                            "plan": plan,
                            "checkpoint_index": last_checkpoint
                        },
                        planner_result
                    )
                    
                    plan = planner_response.updated_plan
                    logfire.info(f"Replanned from checkpoint at step {last_checkpoint + 1}")
            
            # Get the final result path
            final_result_path = self._get_final_result_path(plan)
            
            logfire.info(
                f"Audio processing completed in {time.time() - start_time:.2f}s, "
                f"{iteration} iterations, result: {final_result_path}"
            )
            
            return final_result_path
    
    async def _call_planner_tool(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any],
        planner_result: Any
    ) -> Any:
        """Call a planner tool with arguments and return its result."""
        with logfire.span(f"call_planner_tool_{tool_name}"):
            for message in planner_result.all_messages():
                for part in message.parts:
                    if hasattr(part, "tool_name") and part.tool_name == tool_name:
                        return planner_result.data
            
            raise ValueError(f"Planner did not call expected tool: {tool_name}")
    
    def _find_next_step(self, plan: AudioPlan) -> Optional[int]:
        """Find the index of the next step to execute."""
        for i, step in enumerate(plan.steps):
            if step.status == StepStatus.PENDING and i not in plan.completed_step_indices:
                return i
        return None
    
    async def _execute_step(
        self, 
        plan: AudioPlan, 
        step_index: int
    ) -> Tuple[AudioPlan, ExecutionResult]:
        """Execute a step in the plan.
        
        Args:
            plan: The current plan
            step_index: Index of the step to execute
            
        Returns:
            Tuple of (updated_plan, execution_result)
        """
        with logfire.span("execute_step", step_index=step_index):
            updated_plan = plan.model_copy(deep=True)
            step = updated_plan.steps[step_index]
            
            # Initialize executor dependencies
            executor_deps = ExecutorDependencies(
                context=self.context,
                tool_definitions=self.tool_definitions,
                plan_step_index=step_index,
                plan=updated_plan
            )
            
            # Generate code
            executor_result = await executor_agent.run(
                f"Generate code for step {step_index + 1}: {step.description}",
                deps=executor_deps,
                usage=self.usage,
                usage_limits=self.usage_limits
            )
            
            generated_code = ""
            for message in executor_result.all_messages():
                for part in message.parts:
                    if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                        generated_code = executor_result.data.generated_code
                        break
            
            if not generated_code:
                logfire.error("Executor did not generate code")
                return updated_plan, ExecutionResult(
                    status="FAILURE",
                    error_message="Failed to generate code for step",
                    duration=0.0
                )
            
            # Save the generated code to the step
            step.code = generated_code
            
            # Execute the code using MCP
            execution_result = await self.mcp.execute_code(generated_code)
            
            # Handle execution failure with retries
            retry_count = 0
            max_retries = 2
            
            while (
                execution_result.status == "FAILURE" and 
                retry_count < max_retries
            ):
                retry_count += 1
                logfire.warning(
                    f"Step {step_index + 1} failed, retrying ({retry_count}/{max_retries}): "
                    f"{execution_result.error_message}"
                )
                
                # Update executor dependencies with error info
                executor_deps.execution_result = execution_result
                
                # Request code refinement
                executor_result = await executor_agent.run(
                    f"Refine code for step {step_index + 1} after error: {execution_result.error_message}",
                    deps=executor_deps,
                    usage=self.usage,
                    usage_limits=self.usage_limits
                )
                
                refined_code = ""
                for message in executor_result.all_messages():
                    for part in message.parts:
                        if hasattr(part, "tool_name") and part.tool_name == "refine_code_after_error":
                            refined_code = executor_result.data.generated_code
                            break
                
                if not refined_code:
                    logfire.error("Executor did not generate refined code")
                    break
                
                # Save the refined code
                step.code = refined_code
                
                # Try executing the refined code
                execution_result = await self.mcp.execute_code(refined_code)
            
            # Update the step status based on final execution
            if execution_result.status == "SUCCESS":
                step.status = StepStatus.DONE
                
                # Update current audio path if output was generated
                if execution_result.output_path:
                    updated_plan.current_audio_path = execution_result.output_path
            else:
                step.status = StepStatus.FAILED
            
            return updated_plan, execution_result
    
    def _get_final_result_path(self, plan: AudioPlan) -> str:
        """Get the path to the final result audio file."""
        # Return current audio path as the final result
        return str(plan.current_audio_path) 