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
import json

from pydantic_ai.usage import Usage, UsageLimits

from audio_editor import audio_tools
from .models import (
    AudioPlan, AudioInput, ExecutionResult, ToolDefinition,
    PlanStep, StepStatus, CritiqueResult, QAResult, ErrorAnalysisResult,
    UserFeedbackRequest, UserFeedbackResponse
)
from .dependencies import (
    AudioProcessingContext, PlannerDependencies, ExecutorDependencies,
    CritiqueAgentDependencies, QAAgentDependencies, ErrorAnalysisDependencies
)
from .planner import planner_agent, PlannerResponse
from .executor import executor_agent, CodeGenerationResponse
from .critique_agent import critique_agent, CritiqueResponse
from .qa_agent import qa_agent, QAResponse
from .error_analyzer import error_analyzer_agent, ErrorAnalysisResponse
from .user_feedback import ConsoleUserFeedbackHandler
from .mcp import MCPCodeExecutor


class AudioProcessingCoordinator:
    """Coordinates the multi-agent audio processing system."""
    
    def __init__(
        self, 
        working_dir: str, 
        model_name: str = "gemini-2.0-flash",
        interactive: bool = True,
        enable_critique: bool = True,
        enable_qa: bool = True
    ):
        """Initialize the coordinator.
        
        Args:
            working_dir: Working directory for audio processing
            model_name: Name of the LLM model to use
            interactive: Whether to enable interactive user feedback
            enable_critique: Whether to enable the Critique Agent
            enable_qa: Whether to enable the QA Agent
        """
        self.working_dir = os.path.abspath(working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Create user feedback handler
        self.feedback_handler = ConsoleUserFeedbackHandler(interactive=interactive)
        
        # Create context first, then set feedback handler
        self.context = AudioProcessingContext.create(
            workspace_dir=Path(self.working_dir),
            model_name=model_name
        )
        self.context.user_feedback_handler = self.feedback_handler
        
        self.tool_definitions = self._create_tool_definitions()
        self.mcp = MCPCodeExecutor(self.working_dir)
        self.usage = Usage()
        self.usage_limits = UsageLimits(request_limit=25)
        
        # Feature flags
        self.enable_critique = enable_critique
        self.enable_qa = enable_qa
        self.interactive = interactive
        
        logfire.info(f"AudioProcessingCoordinator initialized with {len(self.tool_definitions)} tools.")
        logfire.info(f"Feature flags: critique={enable_critique}, qa={enable_qa}, interactive={interactive}")
    
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
            planner_deps = PlannerDependencies.from_models(
                context=self.context,
                task_description=task_description,
                tool_definitions=self.tool_definitions,
                audio_input=audio_input
            )

            # --- Debug Log Start ---
            try:
                logfire.debug(f"PlannerDependencies before initial run:\n{json.dumps(planner_deps.model_dump(), indent=2, default=str)}")
            except Exception as e:
                logfire.error(f"Failed to serialize planner_deps for logging: {e}")
            # --- Debug Log End ---

            # Generate initial plan
            try:
                planner_result = await planner_agent.run(
                    f"Create an initial plan for the task: {task_description}",
                    deps=planner_deps,
                    usage=self.usage,
                    usage_limits=self.usage_limits
                )
            except Exception as e:
                logfire.error(f"Audio processing failed during initial planner run: {e}", exc_info=True)
                raise  # Re-raise the exception after logging
            
            plan = await self._call_planner_tool(
                "generate_initial_plan",
                {
                    "task_description": task_description,
                    "current_audio_path": working_input
                },
                planner_result
            )
            
            # If critique is enabled, review the initial plan
            if self.enable_critique:
                plan = await self._critique_plan(plan, task_description)
            
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
                
                # If this is the final step and QA is enabled, perform quality assessment
                if plan.is_complete and self.enable_qa:
                    final_output_path = self._get_final_result_path(plan)
                    if os.path.exists(final_output_path):
                        qa_result = await self._perform_qa(
                            task_description,
                            working_input,
                            final_output_path,
                            plan,
                            execution_result
                        )
                        
                        # If the output doesn't meet requirements and we're interactive,
                        # ask the user if they want to replan
                        if not qa_result.meets_requirements and self.interactive:
                            should_replan = self.feedback_handler.feedback_manager.request_confirmation(
                                "The processed audio does not meet quality standards. Would you like to replan?",
                                f"Issues: {', '.join(qa_result.issues)}",
                                severity="warning"
                            )
                            
                            if should_replan and plan.checkpoint_indices:
                                last_checkpoint = max(plan.checkpoint_indices)
                                planner_deps.current_plan = plan
                                
                                planner_result = await planner_agent.run(
                                    f"Replan from checkpoint at step {last_checkpoint + 1} based on QA feedback",
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
                                plan.is_complete = False
                                logfire.info(f"Replanned from checkpoint at step {last_checkpoint + 1} based on QA feedback")
                                continue
                
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
                        "execution_result": execution_result
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
        """Execute a step in the plan."""
        with logfire.span("execute_step", step_index=step_index):
            step = plan.steps[step_index]
            logfire.info(f"Executing step {step_index + 1}: {step.description}")
            
            # Create executor dependencies
            executor_deps = ExecutorDependencies(
                context=self.context,
                plan=plan,
                plan_step_index=step_index,
                tool_definitions=self.tool_definitions
            )
            
            # Generate code for the step
            executor_result = await executor_agent.run(
                f"Generate code for step {step_index + 1}: {step.description}",
                deps=executor_deps,
                usage=self.usage,
                usage_limits=self.usage_limits
            )
            
            # Get the generated code
            code_response = None
            for message in executor_result.all_messages():
                for part in message.parts:
                    if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                        code_response = executor_result.data
                        break
                if code_response:
                    break
                    
            if not code_response:
                raise ValueError("Executor did not generate code for the step")
                
            generated_code = code_response.generated_code
            
            # If critique is enabled, review the generated code
            if self.enable_critique:
                critique_deps = CritiqueAgentDependencies(
                    context=self.context,
                    tool_definitions=self.tool_definitions,
                    plan=plan,
                    plan_step_index=step_index,
                    generated_code=generated_code,
                    critique_type="code",
                    task_description=plan.task_description
                )
                
                critique_result = await critique_agent.run(
                    f"Critique code for step {step_index + 1}: {step.description}",
                    deps=critique_deps,
                    usage=self.usage,
                    usage_limits=self.usage_limits
                )
                
                critique_response = None
                for message in critique_result.all_messages():
                    for part in message.parts:
                        if hasattr(part, "tool_name") and part.tool_name == "critique_code":
                            critique_response = critique_result.data
                            break
                    if critique_response:
                        break
                        
                if critique_response and not critique_response.is_approved:
                    # Log the critique reasoning
                    logfire.info(f"Code critique: {critique_response.reasoning}")
                    
                    # Use the improved version if available
                    if critique_response.improved_version:
                        generated_code = critique_response.improved_version
                        logfire.info("Using improved code from critique")
            
            # Try to execute the code with retries if needed
            max_retries = 2
            retry_count = 0
            execution_result = None
            
            while retry_count <= max_retries:
                # Execute the code
                execution_result = await self.mcp.execute_code(
                    generated_code, 
                    step.description,
                    {
                        "PLAN": plan,
                        "STEP_INDEX": step_index,
                        "AUDIO_PATH": str(plan.current_audio_path)
                    }
                )
                
                # If execution succeeded, we're done
                if execution_result.status == "SUCCESS":
                    break
                    
                # Otherwise, attempt to analyze and fix the error
                if retry_count < max_retries:
                    retry_count += 1
                    logfire.warning(f"Execution failed, attempt {retry_count}/{max_retries} to fix")
                    
                    # Analyze the error
                    error_deps = ErrorAnalysisDependencies(
                        context=self.context,
                        execution_result=execution_result,
                        plan=plan,
                        plan_step_index=step_index,
                        generated_code=generated_code,
                        tool_definitions=self.tool_definitions
                    )
                    
                    error_result = await error_analyzer_agent.run(
                        f"Analyze error in step {step_index + 1}: {execution_result.error_message}",
                        deps=error_deps,
                        usage=self.usage,
                        usage_limits=self.usage_limits
                    )
                    
                    error_response = None
                    for message in error_result.all_messages():
                        for part in message.parts:
                            if hasattr(part, "tool_name") and part.tool_name == "analyze_error":
                                error_response = error_result.data
                                break
                        if error_response:
                            break
                            
                    if error_response:
                        logfire.info(f"Error analysis: {error_response.root_cause}")
                        
                        # Check if replanning is needed
                        if error_response.requires_replanning:
                            # Mark this step as failed and let the planner handle it
                            break
                            
                        # If we have fix suggestions, apply them
                        if error_response.code_fixes:
                            generated_code = error_response.code_fixes
                            logfire.info("Using fixed code from error analysis")
                        else:
                            # Try to generate a fix with the executor
                            executor_deps.execution_result = execution_result
                            executor_deps.error_analysis = error_response
                            
                            executor_result = await executor_agent.run(
                                f"Fix code for step {step_index + 1} after error: {error_response.error_type}",
                                deps=executor_deps,
                                usage=self.usage,
                                usage_limits=self.usage_limits
                            )
                            
                            code_response = None
                            for message in executor_result.all_messages():
                                for part in message.parts:
                                    if hasattr(part, "tool_name") and part.tool_name == "refine_code_after_error":
                                        code_response = executor_result.data
                                        break
                                if code_response:
                                    break
                                    
                            if code_response:
                                generated_code = code_response.generated_code
                                logfire.info("Using fixed code from executor")
                                
                        # If interactive mode is enabled, show the error and ask for confirmation
                        if self.interactive:
                            # Truncate error message if it's too long
                            error_msg = execution_result.error_message
                            if len(error_msg) > 500:
                                error_msg = error_msg[:500] + "... [truncated]"
                                
                            should_continue = self.feedback_handler.feedback_manager.request_confirmation(
                                "An error occurred during execution. Should we try the fixes?",
                                f"Error: {error_msg}\n\nFixed code ready to try.",
                                severity="warning"
                            )
                            
                            if not should_continue:
                                # User doesn't want to continue with automatic fixes
                                # Ask if they want to provide custom code
                                should_provide_code = self.feedback_handler.feedback_manager.request_confirmation(
                                    "Would you like to provide custom code for this step?",
                                    "If yes, you'll be prompted to enter your code.",
                                    severity="info"
                                )
                                
                                if should_provide_code:
                                    # Show the current code and error
                                    print("\nCurrent code:")
                                    print("-------------")
                                    print(generated_code)
                                    print("\nError message:")
                                    print("-------------")
                                    print(execution_result.error_message)
                                    
                                    # Let the user know they can enter multiline input
                                    print("\nEnter your custom code (type 'END' on a line by itself when done):")
                                    
                                    # Get the user's code
                                    lines = []
                                    while True:
                                        line = input()
                                        if line.strip() == "END":
                                            break
                                        lines.append(line)
                                        
                                    if lines:
                                        generated_code = "\n".join(lines)
                                        logfire.info("Using user-provided code")
                                    else:
                                        logfire.info("No custom code provided, using existing code")
                                else:
                                    # User doesn't want to provide custom code
                                    # Mark as failed and let planner handle it
                                    break
                    else:
                        logfire.warning("Failed to analyze error")
                        break
                else:
                    logfire.error(f"Failed to execute step after {max_retries} attempts")
                    break
            
            # Update the step code with the final version used
            updated_plan = plan.model_copy(deep=True)
            updated_plan.steps[step_index].code = generated_code
            
            return updated_plan, execution_result
    
    async def _critique_plan(self, plan: AudioPlan, task_description: str) -> AudioPlan:
        """Have the Critique Agent review and improve the plan."""
        with logfire.span("critique_plan"):
            critique_deps = CritiqueAgentDependencies(
                context=self.context,
                tool_definitions=self.tool_definitions,
                plan=plan,
                critique_type="plan",
                task_description=task_description
            )
            
            critique_result = await critique_agent.run(
                f"Critique the audio processing plan for: {task_description}",
                deps=critique_deps,
                usage=self.usage,
                usage_limits=self.usage_limits
            )
            
            critique_response = None
            for message in critique_result.all_messages():
                for part in message.parts:
                    if hasattr(part, "tool_name") and part.tool_name == "critique_plan":
                        critique_response = critique_result.data
                        break
                if critique_response:
                    break
                    
            if critique_response and not critique_response.is_approved:
                # Log the critique reasoning
                logfire.info(f"Plan critique: {critique_response.reasoning}")
                
                # If we have specific suggestions, show them
                if critique_response.suggestions:
                    suggestion_text = "\n- ".join([""] + critique_response.suggestions)
                    logfire.info(f"Plan improvement suggestions:{suggestion_text}")
                
                # If we have an improved version and interactive mode is enabled,
                # ask the user if they want to use it
                if critique_response.improved_version and self.interactive:
                    should_use_improved = self.feedback_handler.feedback_manager.request_confirmation(
                        "The Critique Agent has suggested improvements to the plan. Use the improved version?",
                        f"Reasoning: {critique_response.reasoning}",
                        severity="info"
                    )
                    
                    if should_use_improved:
                        # Parse the improved plan (this would need implementation)
                        # For now, just return the original plan
                        logfire.info("Using improved plan from critique")
                        # In a real implementation, you would parse the improved_version
                        # and return a new AudioPlan object
                        
            return plan
    
    async def _perform_qa(
        self,
        task_description: str,
        original_audio_path: str,
        processed_audio_path: str,
        plan: AudioPlan,
        execution_result: ExecutionResult
    ) -> QAResponse:
        """Have the QA Agent verify the quality of the processed audio."""
        with logfire.span("perform_qa"):
            qa_deps = QAAgentDependencies(
                context=self.context,
                task_description=task_description,
                plan=plan,
                execution_result=execution_result,
                original_audio_path=Path(original_audio_path),
                processed_audio_path=Path(processed_audio_path),
                tool_definitions=self.tool_definitions
            )
            
            qa_result = await qa_agent.run(
                f"Evaluate the quality of the processed audio for: {task_description}",
                deps=qa_deps,
                usage=self.usage,
                usage_limits=self.usage_limits
            )
            
            qa_response = None
            for message in qa_result.all_messages():
                for part in message.parts:
                    if hasattr(part, "tool_name") and part.tool_name == "evaluate_audio_output":
                        qa_response = qa_result.data
                        break
                if qa_response:
                    break
                    
            if qa_response:
                # Log the QA assessment
                meets_requirements = "YES" if qa_response.meets_requirements else "NO"
                logfire.info(f"QA assessment - Meets requirements: {meets_requirements}")
                
                if qa_response.reasoning:
                    logfire.info(f"QA reasoning: {qa_response.reasoning}")
                    
                if qa_response.issues:
                    issues_text = "\n- ".join([""] + qa_response.issues)
                    logfire.info(f"QA identified issues:{issues_text}")
                    
                if qa_response.suggestions:
                    suggestion_text = "\n- ".join([""] + qa_response.suggestions)
                    logfire.info(f"QA suggestions:{suggestion_text}")
                    
                return qa_response
            else:
                # Return a default response if QA failed
                logfire.warning("QA evaluation failed")
                return QAResponse(
                    meets_requirements=True,
                    reasoning="QA evaluation failed, assuming requirements are met"
                )
    
    def _get_final_result_path(self, plan: AudioPlan) -> str:
        """Determine the path to the final processed audio file."""
        # Try to find the output path from the last successful step
        for i in reversed(range(len(plan.steps))):
            if plan.steps[i].status == StepStatus.DONE and i in plan.completed_step_indices:
                # Check the last execution result if available
                for output_file in [plan.current_audio_path]:
                    if os.path.exists(str(output_file)):
                        return str(output_file)
                        
        # Fallback to the current audio path in the plan
        return str(plan.current_audio_path) 