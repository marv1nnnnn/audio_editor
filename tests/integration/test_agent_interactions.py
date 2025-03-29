"""
Integration tests for agent interactions.
"""
import pytest
import asyncio
from pathlib import Path
import os

from audio_editor.agents.planner import planner_agent
from audio_editor.agents.executor import executor_agent
from audio_editor.agents.mcp import MCPCodeExecutor
from audio_editor.agents.models import (
    AudioPlan, PlanStep, ExecutionResult, StepStatus,
    AudioInput
)
from audio_editor.agents.dependencies import (
    AudioProcessingContext, PlannerDependencies, ExecutorDependencies
)


@pytest.mark.asyncio
async def test_planner_executor_interaction(
    temp_workspace,
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input,
    sample_audio_file
):
    """Test interaction between Planner and Executor agents."""
    # Initialize dependencies
    planner_deps = PlannerDependencies(
        context=audio_processing_context,
        task_description="Get the length of the audio file",
        tool_definitions=sample_tool_definitions,
        audio_input=sample_audio_input
    )
    
    # Get initial plan from Planner
    planner_result = await planner_agent.run(
        "Create a plan to get the audio file length",
        deps=planner_deps
    )
    
    # Extract plan from result
    plan = None
    for message in planner_result.all_messages():
        for part in message.parts:
            if hasattr(part, "tool_name") and part.tool_name == "generate_initial_plan":
                plan = planner_result.data
                break
    
    assert plan is not None
    assert isinstance(plan, AudioPlan)
    assert len(plan.steps) > 0
    
    # Initialize Executor dependencies
    executor_deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=plan
    )
    
    # Get code from Executor
    executor_result = await executor_agent.run(
        f"Generate code for step: {plan.steps[0].description}",
        deps=executor_deps
    )
    
    # Extract code from result
    code = None
    for message in executor_result.all_messages():
        for part in message.parts:
            if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                code = executor_result.data.generated_code
                break
    
    assert code is not None
    assert isinstance(code, str)
    
    # Execute the code with MCP
    mcp = MCPCodeExecutor(str(temp_workspace))
    execution_result = await mcp.execute_code(code)
    
    assert execution_result.status == "SUCCESS"
    assert float(execution_result.output) > 0  # Should get audio length


@pytest.mark.asyncio
async def test_error_recovery_flow(
    temp_workspace,
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input
):
    """Test error recovery flow between agents."""
    # Create a plan with an intentionally failing step
    plan = AudioPlan(
        task_description="Test error recovery",
        steps=[
            PlanStep(
                description="Try to read a non-existent file",
                tool_name="LEN",
                tool_args={"wav_path": "nonexistent.wav"}
            )
        ],
        current_audio_path=Path("nonexistent.wav")
    )
    
    # Initialize dependencies
    executor_deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=plan
    )
    
    # First execution attempt
    executor_result = await executor_agent.run(
        "Generate code for the failing step",
        deps=executor_deps
    )
    
    code = None
    for message in executor_result.all_messages():
        for part in message.parts:
            if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                code = executor_result.data.generated_code
                break
    
    assert code is not None
    
    # Execute and get error
    mcp = MCPCodeExecutor(str(temp_workspace))
    execution_result = await mcp.execute_code(code)
    
    assert execution_result.status == "FAILURE"
    
    # Update dependencies with error
    executor_deps.execution_result = execution_result
    
    # Try code refinement
    executor_result = await executor_agent.run(
        f"Refine code after error: {execution_result.error_message}",
        deps=executor_deps
    )
    
    refined_code = None
    for message in executor_result.all_messages():
        for part in message.parts:
            if hasattr(part, "tool_name") and part.tool_name == "refine_code_after_error":
                refined_code = executor_result.data.generated_code
                break
    
    assert refined_code is not None
    assert refined_code != code  # Should be different from original code


@pytest.mark.asyncio
async def test_checkpoint_recovery(
    temp_workspace,
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input,
    sample_audio_file
):
    """Test recovery from checkpoints in the workflow."""
    # Create a plan with multiple steps
    plan = AudioPlan(
        task_description="Test checkpoint recovery",
        steps=[
            PlanStep(
                description="Get audio length",
                tool_name="LEN",
                tool_args={"wav_path": str(sample_audio_file)}
            ),
            PlanStep(
                description="Copy audio file",
                tool_name="COPY",
                tool_args={
                    "wav_path": str(sample_audio_file),
                    "output_path": str(temp_workspace / "copy.wav")
                }
            ),
            PlanStep(
                description="Try failing step",
                tool_name="INVALID_TOOL",
                tool_args={}
            )
        ],
        current_audio_path=sample_audio_file,
        checkpoint_indices=[1]  # Set checkpoint after copy
    )
    
    # Execute steps until failure
    mcp = MCPCodeExecutor(str(temp_workspace))
    
    for i in range(2):  # Execute first two steps
        executor_deps = ExecutorDependencies(
            context=audio_processing_context,
            tool_definitions=sample_tool_definitions,
            plan_step_index=i,
            plan=plan
        )
        
        executor_result = await executor_agent.run(
            f"Generate code for step {i + 1}",
            deps=executor_deps
        )
        
        code = None
        for message in executor_result.all_messages():
            for part in message.parts:
                if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                    code = executor_result.data.generated_code
                    break
        
        execution_result = await mcp.execute_code(code)
        assert execution_result.status == "SUCCESS"
        
        # Update plan
        plan.steps[i].status = StepStatus.DONE
        if execution_result.output_path:
            plan.current_audio_path = Path(execution_result.output_path)
    
    # Try failing step
    executor_deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=2,
        plan=plan
    )
    
    executor_result = await executor_agent.run(
        "Generate code for failing step",
        deps=executor_deps
    )
    
    code = None
    for message in executor_result.all_messages():
        for part in message.parts:
            if hasattr(part, "tool_name") and part.tool_name == "generate_code_for_step":
                code = executor_result.data.generated_code
                break
    
    execution_result = await mcp.execute_code(code)
    assert execution_result.status == "FAILURE"
    
    # Verify we can recover from checkpoint
    assert 1 in plan.checkpoint_indices
    assert plan.steps[1].status == StepStatus.DONE
    assert os.path.exists(plan.current_audio_path) 