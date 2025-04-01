"""
Coordinator for the multi-agent audio processing system using Pydantic AI.
"""
import os
import time
import hashlib
import re
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import inspect
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry, BinaryContent
from pydantic_ai.usage import Usage, UsageLimits

from audio_editor import audio_tools
from .mcp import MCPCodeExecutor
from .user_feedback import ConsoleUserFeedbackHandler
from .models import StepInfo, ExecutionResult, PlanResult, ProcessingResult, CodeGenerationResult, AudioContent


# Configure Logfire for debugging
logfire.configure()
Agent.instrument_all()


# Define Pydantic models for structured data
class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    signature: str
    description: str
    docstring: str = ""


class WorkflowState(BaseModel):
    """State of the workflow, used for dependency injection."""
    workspace_dir: Path
    workflow_file: Path
    current_step_id: Optional[str] = None
    task_description: str = ""
    original_audio: Path = ""
    tool_definitions: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    current_markdown: str = ""


# Define the multi-agent system
# 1. Main coordinator agent
coordinator_agent = Agent(
    'gemini-2.0-flash',
    deps_type=WorkflowState,
    result_type=ProcessingResult,
    system_prompt="""
    You are an audio processing coordinator. Your job is to:
    1. Generate a plan for processing audio
    2. Generate code for each step
    3. Execute the code
    4. Track progress in a Markdown file
    5. Return the final processed audio

    You will maintain a workflow in Markdown format.
    """
)


# 2. Planner agent
planner_agent = Agent(
    'gemini-2.0-flash',
    deps_type=WorkflowState,
    result_type=PlanResult,
    system_prompt="""
    You are a planning agent for audio processing. Your job is to:
    1. Create a detailed Product Requirements Document (PRD) based on the task
    2. Design a sequence of specific, achievable processing steps

    Each step must have:
    - A descriptive title
    - A unique ID (step_1, step_2, etc.)
    - A detailed description of what it should accomplish
    - Input and output paths

    Rules for file paths:
    1. The first step MUST use the EXACT input audio filename. **This filename is available in the dependencies object as `deps.original_audio.name`. Use this value directly.**
    2. Each subsequent step's input path MUST match the previous step's output path
    3. Output paths should be descriptive of the operation (e.g., "normalized_audio.wav", "filtered_audio.wav")
    4. All paths are relative to the working directory (`deps.workspace_dir`). **Only use the filename for the paths in the plan.**
    5. Never assume or create arbitrary paths.
    6. NEVER use generic names like "input.wav" or "output.wav"
    7. DO NOT modify or change the input filename obtained from `deps.original_audio.name`.

    Example (Assuming deps.original_audio.name is 'sample1.wav'):
    ### Step 1: Apply High-Pass Filter

    * **ID:** `step_1`
    * **Description:** Remove low frequencies below 400 Hz to reduce rumble
    * **Input Audio:** `sample1.wav`  # MUST match deps.original_audio.name
    * **Output Audio:** `highpass_filtered.wav`

    ### Step 2: Normalize Volume

    * **ID:** `step_2`
    * **Description:** Normalize the audio to broadcast standard
    * **Input Audio:** `highpass_filtered.wav`  # Match previous step's output
    * **Output Audio:** `normalized_audio.wav`
    """
)


# 3. Code generation agent
code_gen_agent = Agent(
    'gemini-2.0-flash',
    deps_type=WorkflowState,
    result_type=CodeGenerationResult,
    system_prompt="""
    You are a code generation agent for audio processing. Your job is to:
    1. Generate a single line of Python code to implement a specific step
    2. Use the correct tool from the available tool definitions
    3. Use the EXACT file paths from the step information
    
    Rules for code generation:
    1. Use ONLY the tools provided in the tool definitions
    2. Each tool takes a wav_path input and returns an output path
    3. ALWAYS use the EXACT input_audio path from the step info - do not modify or assume paths
    4. ALWAYS use the EXACT output_audio path from the step info - do not modify or assume paths
    5. Include any necessary parameters based on the tool's signature
    6. Return exactly one line of executable Python code
    7. Do not add any imports or other statements
    8. Do not use variables - use string literals directly
    9. ALWAYS use keyword arguments (e.g., wav_path="sample_1.wav") instead of positional arguments
    10. NEVER assume or hardcode file paths - they must come from the step info
    
    Example:
    If the step info shows:
    - Input Audio: `sample_1.wav`
    - Output Audio: `normalized_audio.wav`
    
    And the tool is:
    Tool: LOUDNESS_NORM
    Signature: (wav_path: str, volume: float = -23.0, out_wav: str = None)
    
    You would generate:
    LOUDNESS_NORM(wav_path="sample_1.wav", volume=-20.0, out_wav="normalized_audio.wav")
    
    NOT:
    LOUDNESS_NORM(wav_path="input.wav", volume=-20.0, out_wav="output.wav")  # Don't assume paths
    """
)


# System prompts to add context
@coordinator_agent.system_prompt
def add_workspace_info(ctx: RunContext[WorkflowState]) -> str:
    """Add information about the workspace directory."""
    return f"Working directory: {ctx.deps.workspace_dir}\nWorkflow file: {ctx.deps.workflow_file}"


@planner_agent.system_prompt
def add_tool_info(ctx: RunContext[WorkflowState]) -> str:
    """Add information about available tools."""
    tools_text = "\n".join([
        f"{name}: {info['description']} - {info['signature']}"
        for name, info in ctx.deps.tool_definitions.items()
    ])
    return f"Available tools:\n{tools_text}"


@code_gen_agent.system_prompt
def add_code_gen_context(ctx: RunContext[WorkflowState]) -> str:
    """Add tool definitions and step information to the code generation context."""
    with logfire.span("add_code_gen_context"):
        # Get the current step info from the markdown
        step_info = _get_step_info(ctx.deps.workflow_file, ctx.deps.current_step_id)
        if not step_info:
            logfire.error(f"Error: No step information found for step_id {ctx.deps.current_step_id}") # Added logging
            return "Error: No step information found"

        # --- START NEW CODE ---
        # Extract the exact path strings needed
        input_path_value = step_info.input_audio
        output_path_value = step_info.output_audio

        if not input_path_value:
            logfire.error(f"Error: Input audio path missing in step info for {step_info.id}")
            return "Error: Input audio path missing in step info"
        if not output_path_value:
            logfire.error(f"Error: Output audio path missing in step info for {step_info.id}")
            return "Error: Output audio path missing in step info"
        # --- END NEW CODE ---

        # Format tool definitions
        tool_defs = []
        for name, info in ctx.deps.tool_definitions.items():
            tool_defs.append(f"Tool: {name}")
            tool_defs.append(f"Signature: {info['signature']}")
            tool_defs.append(f"Description: {info['description']}")
            tool_defs.append("")

        tools_str = "\n".join(tool_defs)

        # --- MODIFIED RETURN STRING ---
        return f"""
Current step information (for context only):
- ID: {step_info.id}
- Description: {step_info.description}
- Input Audio: {step_info.input_audio}
- Output Audio: {step_info.output_audio}

Available tools:
{tools_str}

Task description: {ctx.deps.task_description}

CRITICAL INSTRUCTIONS FOR CODE GENERATION:
1. You MUST generate exactly one line of Python code.
2. The code MUST call one of the available tools.
3. The code MUST use keyword arguments (e.g., wav_path="...").
4. **Use the EXACT input file path: '{input_path_value}'** for the 'wav_path' argument (or the primary input argument if named differently).
5. **Use the EXACT output file path: '{output_path_value}'** for the 'out_wav' argument (or the primary output argument if named differently).
6. Include other necessary parameters based on the tool's signature and the step description.
7. **DO NOT** use the literal string "original_audio.wav" unless the input path is actually 'original_audio.wav'. Use the value provided: '{input_path_value}'.

Example: If the required input is 'input_file.wav' and output is 'output_file.wav' for LOUDNESS_NORM, generate:
LOUDNESS_NORM(wav_path="input_file.wav", volume=-20.0, out_wav="output_file.wav")

Now, generate the code for step {step_info.id} using input '{input_path_value}' and output '{output_path_value}'.
"""
        # --- END MODIFIED RETURN STRING ---

# Tools for the coordinator agent
@coordinator_agent.tool
async def create_workflow_markdown(
    ctx: RunContext[WorkflowState],
    task_description: str,
    workflow_id: str
) -> str:
    """Create the initial Markdown workflow file."""
    with logfire.span("create_workflow_markdown"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use the actual filename from the original_audio path in the context
        input_audio = ctx.deps.original_audio.name
        
        content = f"""# Audio Processing Workflow: {workflow_id}

## 1. Request Summary

* **Original Request:**
    ```text
    {task_description}
    ```
* **Input Audio:** `{input_audio}`  # Correctly uses context here
* **Timestamp:** `{timestamp}`
* **Workflow ID:** `{workflow_id}`

## 2. Product Requirements Document (PRD)

*This section will be generated by the Planner Agent.*

## 3. Processing Plan & Status

*This section will contain the processing steps generated by the Planner Agent.*

## 4. Final Output

* **Overall Status:** `IN_PROGRESS`
* **Final Audio Path:** `N/A`
* **Summary:** *Processing in progress*

## 5. Workflow Log

* `{timestamp}`: Workflow Initiated.
"""

        with open(ctx.deps.workflow_file, "w") as f:
            f.write(content)
            
        logfire.info(f"Initialized workflow Markdown file: {ctx.deps.workflow_file}")
        return content


@coordinator_agent.tool
async def generate_plan_and_prd(
    ctx: RunContext[WorkflowState]
) -> PlanResult:
    """Generate the Product Requirements Document and plan using the Planner Agent."""
    with logfire.span("generate_plan_and_prd"):
        logfire.info("Generating plan and PRD...")

        # Update workflow log
        _append_workflow_log(ctx.deps.workflow_file, "Generating PRD and plan...")

        # --- START NEW/MODIFIED CODE ---

        # Get the actual input filename from the context dependencies
        original_filename = ctx.deps.original_audio.name
        if not original_filename:
             logfire.error("Original audio filename is missing in deps during planning!")
             # Optionally, return an error or raise an exception
             # For now, we might fallback, but ideally this should be present
             original_filename = "fallback_input.wav" # Or raise error

        # Construct a more explicit prompt for the planner agent for THIS specific run
        # Inject the correct filename directly into the task description/prompt
        planner_prompt = f"""
        Generate a PRD and processing steps for the task: '{ctx.deps.task_description}'.

        VERY IMPORTANT: The input audio file for the *first* step (and any other step requiring the original audio) MUST be named exactly: '{original_filename}'
        Use this filename explicitly where the original audio is needed as input.
        Do NOT use generic names like 'input.wav' or 'original_audio.wav'. Use '{original_filename}'.

        Follow all other planning rules regarding subsequent step paths (output of step N becomes input of step N+1) and descriptive output names.
        """

        logfire.info(f"Running planner agent. Explicitly specifying input filename in prompt: {original_filename}")

        # Call the planner agent using the explicit prompt
        planner_result = await planner_agent.run(
            planner_prompt,  # Use the specifically constructed prompt
            deps=ctx.deps
        )
        # --- END NEW/MODIFIED CODE ---

        # Log the raw planner result for debugging (handle potential serialization errors)
        try:
            # Ensure we are accessing the data attribute which should be PlanResult
            if hasattr(planner_result, 'data') and isinstance(planner_result.data, PlanResult):
                 result_json_str = planner_result.data.model_dump_json()
                 logfire.info(f"Planner agent PLAN result data: {result_json_str}") # Log the actual plan
            else:
                 logfire.warning(f"Planner agent result was not in expected format: {type(planner_result.data)}")
                 result_json_str = "<Error: Planner result data format unexpected>"
        except Exception as e:
            result_json_str = f"<Error serializing planner result data: {e}>"
            logfire.error(result_json_str)


        # Check if the planner actually returned valid data
        if not hasattr(planner_result, 'data') or not isinstance(planner_result.data, PlanResult):
             _append_workflow_log(ctx.deps.workflow_file, f"Planner agent failed to return valid plan data. Result: {result_json_str}")
             # Handle the error appropriately, maybe raise or return a specific error object
             raise ValueError(f"Planner agent did not return a valid PlanResult. See logs.")


        # Update the workflow file with the PRD and plan (using planner_result.data)
        markdown_content = _read_markdown_file(ctx.deps.workflow_file)

        # Replace the placeholder sections
        updated_content = re.sub(
            r"## 2\. Product Requirements Document \(PRD\)\n\n\*This section will be generated by the Planner Agent\.\*",
            f"## 2. Product Requirements Document (PRD)\n\n{planner_result.data.prd}", # Access .data
            markdown_content
        )

        # Create the steps content (using planner_result.data.steps)
        steps_content = "\n\n".join([
            f"### Step {i+1}: {step.title}\n\n"
            f"* **ID:** `{step.id}`\n"
            f"* **Description:** {step.description}\n"
            f"* **Status:** READY\n"
            # --- Ensure the correct filename is used here IF the planner worked ---
            f"* **Input Audio:** `{step.input_audio}`\n"
            f"* **Output Audio:** `{step.output_audio}`\n"
            # --- / ---
            f"* **Code:**\n```python\n# Placeholder - will be generated by Code Generator\n```\n"
            f"* **Execution Results:**\n```text\n# Placeholder - will be filled by Executor\n```\n"
            f"* **Timestamp Start:** `N/A`\n"
            f"* **Timestamp End:** `N/A`"
            for i, step in enumerate(planner_result.data.steps) # Access .data.steps
        ])

        # Replace the processing plan section
        updated_content = re.sub(
            r"## 3\. Processing Plan & Status\n\n\*This section will contain the processing steps generated by the Planner Agent\.\*",
            f"## 3. Processing Plan & Status\n\n{steps_content}",
            updated_content
        )

        # Write the updated content
        _write_markdown_file(ctx.deps.workflow_file, updated_content)

        # Update workflow log
        _append_workflow_log(ctx.deps.workflow_file, f"Generated PRD and processing plan with {len(planner_result.data.steps)} steps.") # Access .data.steps

        return planner_result.data # Return the PlanResult object


@coordinator_agent.tool
async def find_next_step(
    ctx: RunContext[WorkflowState]
) -> Optional[str]:
    """Find the ID of the next step to process."""
    with logfire.span("find_next_step"):
        markdown_content = _read_markdown_file(ctx.deps.workflow_file)
        
        # Find all step sections
        step_pattern = r"### Step \d+: .*?\n\n(.*?)(?=\n### |\n## |\Z)"
        step_matches = re.finditer(step_pattern, markdown_content, re.DOTALL)
        
        ready_step_id = None
        pending_step_id = None
        
        for match in step_matches:
            step_content = match.group(1)
            
            # Get step ID
            id_pattern = r"\* \*\*ID:\*\* `(.*?)`"
            id_match = re.search(id_pattern, step_content)
            
            if not id_match:
                continue
                
            step_id = id_match.group(1)
            
            # Get step status
            status_pattern = r"\* \*\*Status:\*\* (.*?)(?=\n)"
            status_match = re.search(status_pattern, step_content)
            
            if not status_match:
                continue
                
            status = status_match.group(1)
            
            if status == "READY":
                # Return the first READY step
                logfire.info(f"Found next step: {step_id} (READY)")
                return step_id
            elif status == "PENDING" and pending_step_id is None:
                # Remember the first PENDING step
                pending_step_id = step_id
        
        # If no READY steps found, return the first PENDING step
        if pending_step_id:
            logfire.info(f"Found next step: {pending_step_id} (PENDING)")
            return pending_step_id
            
        logfire.info("No next step found")
        return None


@coordinator_agent.tool
async def generate_code_for_step(
    ctx: RunContext[WorkflowState],
    step_id: str
) -> str:
    """Generate code for a specific step using the Code Generation Agent."""
    with logfire.span("generate_code_for_step", step_id=step_id):
        # Update step status to RUNNING
        _update_step_fields(
            ctx.deps.workflow_file, 
            step_id, 
            {
                "Status": "RUNNING",
                "Timestamp Start": f"`{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            }
        )
        _append_workflow_log(ctx.deps.workflow_file, f"Generating code for step {step_id}...")
        
        # Get step info to access input/output paths
        step_info = _get_step_info(ctx.deps.workflow_file, step_id)
        if not step_info:
            error_msg = f"Error: No step information found for step_id {step_id}"
            logfire.error(error_msg)
            _update_step_fields(
                ctx.deps.workflow_file,
                step_id,
                {"Status": "FAILED"},
                execution_results=error_msg
            )
            return error_msg
        
        # Load audio content for direct model access
        input_audio_path = ctx.deps.workspace_dir / step_info.input_audio
        audio_content = _load_audio_content(input_audio_path)
        
        # Update context with current step
        updated_deps = ctx.deps.model_copy()
        updated_deps.current_step_id = step_id
        
        # Add audio content to dependencies if available
        if audio_content:
            logfire.info(f"Adding audio content to dependencies for step {step_id}")
            updated_deps.audio_content = audio_content
        
        # Call the code generation agent
        try:
            deps_json = updated_deps.model_dump_json()
        except Exception as e:
            deps_json = f"<Error serializing deps: {e}>"
        logfire.info(f"Running code generation agent with deps: {deps_json}")
        
        # Create message list with both text and audio if available
        messages = [f"Generate code for step {step_id} in the workflow"]
        
        # If we have audio content, include it in the request to the model
        if audio_content and audio_content.content:
            logfire.info("Including audio content in the request to the model")
            messages.append(audio_content.content)
        
        # Run the agent with the updated dependencies and messages
        code_result = await code_gen_agent.run(
            messages,
            deps=updated_deps
        )
        
        # Handle the result, which might be a string or an AgentRunResult
        if hasattr(code_result, 'data'):
            if hasattr(code_result.data, 'code'):
                generated_code = code_result.data.code
            else:
                generated_code = str(code_result.data)
        else:
            generated_code = str(code_result)
            
        logfire.info(f"Generated code: {generated_code}")
        
        # Update the step with the generated code
        _update_step_fields(
            ctx.deps.workflow_file, 
            step_id, 
            {"Status": "CODE_GENERATED"},
            code=generated_code
        )
        _append_workflow_log(ctx.deps.workflow_file, f"Generated code for step {step_id}.")
        
        return generated_code


@coordinator_agent.tool
async def execute_code_for_step(
    ctx: RunContext[WorkflowState],
    step_id: str,
    code: str
) -> ExecutionResult:
    """Execute the code for a specific step."""
    with logfire.span("execute_code_for_step", step_id=step_id):
        # Update step status to EXECUTING
        _update_step_fields(ctx.deps.workflow_file, step_id, {"Status": "EXECUTING"})
        _append_workflow_log(ctx.deps.workflow_file, f"Executing code for step {step_id}...")
        
        # Create MCP executor
        mcp = MCPCodeExecutor(ctx.deps.workspace_dir)
        
        # Execute the code
        try:
            logfire.info(f"Executing code: {code}")
            result = await mcp.execute_code(code, f"Step {step_id} execution")
            
            if result.status == "SUCCESS":
                # Update step status to DONE
                _update_step_fields(
                    ctx.deps.workflow_file, 
                    step_id, 
                    {
                        "Status": "DONE",
                        "Timestamp End": f"`{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    },
                    execution_results=result.output
                )
                _append_workflow_log(
                    ctx.deps.workflow_file, 
                    f"Step {step_id} completed successfully. Output: {result.output_path}"
                )
            else:
                # Update step status to FAILED
                _update_step_fields(
                    ctx.deps.workflow_file, 
                    step_id, 
                    {
                        "Status": "FAILED",
                        "Timestamp End": f"`{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    },
                    execution_results=result.error_message
                )
                _append_workflow_log(ctx.deps.workflow_file, f"Step {step_id} failed: {result.error_message}")
                
                # Try to analyze and fix the error
                fixed = await analyze_and_fix_error(ctx, step_id, code, result.error_message)
                if fixed:
                    # Reset the step to READY
                    _update_step_fields(ctx.deps.workflow_file, step_id, {"Status": "READY"})
                    _append_workflow_log(ctx.deps.workflow_file, f"Fixed error in step {step_id}. Retrying...")
                    
                    # Raise ModelRetry to let the agent know it should retry
                    raise ModelRetry(f"Fixed error in step {step_id}. Retrying...")
            
            return result
        except ModelRetry:
            # Let the ModelRetry propagate
            raise
        except Exception as e:
            # Handle unexpected errors
            error_message = f"Unexpected error: {str(e)}"
            logfire.error(error_message)
            
            # Update step status to FAILED
            _update_step_fields(
                ctx.deps.workflow_file, 
                step_id, 
                {
                    "Status": "FAILED",
                    "Timestamp End": f"`{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                },
                execution_results=error_message
            )
            _append_workflow_log(ctx.deps.workflow_file, f"Step {step_id} failed with exception: {error_message}")
            
            # Create a failed result
            return ExecutionResult(
                status="FAILURE",
                error_message=error_message,
                duration=0.0
            )


@coordinator_agent.tool
async def analyze_and_fix_error(
    ctx: RunContext[WorkflowState],
    step_id: str,
    code: str,
    error_message: str
) -> bool:
    """Analyze an error in a step and attempt to fix it."""
    with logfire.span("analyze_and_fix_error", step_id=step_id):
        # Define the error analyzer as an inline Agent
        error_analyzer = Agent(
            'gemini-2.0-flash',
            system_prompt="""
            You are an Error Analyzer for audio processing code. Your task is to:
            1. Analyze the error message
            2. Identify the root cause
            3. Generate a fixed version of the code
            
            The code is typically a single line calling an audio processing function.
            Focus on fixing parameter types, file paths, and function names.
            
            IMPORTANT: When fixing file paths:
            1. Use the exact input file path from the step info
            2. Do not modify or assume paths
            3. Keep the output path as specified
            """
        )
        
        # Get the step info to ensure we use the correct file paths
        step_info = _get_step_info(ctx.deps.workflow_file, step_id)
        if not step_info:
            logfire.error(f"Could not find step info for {step_id}")
            return False
            
        # Prepare the prompt
        prompt = f"""
        Analyze and fix the following code that generated an error:
        
        Code:
        ```python
        {code}
        ```
        
        Error message:
        ```
        {error_message}
        ```
        
        Step information:
        - Input Audio: {step_info.input_audio}
        - Output Audio: {step_info.output_audio}
        
        Available tools:
        {json.dumps(ctx.deps.tool_definitions, indent=2)}
        
        Return ONLY the fixed code, without any explanation or formatting.
        Make sure to use the exact input file path: {step_info.input_audio}
        """
        
        try:
            # Call the error analyzer
            logfire.info(f"Running error analyzer with prompt: {prompt}")
            result = await error_analyzer.run(prompt)
            
            # Handle the result, which might be a string or an AgentRunResult
            if hasattr(result, 'data'):
                fixed_code = result.data
            else:
                fixed_code = str(result)
                
            # Clean up the code if it's wrapped in markdown
            fixed_code = fixed_code.strip()
            if fixed_code.startswith('```python'):
                fixed_code = fixed_code[8:]
            if fixed_code.endswith('```'):
                fixed_code = fixed_code[:-3]
            fixed_code = fixed_code.strip()
            
            if fixed_code != code:
                logfire.info(f"Generated fixed code: {fixed_code}")
                
                # Update the step with the fixed code
                _update_step_fields(
                    ctx.deps.workflow_file, 
                    step_id, 
                    {},
                    code=fixed_code
                )
                
                return True
            else:
                logfire.warning(f"Error analyzer did not generate different code for step {step_id}")
                return False
        except Exception as e:
            logfire.error(f"Error analysis failed: {str(e)}")
            return False


@coordinator_agent.tool
async def get_final_output_path(
    ctx: RunContext[WorkflowState]
) -> str:
    """Get the path to the final output file from the last completed step."""
    with logfire.span("get_final_output_path"):
        markdown_content = _read_markdown_file(ctx.deps.workflow_file)
        
        # Find all step sections
        step_pattern = r"### Step \d+: .*?\n\n(.*?)(?=\n### |\n## |\Z)"
        step_matches = list(re.finditer(step_pattern, markdown_content, re.DOTALL))
        
        # Go through steps in reverse to find the last completed one
        for match in reversed(step_matches):
            step_content = match.group(1)
            
            # Check if the step is DONE
            status_pattern = r"\* \*\*Status:\*\* (.*?)(?=\n)"
            status_match = re.search(status_pattern, step_content)
            
            if status_match and status_match.group(1) == "DONE":
                # Get the output path
                output_pattern = r"\* \*\*Output Audio:\*\* `(.*?)`"
                output_match = re.search(output_pattern, step_content)
                
                if output_match:
                    output_path = output_match.group(1)
                    logfire.info(f"Found final output path: {output_path}")
                    return output_path
        
        # If no completed steps found, return the original audio path
        return ctx.deps.original_audio


@coordinator_agent.tool
async def finish_workflow(
    ctx: RunContext[WorkflowState],
    final_output_path: str,
    steps_completed: int
) -> None:
    """Mark the workflow as complete and update the final output section."""
    with logfire.span("finish_workflow"):
        # Update the final output section
        _update_workflow_section(ctx.deps.workflow_file, "Final Output", {
            "Overall Status": "SUCCESS",
            "Final Audio Path": f"`{final_output_path}`",
            "Summary": f"Audio processing completed successfully with {steps_completed} steps."
        })
        
        # Add final log entry
        _append_workflow_log(ctx.deps.workflow_file, f"Workflow finished successfully. Final output: {final_output_path}")
        
        logfire.info(f"Workflow completed successfully. Final output: {final_output_path}")



# Helper functions for Markdown manipulation
def _read_markdown_file(markdown_file: str) -> str:
    """Read the content of a Markdown file."""
    with open(markdown_file, "r") as f:
        return f.read()


def _write_markdown_file(markdown_file: str, content: str) -> None:
    """Write content to a Markdown file."""
    with open(markdown_file, "w") as f:
        f.write(content)


def _append_workflow_log(markdown_file: str, log_entry: str) -> None:
    """Append an entry to the Workflow Log section."""
    with logfire.span("append_workflow_log"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"* `{timestamp}`: {log_entry}"
        
        markdown_content = _read_markdown_file(markdown_file)
        
        # Find the Workflow Log section
        log_section_pattern = r"## 5\. Workflow Log\n\n((?:\* `.*`.*\n)*)"
        match = re.search(log_section_pattern, markdown_content)
        
        if match:
            # Append to existing logs
            existing_logs = match.group(1)
            updated_logs = existing_logs + log_line + "\n"
            markdown_content = markdown_content.replace(match.group(0), f"## 5. Workflow Log\n\n{updated_logs}")
        else:
            # Fallback: Append to the end of the file
            markdown_content += f"\n* `{timestamp}`: {log_entry}\n"
        
        _write_markdown_file(markdown_file, markdown_content)
        logfire.debug(f"Added log entry: {log_entry}")


def _update_workflow_section(
    markdown_file: str, 
    section_name: str, 
    fields: Dict[str, str]
) -> None:
    """Update fields in a specific section of the workflow Markdown file."""
    with logfire.span("update_workflow_section"):
        markdown_content = _read_markdown_file(markdown_file)
        
        # Find the section
        section_pattern = fr"## (?:\d+\. )?{re.escape(section_name)}\n\n(.*?)(?=\n## |\Z)"
        match = re.search(section_pattern, markdown_content, re.DOTALL)
        
        if not match:
            logfire.warning(f"Section '{section_name}' not found in {markdown_file}")
            return
        
        section_content = match.group(1)
        updated_section = section_content
        
        # Update each field
        for field_name, field_value in fields.items():
            field_pattern = fr"\* \*\*{re.escape(field_name)}:\*\* .*?(?=\n\* \*\*|\n```|\Z)"
            replacement = f"* **{field_name}:** {field_value}"
            
            if re.search(field_pattern, updated_section, re.DOTALL):
                # Update existing field
                updated_section = re.sub(
                    field_pattern,
                    replacement,
                    updated_section,
                    flags=re.DOTALL
                )
            else:
                # Add new field (before code blocks)
                code_pos = updated_section.find("```python")
                if code_pos >= 0:
                    updated_section = (
                        updated_section[:code_pos] + 
                        f"{replacement}\n" + 
                        updated_section[code_pos:]
                    )
                else:
                    updated_section += f"\n{replacement}"
        
        # Replace the section in the full content
        updated_content = markdown_content.replace(
            match.group(0),
            f"## {section_name}\n\n{updated_section}"
        )
        
        _write_markdown_file(markdown_file, updated_content)
        logfire.debug(f"Updated section '{section_name}' with {len(fields)} fields")


def _update_step_fields(
    markdown_file: str, 
    step_id: str, 
    fields: Dict[str, str],
    code: Optional[str] = None,
    execution_results: Optional[str] = None
) -> bool:
    """Update fields for a specific step in the workflow Markdown file."""
    with logfire.span("update_step_fields"):
        markdown_content = _read_markdown_file(markdown_file)
        
        # Find the step section
        step_pattern = r"(### Step \d+: .*?\n\n)(.*?)(?=\n### |\n## |\Z)"
        step_matches = list(re.finditer(step_pattern, markdown_content, re.DOTALL))
        
        for match in step_matches:
            step_header = match.group(1)
            step_content = match.group(2)
            
            # Check if this step has the requested ID
            id_pattern = r"\* \*\*ID:\*\* `(.*?)`"
            id_match = re.search(id_pattern, step_content)
            
            if id_match and id_match.group(1) == step_id:
                updated_content = step_content
                
                # Update each field
                for field_name, field_value in fields.items():
                    field_pattern = fr"\* \*\*{re.escape(field_name)}:\*\* .*?(?=\n\* \*\*|\n```|\Z)"
                    replacement = f"* **{field_name}:** {field_value}"
                    
                    if re.search(field_pattern, updated_content, re.DOTALL):
                        # Update existing field
                        updated_content = re.sub(
                            field_pattern,
                            replacement,
                            updated_content,
                            flags=re.DOTALL
                        )
                    else:
                        # Add new field (before code blocks)
                        code_pos = updated_content.find("```python")
                        if code_pos >= 0:
                            updated_content = (
                                updated_content[:code_pos] + 
                                f"{replacement}\n" + 
                                updated_content[code_pos:]
                            )
                        else:
                            updated_content += f"\n{replacement}"
                
                # Update code block if provided
                if code is not None:
                    code_pattern = r"(```python\n).*?(\n```)"
                    if re.search(code_pattern, updated_content, re.DOTALL):
                        updated_content = re.sub(
                            code_pattern,
                            fr"\1{code}\2",
                            updated_content,
                            flags=re.DOTALL
                        )
                    else:
                        # Add code block if not present
                        updated_content += f"\n```python\n{code}\n```"
                
                # Update execution results if provided
                if execution_results is not None:
                    results_pattern = r"(```text\n).*?(\n```)"
                    if re.search(results_pattern, updated_content, re.DOTALL):
                        updated_content = re.sub(
                            results_pattern,
                            fr"\1{execution_results}\2",
                            updated_content,
                            flags=re.DOTALL
                        )
                    else:
                        # Add execution results block if not present
                        updated_content += f"\n```text\n{execution_results}\n```"
                
                # Replace the step in the full content
                full_step = step_header + updated_content
                markdown_content = markdown_content.replace(
                    match.group(0),
                    full_step
                )
                
                _write_markdown_file(markdown_file, markdown_content)
                logfire.debug(f"Updated step '{step_id}' with {len(fields)} fields")
                
                if code is not None:
                    logfire.debug(f"Updated code for step '{step_id}'")
                    
                if execution_results is not None:
                    logfire.debug(f"Updated execution results for step '{step_id}'")
                
                return True
        
        logfire.warning(f"Step '{step_id}' not found in {markdown_file}")
        return False


def _get_step_info(workflow_file: Path, step_id: str) -> Optional[StepInfo]:
    """Get information about a specific step from the workflow markdown file.
    
    Args:
        workflow_file: Path to the workflow markdown file
        step_id: ID of the step to find (e.g., step_1)
        
    Returns:
        StepInfo object if found, None otherwise
    """
    with logfire.span("get_step_info", step_id=step_id):
        if not workflow_file.exists():
            logfire.error(f"Workflow file not found: {workflow_file}")
            return None
            
        with open(workflow_file, "r") as f:
            content = f.read()
            
        # Find the step section
        step_pattern = f"### Step \\d+: .*?\\n\\n(.*?)(?=\\n### |\\n## |\\Z)"
        step_matches = list(re.finditer(step_pattern, content, re.DOTALL))
        
        for match in step_matches:
            step_content = match.group(1)
            
            # Check if this is the step we're looking for
            id_pattern = r"\* \*\*ID:\*\* `(.*?)`"
            id_match = re.search(id_pattern, step_content)
            
            if id_match and id_match.group(1) == step_id:
                # Extract step information
                title_pattern = r"### Step \d+: (.*?)\\n"
                title_match = re.search(title_pattern, content[:match.start()])
                
                desc_pattern = r"\* \*\*Description:\*\* (.*?)\\n"
                desc_match = re.search(desc_pattern, step_content)
                
                status_pattern = r"\* \*\*Status:\*\* (.*?)\\n"
                status_match = re.search(status_pattern, step_content)
                
                input_pattern = r"\* \*\*Input Audio:\*\* `(.*?)`"
                input_match = re.search(input_pattern, step_content)
                
                output_pattern = r"\* \*\*Output Audio:\*\* `(.*?)`"
                output_match = re.search(output_pattern, step_content)
                
                code_pattern = r"\* \*\*Code:\*\*\s*```python\s*(.*?)\s*```"
                code_match = re.search(code_pattern, step_content, re.DOTALL)
                
                results_pattern = r"\* \*\*Execution Results:\*\*\s*```text\s*(.*?)\s*```"
                results_match = re.search(results_pattern, step_content, re.DOTALL)
                
                return StepInfo(
                    id=step_id,
                    title=title_match.group(1) if title_match else "Unknown Step",
                    description=desc_match.group(1) if desc_match else "",
                    status=status_match.group(1) if status_match else "PENDING",
                    input_audio=input_match.group(1) if input_match else "",
                    output_audio=output_match.group(1) if output_match else "",
                    code=code_match.group(1).strip() if code_match else None,
                    execution_results=results_match.group(1).strip() if results_match else None
                )
                
        logfire.warning(f"Step {step_id} not found in workflow file")
        return None


def _create_workflow_file(workflow_file: Path, task_description: str, input_audio_name: str, transcript: str = "") -> None:
    """Create and initialize a new workflow markdown file.
    
    Args:
        workflow_file: Path to the workflow file to create
        task_description: Description of the processing task
        input_audio_name: Name of the input audio file
        transcript: Transcript of the audio (if available)
    """
    with logfire.span("create_workflow_file", file=str(workflow_file)):
        # Create basic markdown template
        template = f"""# Audio Processing Workflow

## 1. Task Description

{task_description}

## 2. Product Requirements Document (PRD)

*This section will be generated by the Planner Agent.*

## 3. Processing Plan & Status

*This section will contain the processing steps generated by the Planner Agent.*

## 4. Input Information

* **Input Audio:** `{input_audio_name}`
* **Timestamp:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

"""
        # Add transcript if available
        if transcript:
            template += f"""## 5. Audio Transcript

```
{transcript}
```

"""

        # Add log section
        template += """## Workflow Log

*This section contains logs of the workflow execution.*

"""
        # Write the file
        with open(workflow_file, "w") as f:
            f.write(template)
            
        logfire.info(f"Created workflow file: {workflow_file}")


def _load_audio_content(audio_path: Path) -> Optional[AudioContent]:
    """Load audio content for direct model access.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        AudioContent object with the audio data loaded, or None if loading fails
    """
    with logfire.span("load_audio_content", path=str(audio_path)):
        try:
            if not audio_path.exists():
                logfire.error(f"Audio file not found: {audio_path}")
                return None
                
            # Create an AudioContent object with the file path
            audio_content = AudioContent(file_path=audio_path)
            
            # Load the binary data
            audio_data = audio_path.read_bytes()
            
            # Set the content field using BinaryContent
            audio_content.content = BinaryContent(
                data=audio_data,
                media_type='audio/wav'  # Assuming WAV format, could be made more dynamic
            )
            
            logfire.info(f"Loaded audio content from {audio_path} ({len(audio_data)} bytes)")
            return audio_content
            
        except Exception as e:
            logfire.error(f"Failed to load audio content from {audio_path}: {e}")
            return None


class AudioProcessingCoordinator:
    """
    Coordinator for the multi-agent audio processing system.
    Orchestrates planning, code generation, and execution.
    """
    def __init__(
        self, 
        working_dir: str, 
        model_name: str = "gemini-2.0-flash",
        interactive: bool = True,
        enable_error_analyzer: bool = True,
        enable_qa: bool = True,
        enable_audio_content: bool = True
    ):
        """Initialize the coordinator."""
        self.working_dir = Path(working_dir)
        self.model_name = model_name
        self.interactive = interactive
        self.enable_error_analyzer = enable_error_analyzer
        self.enable_qa = enable_qa
        self.enable_audio_content = enable_audio_content
        
        # Create necessary directories
        self.docs_dir = self.working_dir / "docs"
        self.audio_dir = self.working_dir / "audio"
        self.docs_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        # Set up user feedback handler if interactive
        if interactive:
            self.user_feedback_handler = ConsoleUserFeedbackHandler()
        else:
            self.user_feedback_handler = None
        
        # For error and usage tracking
        self.error_count = 0
        self.total_requests = 0
        self.retry_limit = 3
        
        # Gather tool definitions
        self.tool_definitions = {}
        for name, func in inspect.getmembers(audio_tools):
            if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                doc = inspect.getdoc(func) or "No description available."
                first_line_doc = doc.splitlines()[0].strip()
                
                try:
                    sig = str(inspect.signature(func))
                    sig = sig.replace("NoneType", "None")
                    full_sig = f"{name}{sig}"
                except ValueError:
                    full_sig = f"{name}(...)"  # Fallback
                    
                self.tool_definitions[name] = {
                    "signature": full_sig,
                    "description": first_line_doc,
                    "docstring": doc
                }
        
        self.usage = Usage()
        self.usage_limits = UsageLimits(request_limit=25)
        
        logfire.info(f"AudioProcessingCoordinator initialized with {len(self.tool_definitions)} tools.")
        logfire.info(f"Feature flags: error_analyzer={enable_error_analyzer}, qa={enable_qa}, interactive={interactive}")
    
    async def run_workflow(
        self, 
        task_description: str, 
        input_audio_path: str, 
        transcript: str = ""
    ) -> str:
        """
        Run the audio processing workflow.
        
        Args:
            task_description: Description of the processing task
            input_audio_path: Path to the input audio file
            transcript: Transcript of the audio (if available)
            
        Returns:
            Path to the output audio file
        """
        # Create unique workflow ID based on inputs
        task_hash = hashlib.md5((task_description + input_audio_path).encode()).hexdigest()[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_id = f"workflow_{timestamp}_{task_hash}"
        
        # Create workflow markdown file
        workflow_file = self.docs_dir / f"{workflow_id}.md"
        
        # Copy the input audio to the workspace
        input_audio = Path(input_audio_path)
        workspace_input_audio = self.audio_dir / input_audio.name
        
        import shutil
        shutil.copy(input_audio, workspace_input_audio)
        
        # Initialize the workflow file
        _create_workflow_file(
            workflow_file, 
            task_description, 
            workspace_input_audio.name, 
            transcript
        )
        
        # Load audio content for direct model access if enabled
        audio_content = None
        if self.enable_audio_content:
            audio_content = _load_audio_content(workspace_input_audio)
            if audio_content:
                logfire.info(f"Loaded audio content from {workspace_input_audio} for direct model access")
            else:
                logfire.warning(f"Failed to load audio content from {workspace_input_audio}")
        
        # Create dependencies for the coordinator agent
        deps = WorkflowState(
            workspace_dir=self.working_dir,
            workflow_file=workflow_file,
            task_description=task_description,
            original_audio=workspace_input_audio,
            tool_definitions=self.tool_definitions
        )
        
        # Run the coordinator agent
        try:
            # Use the updated coordinator_agent with RunContext
            result = await coordinator_agent.run(
                f"""
                Process the audio file according to the task:
                "{task_description}"
                
                The audio file is located at: {workspace_input_audio}
                
                Follow these steps:
                1. Generate a plan using the Planner Agent
                2. For each step in the plan:
                   a. Generate code to execute the step
                   b. Execute the code to process the audio
                3. Return the final processed audio file
                """,
                deps=deps
            )
            
            # Handle the result (might be an AgentRunResult or a string)
            if hasattr(result, 'data') and isinstance(result.data, ProcessingResult):
                # Get the path from the ProcessingResult
                final_output_path = result.data.output_path
                logfire.info(f"Processing completed successfully: {final_output_path}")
                return self.working_dir / final_output_path
            else:
                # If it's a string or another type, try to extract the path
                if isinstance(result, str):
                    logfire.warning(f"Agent returned string instead of ProcessingResult: {result}")
                    # Try to find a path in the string
                    import re
                    path_match = re.search(r'(?:output|final|result).*?(?:path|file).*?[\'"]([^\'"]+)[\'"]', result, re.IGNORECASE)
                    if path_match:
                        path = path_match.group(1)
                        return self.working_dir / path
                
                # Fallback: Look for the most recently modified audio file in the workspace
                audio_files = list(self.audio_dir.glob("*.wav"))
                if audio_files:
                    most_recent = max(audio_files, key=os.path.getmtime)
                    logfire.warning(f"Using most recent audio file as result: {most_recent}")
                    return most_recent
                
                # If all else fails, return the input audio
                logfire.error("Failed to determine output path, returning input audio")
                return workspace_input_audio
        
        except Exception as e:
            logfire.error(f"Error running coordinator agent: {e}", exc_info=True)
            raise 