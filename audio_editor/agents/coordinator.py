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
from .prompts import (
    DEFAULT_AUDIO_QUALITY_PROMPT,
    DEFAULT_AUDIO_COMPARISON_PROMPT,
    DEFAULT_MULTI_COMPARISON_PROMPT,
    DEFAULT_AUDIO_GENERATION_PROMPT,
    ITERATIVE_IMPROVEMENT_TEMPLATE,
    FINAL_QUALITY_VERIFICATION_PROMPT
)


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


# Enhance models for quality assessment
class QualityAssessmentResult(BaseModel):
    """Result of audio quality assessment."""
    assessment: str
    score: Optional[float] = None
    recommendations: List[str] = Field(default_factory=list)
    
class AudioComparisonResult(BaseModel):
    """Result of comparing audio files."""
    comparison: str
    improvements: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    preferred_file: Optional[str] = None


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
    3. Include quality validation steps after significant processing
    4. Consider reference audio generation when it would help guide processing

    Each step must have:
    - A descriptive title
    - A unique ID (step_1, step_2, etc.)
    - A detailed description of what it should accomplish
    - Input and output paths
    - Step type: "processing", "validation", or "reference"

    For validation steps:
    - Add these after significant processing operations
    - Use "validation" as the step type
    - The output path should match the input path (validation doesn't modify audio)
    - Describe what specific aspects of quality to validate

    For reference generation steps:
    - Use "reference" as the step type
    - Describe the characteristics the reference should have
    - Name the output with a descriptive name like "professional_reference.wav"
    
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
    * **Step Type:** `processing`
    * **Input Audio:** `sample1.wav`  # MUST match deps.original_audio.name
    * **Output Audio:** `highpass_filtered.wav`

    ### Step 2: Validate High-Pass Filter Results

    * **ID:** `step_2`
    * **Description:** Analyze the frequency content to ensure rumble is removed without affecting vocal clarity
    * **Step Type:** `validation`
    * **Input Audio:** `highpass_filtered.wav`  # Match previous step's output
    * **Output Audio:** `highpass_filtered.wav`  # Same as input for validation steps

    ### Step 3: Normalize Volume

    * **ID:** `step_3`
    * **Description:** Normalize the audio to broadcast standard
    * **Step Type:** `processing`
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
    1. Generate Python code to implement a specific step in the audio processing workflow
    2. Use the correct tool from the available tool definitions
    3. Use the EXACT file paths from the step information
    4. Generate different code based on the step type (processing, validation, or reference)
    
    Rules for code generation:
    1. Use ONLY the tools provided in the tool definitions
    2. Each tool takes a wav_path input and usually returns an output path
    3. ALWAYS use the EXACT input_audio path from the step info - do not modify or assume paths
    4. ALWAYS use the EXACT output_audio path from the step info - do not modify or assume paths
    5. Include any necessary parameters based on the tool's signature
    6. Return executable Python code - a single line for processing steps or multiple lines for analysis
    7. Do not add any imports or other statements not needed for execution
    8. ALWAYS use keyword arguments (e.g., wav_path="sample_1.wav") instead of positional arguments
    9. NEVER assume or hardcode file paths - they must come from the step info
    
    FOR DIFFERENT STEP TYPES:
    
    For "processing" steps:
    - Return a single line of code using a processing function like LOUDNESS_NORM
    - Example: LOUDNESS_NORM(wav_path="input.wav", volume=-20.0, out_wav="output.wav")
    
    For "validation" steps:
    - Use AUDIO_QA to analyze the audio quality
    - Provide a specific prompt relevant to the validation goal
    - Example: AUDIO_QA(wav_path="processed.wav", task="Analyze the frequency balance and identify any issues in the low end")
    
    For "reference" steps:
    - Use AUDIO_GENERATE to create reference audio
    - Provide a detailed description based on the step requirements
    - Example: AUDIO_GENERATE(text="Professional quality music with clear vocals and balanced frequency response", filename="reference.wav", audio_length_in_s=10.0)
    
    For comparison steps:
    - Use AUDIO_DIFF to compare multiple audio files
    - Provide specific comparison instructions
    - Example: AUDIO_DIFF(wav_paths=["original.wav", "processed.wav"], task="Compare the original and processed files and identify improvements and remaining issues")
    
    The MCP executor will automatically handle placing the files in the correct audio directory.
    You should NOT include the audio directory in the paths you generate.
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

        # Extract the exact path strings needed
        input_path_value = step_info.input_audio
        output_path_value = step_info.output_audio
        
        # Get the step type (default to "processing" for backward compatibility)
        step_type = getattr(step_info, "step_type", "processing")

        if not input_path_value:
            logfire.error(f"Error: Input audio path missing in step info for {step_info.id}")
            return "Error: Input audio path missing in step info"
        if not output_path_value:
            logfire.error(f"Error: Output audio path missing in step info for {step_info.id}")
            return "Error: Output audio path missing in step info"

        # Format tool definitions
        tool_defs = []
        for name, info in ctx.deps.tool_definitions.items():
            tool_defs.append(f"Tool: {name}")
            tool_defs.append(f"Signature: {info['signature']}")
            tool_defs.append(f"Description: {info['description']}")
            tool_defs.append("")

        tools_str = "\n".join(tool_defs)
        
        # Add appropriate prompt suggestions for different step types
        prompt_suggestion = ""
        if step_type == "validation":
            prompt_suggestion = f"""
            For this validation step, consider using one of these prompt templates:
            
            Quality Analysis:
            ```
            {DEFAULT_AUDIO_QUALITY_PROMPT[:200]}...
            ```
            
            Or for specific validation focused on step purpose:
            "Analyze this audio and verify that {step_info.description}"
            """
        elif step_type == "reference":
            prompt_suggestion = f"""
            For this reference generation step, consider using this prompt template:
            
            ```
            {DEFAULT_AUDIO_GENERATION_PROMPT[:200]}...
            ```
            
            Or a more specific prompt based on the step requirements:
            "Generate audio that {step_info.description}"
            """
        elif "compare" in step_info.description.lower() or "diff" in step_info.description.lower():
            prompt_suggestion = f"""
            For this comparison step, consider using this prompt template:
            
            ```
            {DEFAULT_AUDIO_COMPARISON_PROMPT[:200]}...
            ```
            """

        return f"""
Current step information:
- ID: {step_info.id}
- Description: {step_info.description}
- Step Type: {step_type}
- Input Audio: {step_info.input_audio}
- Output Audio: {step_info.output_audio}

Available tools:
{tools_str}

Task description: {ctx.deps.task_description}

{prompt_suggestion}

CRITICAL INSTRUCTIONS FOR CODE GENERATION:
1. You MUST generate code appropriate for a {step_type.upper()} step.
2. The code MUST call one of the available tools.
3. The code MUST use keyword arguments (e.g., wav_path="...").
4. **Use the EXACT input file path: '{input_path_value}'** for the 'wav_path' argument (or the primary input argument if named differently).
5. **Use the EXACT output file path: '{output_path_value}'** for the 'out_wav' argument (or the primary output argument if named differently).
6. Include other necessary parameters based on the tool's signature and the step description.

Now, generate the code for step {step_info.id} ({step_type} step) using input '{input_path_value}' and output '{output_path_value}'.
"""

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


@coordinator_agent.tool
async def assess_audio_quality(
    ctx: RunContext[WorkflowState],
    audio_path: str,
    specific_question: Optional[str] = None
) -> QualityAssessmentResult:
    """Assess the quality of an audio file using AUDIO_QA."""
    with logfire.span("assess_audio_quality"):
        if not specific_question:
            specific_question = "Analyze this audio for quality, balance, and professional sound. What issues need fixing?"
            
        # Get the full path to the audio file
        full_audio_path = ctx.deps.workspace_dir / "audio" / audio_path
        
        if not full_audio_path.exists():
            error_msg = f"Audio file not found: {full_audio_path}"
            logfire.error(error_msg)
            return QualityAssessmentResult(
                assessment=f"Error: {error_msg}",
                recommendations=["Ensure the audio file exists before assessment"]
            )
            
        try:
            # Call AUDIO_QA function
            logfire.info(f"Assessing audio quality for {full_audio_path}")
            assessment = audio_tools.AUDIO_QA(str(full_audio_path), specific_question)
            
            # Parse the assessment to extract recommendations
            recommendations = []
            for line in assessment.split('\n'):
                if any(kw in line.lower() for kw in ['recommend', 'suggest', 'should', 'improve', 'enhance']):
                    recommendations.append(line.strip())
                    
            # Add to workflow log
            _append_workflow_log(ctx.deps.workflow_file, f"Assessed audio quality for {audio_path}")
            
            return QualityAssessmentResult(
                assessment=assessment,
                recommendations=recommendations[:5]  # Limit to top 5 recommendations
            )
        except Exception as e:
            error_msg = f"Error assessing audio quality: {str(e)}"
            logfire.error(error_msg)
            return QualityAssessmentResult(
                assessment=error_msg,
                recommendations=["Try a different analysis approach"]
            )

@coordinator_agent.tool
async def compare_audio_files(
    ctx: RunContext[WorkflowState],
    original_audio: str,
    processed_audio: str,
    comparison_focus: Optional[str] = None
) -> AudioComparisonResult:
    """Compare original and processed audio files using AUDIO_DIFF."""
    with logfire.span("compare_audio_files"):
        # Get the full paths to the audio files
        original_path = ctx.deps.workspace_dir / "audio" / original_audio
        processed_path = ctx.deps.workspace_dir / "audio" / processed_audio
        
        if not original_path.exists() or not processed_path.exists():
            missing = []
            if not original_path.exists():
                missing.append(f"Original audio: {original_path}")
            if not processed_path.exists():
                missing.append(f"Processed audio: {processed_path}")
                
            error_msg = f"Audio file(s) not found: {', '.join(missing)}"
            logfire.error(error_msg)
            return AudioComparisonResult(
                comparison=f"Error: {error_msg}",
                improvements=[],
                issues=["Missing audio files"]
            )
            
        try:
            # Create comparison focus if not provided
            if not comparison_focus:
                comparison_focus = """
                Compare these audio files and analyze the differences:
                1. What improvements were made in the processed file?
                2. What issues were fixed?
                3. What new issues might have been introduced?
                4. Which file sounds better overall and why?
                5. What should be the next processing step to further improve?
                """
                
            # Call AUDIO_DIFF function
            logfire.info(f"Comparing audio files: {original_path} and {processed_path}")
            comparison = audio_tools.AUDIO_DIFF(
                [str(original_path), str(processed_path)],
                comparison_focus
            )
            
            # Parse the comparison to extract improvements and issues
            improvements = []
            issues = []
            preferred_file = None
            
            lines = comparison.split('\n')
            for i, line in enumerate(lines):
                # Detect improvements
                if any(kw in line.lower() for kw in ['improve', 'better', 'enhanced', 'fixed']):
                    improvements.append(line.strip())
                # Detect issues
                if any(kw in line.lower() for kw in ['issue', 'problem', 'worse', 'degraded']):
                    issues.append(line.strip())
                # Detect preferred file
                if 'better' in line.lower() and ('file' in line.lower() or 'audio' in line.lower()):
                    if 'original' in line.lower() or 'first' in line.lower():
                        preferred_file = original_audio
                    elif 'processed' in line.lower() or 'second' in line.lower():
                        preferred_file = processed_audio
            
            # Add to workflow log
            _append_workflow_log(ctx.deps.workflow_file, f"Compared audio files: {original_audio} vs {processed_audio}")
            
            return AudioComparisonResult(
                comparison=comparison,
                improvements=improvements[:3],  # Limit to top 3
                issues=issues[:3],  # Limit to top 3
                preferred_file=preferred_file
            )
        except Exception as e:
            error_msg = f"Error comparing audio files: {str(e)}"
            logfire.error(error_msg)
            return AudioComparisonResult(
                comparison=error_msg,
                improvements=[],
                issues=["Error during comparison"]
            )

@coordinator_agent.tool
async def generate_audio_example(
    ctx: RunContext[WorkflowState],
    description: str,
    output_filename: str,
    duration: float = 5.0
) -> str:
    """Generate an audio example based on a text description."""
    with logfire.span("generate_audio_example"):
        # Ensure output filename has .wav extension
        if not output_filename.lower().endswith(".wav"):
            output_filename += ".wav"
            
        # Get the full path for the output file
        output_path = ctx.deps.workspace_dir / "audio" / output_filename
        
        try:
            # Call AUDIO_GENERATE function
            logfire.info(f"Generating audio example: {description}")
            result_path = audio_tools.AUDIO_GENERATE(
                description,
                str(output_path),
                audio_length_in_s=duration
            )
            
            # Add to workflow log
            _append_workflow_log(
                ctx.deps.workflow_file,
                f"Generated audio example: '{description}' -> {output_filename}"
            )
            
            return output_filename
        except Exception as e:
            error_msg = f"Error generating audio example: {str(e)}"
            logfire.error(error_msg)
            return f"Error: {error_msg}"

# Enhance the execute_code_for_step to include quality assessment
@coordinator_agent.tool
async def execute_code_for_step_with_validation(
    ctx: RunContext[WorkflowState],
    step_id: str,
    code: str,
    validate_output: bool = True
) -> ExecutionResult:
    """Execute the code for a specific step and validate the output quality."""
    with logfire.span("execute_code_for_step_with_validation", step_id=step_id):
        # Get the original execution result
        execution_result = await execute_code_for_step(ctx, step_id, code)
        
        # If execution failed or validation not requested, return the result
        if execution_result.status != "SUCCESS" or not validate_output:
            return execution_result
            
        # Get step info to access input/output paths
        step_info = _get_step_info(ctx.deps.workflow_file, step_id)
        if not step_info or not step_info.output_audio:
            logfire.warning(f"Cannot validate output: Missing step info or output path for step {step_id}")
            return execution_result
            
        try:
            # Assess the quality of the output audio
            logfire.info(f"Validating output quality for step {step_id}")
            assessment_result = await assess_audio_quality(ctx, step_info.output_audio)
            
            # If this isn't the first step, compare with input
            comparison_result = None
            if step_info.input_audio != ctx.deps.original_audio.name:
                logfire.info(f"Comparing input and output for step {step_id}")
                comparison_result = await compare_audio_files(
                    ctx,
                    step_info.input_audio,
                    step_info.output_audio
                )
                
            # Update the step with the validation results
            validation_text = f"Quality Assessment:\n{assessment_result.assessment}\n\n"
            if comparison_result:
                validation_text += f"Comparison with Input:\n{comparison_result.comparison}\n\n"
                
            # Add recommendations if any
            if assessment_result.recommendations:
                validation_text += "Recommendations:\n"
                for rec in assessment_result.recommendations:
                    validation_text += f"- {rec}\n"
                    
            # Update the step in the workflow markdown
            _update_step_fields(
                ctx.deps.workflow_file,
                step_id,
                {"Quality Validation": "DONE"},
                execution_results=f"{execution_result.output}\n\n--- Validation Results ---\n{validation_text}"
            )
            
            # Add to workflow log
            _append_workflow_log(
                ctx.deps.workflow_file,
                f"Validated output quality for step {step_id}"
            )
            
            # If issues detected, consider suggesting improvements
            if assessment_result.recommendations or (comparison_result and comparison_result.issues):
                logfire.info(f"Quality issues detected in step {step_id}, suggesting improvements")
                
            # Return the original execution result
            return execution_result
        except Exception as e:
            logfire.error(f"Error during quality validation: {str(e)}")
            return execution_result

# Update planner agent to include quality validation steps
@planner_agent.system_prompt
def add_quality_validation_guidance(ctx: RunContext[WorkflowState]) -> str:
    """Add guidance for including quality validation steps in the plan."""
    return """
    For more effective audio processing, consider including these types of steps in your plan:
    
    1. Quality Assessment Steps: After significant processing operations, add a step to check audio quality
    2. Comparison Steps: Compare before/after for important changes to ensure improvements
    3. Reference Generation: When appropriate, generate reference audio for comparison
    
    Example quality assessment step:
    ### Step 4: Validate Processing Quality
    
    * **ID:** `step_4`
    * **Description:** Analyze the processed audio to verify quality and identify any issues
    * **Input Audio:** `previous_step_output.wav`
    * **Output Audio:** `same_as_input.wav`  # Quality check doesn't modify the file
    
    These validation steps help ensure the audio quality meets requirements at each stage.
    """

# Enhance the main processing function
async def process_audio_with_validation(
    ctx: RunContext[WorkflowState],
    final_validation: bool = True
) -> ProcessingResult:
    """Process the audio with quality validation at key points."""
    with logfire.span("process_audio_with_validation"):
        # Generate the plan
        plan_result = await generate_plan_and_prd(ctx)
        
        # Track completed steps
        completed_steps = 0
        
        # Process each step
        while True:
            # Find the next step to process
            next_step_id = await find_next_step(ctx)
            if not next_step_id:
                # No more steps to process
                break
                
            # Generate code for the step
            ctx.deps.current_step_id = next_step_id
            generated_code = await generate_code_for_step(ctx, next_step_id)
            
            # Execute the code with validation
            try:
                result = await execute_code_for_step_with_validation(ctx, next_step_id, generated_code)
                if result.status == "SUCCESS":
                    completed_steps += 1
            except ModelRetry:
                # Retry will be handled by the coordinator agent
                continue
                
        # Get the final output path
        final_output_path = await get_final_output_path(ctx)
        
        # Perform comprehensive final validation if requested
        if final_validation and final_output_path:
            # Compare original and final output
            logfire.info("Performing final quality validation")
            
            # Get original audio name
            original_audio_name = ctx.deps.original_audio.name
            
            # Assess final output quality
            final_assessment = await assess_audio_quality(
                ctx, 
                final_output_path,
                "Evaluate the final processed audio. Does it meet professional standards? What are its strengths and weaknesses?"
            )
            
            # Compare with original
            final_comparison = await compare_audio_files(
                ctx,
                original_audio_name,
                final_output_path,
                "Compare the original and final processed audio. What specific improvements were made? Are there any remaining issues?"
            )
            
            # Update final output section with assessment
            _update_workflow_section(
                ctx.deps.workflow_file,
                "Final Output",
                {
                    "Quality Assessment": "DONE",
                    "Comparison with Original": "DONE"
                }
            )
            
            # Add detailed assessment to the workflow
            final_validation_section = f"""
## 6. Final Quality Assessment

### Audio Quality Analysis
{final_assessment.assessment}

### Comparison with Original
{final_comparison.comparison}

### Key Improvements
{chr(10).join([f"* {item}" for item in final_comparison.improvements])}

### Remaining Issues
{chr(10).join([f"* {item}" for item in final_comparison.issues])}

### Overall Verdict
The final processed audio is {"better than" if final_comparison.preferred_file == final_output_path else "not clearly better than"} the original.
"""
            
            # Add the validation section to the workflow file
            markdown_content = _read_markdown_file(ctx.deps.workflow_file)
            if "## 6. Final Quality Assessment" not in markdown_content:
                _write_markdown_file(ctx.deps.workflow_file, markdown_content + final_validation_section)
            
            # Add to workflow log
            _append_workflow_log(
                ctx.deps.workflow_file,
                f"Completed final quality validation. {'Improvements detected.' if final_comparison.preferred_file == final_output_path else 'Mixed results.'}"
            )
        
        # Finish the workflow
        await finish_workflow(ctx, final_output_path, completed_steps)
        
        return ProcessingResult(
            success=True,
            output_path=final_output_path,
            steps_completed=completed_steps
        )


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
        transcript: str = "",
        validate_quality: bool = True  # Add this parameter
    ) -> str:
        """
        Run the audio processing workflow.
        
        Args:
            task_description: Description of the processing task
            input_audio_path: Path to the input audio file
            transcript: Transcript of the audio (if available)
            validate_quality: Whether to perform quality validation
            
        Returns:
            Path to the output audio file
        """
        # Create unique workflow ID based on inputs
        task_hash = hashlib.md5((task_description + input_audio_path).encode()).hexdigest()[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_id = f"workflow_{timestamp}_{task_hash}"
        
        # Create workflow markdown file
        workflow_file = self.docs_dir / f"{workflow_id}.md"
        
        # Copy the input audio to the workspace audio directory
        input_audio = Path(input_audio_path)
        workspace_input_audio = self.audio_dir / input_audio.name
        
        import shutil
        shutil.copy(input_audio, workspace_input_audio)
        logfire.info(f"Copied input audio to workspace: {workspace_input_audio}")
        
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
            # Use the updated coordinator_agent with RunContext and process_audio_with_validation
            if validate_quality:
                # Use enhanced processing with validation
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
                       c. Validate the quality of the processed audio
                    3. Perform final quality assessment
                    4. Return the final processed audio file
                    """,
                    deps=deps,
                    tools=[process_audio_with_validation]
                )
            else:
                # Use original processing without validation
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
                return final_output_path
            else:
                # If it's a string or another type, try to extract the path
                if isinstance(result, str):
                    logfire.warning(f"Agent returned string instead of ProcessingResult: {result}")
                    # Try to find a path in the string
                    import re
                    path_match = re.search(r'(?:output|final|result).*?(?:path|file).*?[\'"]([^\'"]+)[\'"]', result, re.IGNORECASE)
                    if path_match:
                        path = path_match.group(1)
                        return os.path.join(self.working_dir, 'audio', path)
                
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