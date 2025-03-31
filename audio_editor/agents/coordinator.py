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
from typing import Dict, List, Optional, Union, Any
import json

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.usage import Usage, UsageLimits

from audio_editor import audio_tools
from .mcp import MCPCodeExecutor
from .user_feedback import ConsoleUserFeedbackHandler


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


class ExecutionResult(BaseModel):
    """Result of executing code."""
    status: str
    output: str = ""
    error_message: str = ""
    output_path: Optional[str] = None
    output_paths: Optional[List[str]] = None
    duration: float = 0.0


class StepInfo(BaseModel):
    """Information about a processing step."""
    id: str
    description: str
    status: str
    input_audio: str
    output_audio: str
    code: Optional[str] = None
    execution_results: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None


class WorkflowState(BaseModel):
    """State of the workflow, used for dependency injection."""
    workspace_dir: Path
    workflow_file: Path
    current_step_id: Optional[str] = None
    task_description: str = ""
    original_audio: Path = ""
    tool_definitions: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    current_markdown: str = ""


class PlanResult(BaseModel):
    """Result from the planner agent."""
    prd: str = Field(..., description="The Product Requirements Document")
    steps: List[Dict[str, str]] = Field(..., description="The list of processing steps")


class CodeGenerationResult(BaseModel):
    """Result from the code generation agent."""
    code: str = Field(..., description="Generated Python code for the step")


class ProcessingResult(BaseModel):
    """Final result of the audio processing."""
    output_path: str = Field(..., description="Path to the final processed audio")
    status: str = Field(..., description="Overall processing status")
    steps_completed: int = Field(..., description="Number of completed steps")


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
    
    Each step should have:
    - A descriptive title
    - A unique ID (step_1, step_2, etc.)
    - A detailed description of what it should accomplish
    - Input and output paths
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
    3. Use the correct input/output paths for audio files
    
    You must return exactly one line of executable Python code.
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
def add_step_context(ctx: RunContext[WorkflowState]) -> str:
    """Add context about the current step."""
    if not ctx.deps.current_step_id:
        return ""
    return f"Currently generating code for step: {ctx.deps.current_step_id}"


# Tools for the coordinator agent
@coordinator_agent.tool
async def create_workflow_markdown(
    ctx: RunContext[WorkflowState],
    task_description: str,
    audio_path: str,
    workflow_id: str
) -> str:
    """Create the initial Markdown workflow file."""
    with logfire.span("create_workflow_markdown"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Audio Processing Workflow: {workflow_id}

## 1. Request Summary

* **Original Request:**
    ```text
    {task_description}
    ```
* **Input Audio:** `{audio_path}`
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
        
        # Call the planner agent
        planner_result = await planner_agent.run(
            ctx.deps.task_description,
            deps=ctx.deps
        )
        
        # Update the workflow file with the PRD and plan
        markdown_content = _read_markdown_file(ctx.deps.workflow_file)
        
        # Replace the placeholder sections
        updated_content = re.sub(
            r"## 2\. Product Requirements Document \(PRD\)\n\n\*This section will be generated by the Planner Agent\.\*",
            f"## 2. Product Requirements Document (PRD)\n\n{planner_result.data.prd}",
            markdown_content
        )
        
        # Create the steps content
        steps_content = "\n\n".join([
            f"### Step {i+1}: {step['title']}\n\n"
            f"* **ID:** `{step['id']}`\n"
            f"* **Description:** {step['description']}\n"
            f"* **Status:** READY\n"
            f"* **Input Audio:** `{step['input_audio']}`\n"
            f"* **Output Audio:** `{step['output_audio']}`\n"
            f"* **Code:**\n```python\n# Placeholder - will be generated by Code Generator\n```\n"
            f"* **Execution Results:**\n```text\n# Placeholder - will be filled by Executor\n```\n"
            f"* **Timestamp Start:** `N/A`\n"
            f"* **Timestamp End:** `N/A`"
            for i, step in enumerate(planner_result.data.steps)
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
        _append_workflow_log(ctx.deps.workflow_file, f"Generated PRD and processing plan with {len(planner_result.data.steps)} steps.")
        
        return planner_result.data


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
        
        # Update context with current step
        updated_deps = ctx.deps.model_copy()
        updated_deps.current_step_id = step_id
        
        # Call the code generation agent
        logfire.info(f"Generating code for step {step_id}")
        code_result = await code_gen_agent.run(
            f"Generate code for step {step_id} in the workflow",
            deps=updated_deps
        )
        
        generated_code = code_result.data.code
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
            """
        )
        
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
        
        Available tools:
        {json.dumps(ctx.deps.tool_definitions, indent=2)}
        
        Return ONLY the fixed code, without any explanation or formatting.
        """
        
        try:
            # Call the error analyzer
            logfire.info(f"Analyzing error in step {step_id}")
            result = await error_analyzer.run(prompt)
            
            # Extract the fixed code
            fixed_code = result.data.strip()
            
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


class AudioProcessingCoordinator:
    """
    Markdown-centric audio processing coordinator using Pydantic AI.
    This class provides a backwards-compatible API while using the new implementation.
    """
    
    def __init__(
        self, 
        working_dir: Path | str, 
        model_name: str = "gemini-2.0-flash",
        interactive: bool = True,
        enable_error_analyzer: bool = True,
        enable_qa: bool = True
    ):
        """Initialize the coordinator."""
        self.working_dir = Path(working_dir).absolute()
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.docs_dir = self.working_dir / "docs"
        os.makedirs(self.docs_dir, exist_ok=True)
        
        self.model_name = model_name
        self.interactive = interactive
        self.enable_error_analyzer = enable_error_analyzer
        self.enable_qa = enable_qa
        
        # Create user feedback handler (for backward compatibility)
        self.feedback_handler = ConsoleUserFeedbackHandler(interactive=interactive)
        
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
        audio_file_path: str,
        audio_transcript: str = ""
    ) -> str:
        """Process audio using the Pydantic AI-based Markdown-centric workflow."""
        with logfire.span("run_workflow", task=task_description, file=audio_file_path):
            start_time = time.time()
            
            # Copy input file to working directory
            input_basename = Path(audio_file_path).name
            working_input = self.working_dir / input_basename
            
            # Copy file if not already in working directory
            if Path(audio_file_path).absolute() != working_input.absolute():
                import shutil
                shutil.copy(audio_file_path, working_input)
                logfire.info(f"Copied {audio_file_path} to {working_input}")
            
            # Create a workflow ID and workflow markdown file
            workflow_id = f"workflow_{int(time.time())}_{hashlib.md5(task_description.encode()).hexdigest()[:8]}"
            workflow_file = self.docs_dir / f"{workflow_id}.md"
            
            # Set up the workflow state for dependency injection
            workflow_state = WorkflowState(
                workspace_dir=self.working_dir,
                workflow_file=workflow_file,
                task_description=task_description,
                original_audio=working_input,
                tool_definitions=self.tool_definitions
            )
            
            try:
                # Run the coordinator agent
                result = await coordinator_agent.run(
                    f"Process audio file according to this request: {task_description}",
                    deps=workflow_state,
                    usage=self.usage,
                    usage_limits=self.usage_limits
                )
                
                # Return the final output path
                final_output_path = result.data.output_path
                logfire.info(f"Audio processing completed in {time.time() - start_time:.2f}s")
                return final_output_path
                
            except Exception as e:
                logfire.error(f"Audio processing failed: {str(e)}", exc_info=True)
                raise 