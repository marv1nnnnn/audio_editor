# Pydantic AI Refactoring

This document explains the refactoring of the audio_editor project to use Pydantic AI and Logfire.

## Overview

The audio processing system has been refactored to use:

1. **Pydantic AI Agents**: For structured LLM interactions with proper typing
2. **Pydantic Models**: For robust data validation
3. **Logfire**: For comprehensive instrumentation and debugging
4. **Markdown-Centric Workflow**: Maintained but enhanced with structured models

## Key Changes

### 1. Structured Models with Pydantic

All data structures are now defined as Pydantic models:

```python
class StepInfo(BaseModel):
    """Information about a processing step."""
    id: str = Field(..., description="Unique identifier for the step (e.g., step_1)")
    title: str = Field(..., description="Brief title for the step")
    description: str = Field(..., description="Detailed description of what the step should accomplish")
    status: str = Field(default="PENDING", description="Current status of the step")
    # ...
```

This provides:
- Automatic validation
- Self-documenting schema
- Better IDE support
- Consistent error handling

### 2. Agent-Based Architecture

The system now uses Pydantic AI agents with well-defined roles:

```python
# Planner agent
planner_agent = Agent(
    'gemini-2.0-flash',
    deps_type=WorkflowState,
    result_type=PlanResult,
    system_prompt="..."
)

# Code generation agent
code_gen_agent = Agent(
    'gemini-2.0-flash',
    deps_type=WorkflowState, 
    result_type=CodeGenerationResult,
    system_prompt="..."
)
```

### 3. Dependency Injection

Dependencies are explicitly defined and injected:

```python
class WorkflowState(BaseModel):
    """State of the workflow, used for dependency injection."""
    workspace_dir: str
    workflow_file: str
    current_step_id: Optional[str] = None
    task_description: str = ""
    # ...
```

Tool functions have access to dependencies through the `RunContext`:

```python
@coordinator_agent.tool
async def generate_code_for_step(
    ctx: RunContext[WorkflowState],
    step_id: str
) -> str:
    # Access deps with ctx.deps.workspace_dir, etc.
```

### 4. Logfire Instrumentation

Comprehensive logging with spans for performance monitoring:

```python
with logfire.span("execute_code", description=description):
    # Code execution logic
    logfire.info(f"Executing {func_name} with args: {kwargs}")
```

### 5. Error Handling and Retries

More robust error handling using `ModelRetry`:

```python
# Try to analyze and fix the error
fixed = await analyze_and_fix_error(ctx, step_id, code, result.error_message)
if fixed:
    # Reset the step to READY
    _update_step_fields(ctx.deps.workflow_file, step_id, {"Status": "READY"})
    # Raise ModelRetry to let the agent know it should retry
    raise ModelRetry(f"Fixed error in step {step_id}. Retrying...")
```

## Usage

The system can be used in the same way as before:

```bash
python -m audio_editor.agents.main --task "Normalize audio and add reverb" --input input.wav
```

New flags:
- `--legacy`: Use the old processing method (not using Markdown workflow)
- `--log-level`: Set logging level (debug/info/warning/error)

## Implementation Details

### Key Files

1. `coordinator.py`: Contains the main agent definitions and tools
2. `models.py`: Contains all Pydantic models
3. `dependencies.py`: Defines dependency injection structures
4. `mcp.py`: Enhanced MCP for code execution with better error handling
5. `main.py`: Updated CLI interface

### Workflow Changes

The Markdown-centric workflow is maintained but with enhanced internal structure:
- Better parsing and error handling
- Stronger typing
- More detailed logging
- More robust error recovery

## Benefits

1. **Type Safety**: All code is properly typed and validated
2. **Debugging**: Comprehensive Logfire instrumentation
3. **Robustness**: Better error handling and retries
4. **Maintainability**: Clear structure and self-documenting code
5. **Extensibility**: Easy to add new tools and capabilities 