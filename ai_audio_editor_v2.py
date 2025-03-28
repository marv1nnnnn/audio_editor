import os
import argparse
import json
import traceback
import asyncio
import re
import ast
import inspect
import difflib
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from time import time
from dataclasses import dataclass, field
import logfire
from google import generativeai, genai
# Need ast.unparse for robust reconstruction, requires Python 3.9+
try:
    from ast import unparse
except ImportError:
    # Basic fallback for older Python if needed, less robust
    def unparse(node):
        return ast.dump(node)
    print("Warning: Using basic ast.dump as fallback for ast.unparse. Requires Python 3.9+ for full robustness.")


# Import tools from audio_tools module
import audio_tools

# --- Configuration ---
# Consider loading from environment or config file
# os.environ["GEMINI_API_KEY"] = "YOUR_KEY_HERE"  # REPLACE WITH YOUR KEY
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Initialize logging
logfire.configure()

# Configure Gemini
if os.environ.get("GEMINI_API_KEY"):
    try:
        generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except Exception as e:
        print(f"Error configuring GenerativeAI: {e}")
        logfire.error(f"GenerativeAI configuration failed: {e}")
        # Decide if you want to exit or continue without Gemini
else:
    print("Warning: GEMINI_API_KEY environment variable not set. AI features will not work.")
    logfire.warning("GEMINI_API_KEY not set.")


# --- Data Models ---
@dataclass
class StepExecutionResult:
    """Result of executing a step in the audio processing pipeline"""
    success: bool
    message: str
    duration: float
    output_path: Optional[str] = None  # Single primary output file
    output_paths: Optional[List[str]] = None  # Multiple outputs (e.g., SPLIT)

@dataclass
class AudioProcessingPlan:
    """Plan for processing an audio file"""
    task_description: str
    steps: List[str]  # Markdown checklist steps "- [ ] ..." or "- [x] ..."
    current_audio_path: str
    completed_steps_indices: List[int] = field(default_factory=list)
    is_complete: bool = False
    clarification_question: Optional[str] = None
    user_feedback: Optional[str] = None
    original_input_path: Optional[str] = None # Store the initial input path

# --- AI Audio Editor Class ---
class AIAudioEditor:
    def __init__(self, model_name: str = "gemini-1.5-flash"): # Updated default model
        """Initialize the AI Audio Editor."""
        with logfire.span("init_audio_editor", model_name=model_name) as span:
            self.model_name = model_name
            self.available_tools = self._gather_tools()
            # Create a dedicated working directory for each run if desired, or use cwd
            # For simplicity, using cwd for now, but a unique run dir is better practice
            self.working_dir = os.path.abspath("audio_editor_work")
            os.makedirs(self.working_dir, exist_ok=True)
            logfire.info(f"Using working directory for outputs: {self.working_dir}")
            span.set_attribute("num_tools", len(self.available_tools))

    def _gather_tools(self) -> Dict[str, Callable]:
        """Gathers all callable tool functions from the audio_tools module."""
        tools = {}
        for name, func in inspect.getmembers(audio_tools):
            # Tools are uppercase functions
            if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                tools[name] = func
        logfire.info(f"Found {len(tools)} tools in audio_tools module: {list(tools.keys())}")
        return tools

    async def _call_llm(self, prompt: str, model_name: str = None, expect_json: bool = False) -> str:
        """Call Gemini and get a text/JSON response."""
        # Check if Gemini is configured
        if not os.environ.get("GEMINI_API_KEY"):
             raise ConnectionError("GEMINI_API_KEY not set. Cannot call LLM.")

        model_to_use = model_name or self.model_name
        with logfire.span("call_llm", model=model_to_use, expect_json=expect_json):
            try:
                model = generativeai.GenerativeModel(model_to_use)
                # Configure for JSON output if requested
                generation_config = None
                if expect_json:
                    generation_config = generativeai.types.GenerationConfig(
                        response_mime_type="application/json",
                    )

                logfire.debug(f"Sending prompt to {model_to_use}:\n{prompt}")
                response = await model.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )

                if not response.candidates:
                    logfire.error("LLM response blocked or empty.", response_details=str(response))
                    raise ValueError("LLM response blocked or empty.")

                response_text = response.candidates[0].content.parts[0].text
                logfire.debug(f"Received response from {model_to_use}:\n{response_text}")

                # Basic cleanup for JSON extraction if needed, but rely on mime_type
                if expect_json:
                    # response_mime_type should handle this, but add fallback cleanup
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                return response_text
            except Exception as e:
                logfire.error(f"LLM call failed: {e}\nPrompt:\n{prompt}", exc_info=True)
                raise

    async def _generate_initial_plan(self, task: str, audio_file_path: str) -> AudioProcessingPlan:
        """Generate a step-by-step plan using the LLM."""
        with logfire.span("generate_initial_plan", task=task, audio_file=audio_file_path):
            # Create tool list with signatures and docstrings
            tool_list = []
            for name, func in self.available_tools.items():
                doc = inspect.getdoc(func) or 'No description available.'
                # Extract first line of docstring if multiline
                first_line_doc = doc.splitlines()[0].strip()
                try:
                    sig = str(inspect.signature(func))
                    # Clean signature slightly for readability
                    sig = sig.replace('NoneType', 'None')
                except ValueError:
                    sig = '(...)' # Fallback for built-ins if any slipped through
                tool_list.append(f"- {name}{sig}: {first_line_doc}")

            tool_list_str = "\n".join(tool_list)
            base_filename = os.path.basename(audio_file_path)

            # Generate plan with LLM
            planning_prompt = f"""
You are an expert audio engineer AI. Create a step-by-step plan to achieve the user's goal for the given audio file.

User Task: "{task}"
Initial Audio File: "{base_filename}" (located in the working directory)

Available Tools:
{tool_list_str}

Instructions for Planning:
1.  Start the plan with `AUDIO_QA` to analyze the initial audio and understand its properties relevant to the task. Use a descriptive `task` parameter for AUDIO_QA.
2.  Break down the user task into logical steps using the available tools.
3.  Use `AUDIO_QA` strategically between steps if you need to determine parameters (e.g., noise level, frequency ranges) based on the current audio state.
4.  For creative tasks (generation, mixing), use `AUDIO_GENERATE` and `MIX`. Specify clear prompts for generation.
5.  Use `WRITE_AUDIO` explicitly when saving intermediate or final results. The `wav` parameter for `WRITE_AUDIO` *must* be the direct output of another tool (like a NumPy array) or a `READ_AUDIO_NUMPY` call. Give outputs meaningful names (e.g., `'{base_filename}_denoised.wav'`).
6.  Include a final `AUDIO_QA` step to verify the result against the original task.
7.  Format the plan as a Markdown checklist using `- [ ] TOOL_NAME(param1='value', param2=123)`.
8.  Parameter values MUST be enclosed in single or double quotes for strings, or be valid numbers/booleans. File paths should be relative to the working directory (e.g., `'input_file.wav'`, `'step1_output.wav'`).
9.  Ensure all function calls use *only* the tools listed above and their *exact* parameter names. Do not invent tools or parameters.
10. Keep the plan concise and focused on achieving the user task.

Create the Markdown checklist plan now:"""

            plan_text = await self._call_llm(planning_prompt)

            # Parse markdown checklist
            steps = [line.strip() for line in plan_text.splitlines()
                     if re.match(r"^\s*-\s*\[\s*\]", line)]

            if not steps:
                logfire.warning("LLM did not produce a valid Markdown checklist. Attempting fallback parsing.", raw_plan=plan_text)
                # Fallback parsing: look for lines that look like function calls
                steps = []
                potential_steps = [line.strip() for line in plan_text.splitlines() if line.strip()]
                for line in potential_steps:
                    # Check if it looks like a function call within a list item
                    if re.search(r"^\s*-\s*([A-Z_][A-Z0-9_]*)\s*\(.*\)", line):
                         steps.append(f"- [ ] {line.lstrip('-').strip()}")
                    # Check if it looks *only* like a function call
                    elif re.match(r"([A-Z_][A-Z0-9_]*)\s*\(.*\)", line):
                         steps.append(f"- [ ] {line}")

                if not steps:
                    logfire.error("Failed to generate or parse any steps from LLM plan.", raw_plan=plan_text)
                    raise ValueError("Failed to generate a valid plan checklist from LLM.")

            # Ensure file paths in the initial plan point to the correct input file
            initial_base = os.path.basename(audio_file_path)
            corrected_steps = []
            for step in steps:
                 # Simple heuristic: replace placeholder paths if they don't exist yet
                 step = re.sub(r"wav_path=['\"]([^'\"]+)['\"]",
                               lambda m: f"wav_path='{initial_base}'" if not os.path.exists(os.path.join(self.working_dir, m.group(1))) and m.group(1) != initial_base else m.group(0),
                               step)
                 # Ensure first AUDIO_QA uses the correct initial file
                 if "AUDIO_QA" in step and corrected_steps == []:
                     step = re.sub(r"wav_path=['\"]([^'\"]+)['\"]", f"wav_path='{initial_base}'", step)
                 corrected_steps.append(step)
            steps = corrected_steps

            # Optionally ensure first step is AUDIO_QA (though prompt asks for it)
            if not steps or "AUDIO_QA" not in steps[0]:
                initial_qa_task = f"Analyze this audio '{initial_base}' to understand its properties relevant for the task: {task}"
                initial_qa = f"- [ ] AUDIO_QA(wav_path='{initial_base}', task='{initial_qa_task}')"
                steps.insert(0, initial_qa)
                logfire.info("Prepended initial AUDIO_QA step as it was missing.")


            plan = AudioProcessingPlan(
                task_description=task,
                steps=steps,
                current_audio_path=audio_file_path, # Store full path initially
                original_input_path=audio_file_path
            )
            logfire.info(f"Generated initial plan with {len(plan.steps)} steps.", plan_steps=plan.steps)
            return plan

    async def _generate_code_for_step(self, plan: AudioProcessingPlan, step_index: int) -> str:
        """
        Generate/Extract Python code for a specific plan step.
        Focuses on extracting the call directly from the markdown step.
        Removes aggressive auto-correction of file paths. Relies on LLM review for path fixing.
        """
        with logfire.span("generate_code_for_step", step_index=step_index, step_text=plan.steps[step_index]):
            step_text = plan.steps[step_index]
            step_instruction = step_text.split("]", 1)[-1].strip() # Get "TOOL(...)" part

            # Basic validation: Does it look like a function call?
            match = re.match(r"([A-Z_][A-Z0-9_]*)\s*\((.*)\)", step_instruction, re.DOTALL)
            if not match:
                 logfire.error(f"Step {step_index+1} instruction does not look like a valid tool call: '{step_instruction}'")
                 # Ask LLM to fix the specific step format
                 fix_prompt = f"""
The following step from an audio processing plan is malformed.
Malformed Step: "{step_text}"
It should be in the format: "- [ ] TOOL_NAME(param='value')"

Available Tools: {list(self.available_tools.keys())}

Please rewrite the step correctly. Output only the corrected step line:
"""
                 corrected_step_text = await self._call_llm(fix_prompt)
                 corrected_step_instruction = corrected_step_text.split("]", 1)[-1].strip()

                 # Retry match
                 match = re.match(r"([A-Z_][A-Z0-9_]*)\s*\((.*)\)", corrected_step_instruction, re.DOTALL)
                 if not match:
                     raise ValueError(f"Failed to parse or correct step instruction: '{step_instruction}' -> '{corrected_step_instruction}'")
                 else:
                     logfire.info(f"LLM corrected step format: '{corrected_step_instruction}'")
                     step_instruction = corrected_step_instruction
                     # Update the plan with the corrected step text
                     plan.steps[step_index] = corrected_step_text.strip()


            func_name, _ = match.groups() # We only need func_name for validation here

            # Validate function name exists using available tools
            if func_name not in self.available_tools:
                available_tool_names = list(self.available_tools.keys())
                closest_matches = difflib.get_close_matches(func_name, available_tool_names, n=1, cutoff=0.7)
                if closest_matches:
                    logfire.warning(f"Tool '{func_name}' in step not found. Found close match '{closest_matches[0]}'. Auto-correcting.")
                    step_instruction = step_instruction.replace(func_name, closest_matches[0], 1)
                    # Update plan step text as well
                    plan.steps[step_index] = plan.steps[step_index].replace(func_name, closest_matches[0], 1)
                else:
                    # If no close match, raise error to force LLM review to fix it
                    logfire.error(f"Tool '{func_name}' in step instruction does not exist and no close match found.", available=available_tool_names)
                    raise NameError(f"Tool '{func_name}' in step instruction does not exist. Available tools: {available_tool_names}")

            # Prepend 'audio_tools.' if missing
            if not step_instruction.startswith("audio_tools."):
                code_string = f"audio_tools.{step_instruction}"
            else:
                code_string = step_instruction

            logfire.info(f"Code for step {step_index + 1}: {code_string}")
            return code_string


    async def _evaluate_ast_node(self, node: ast.expr, current_plan: AudioProcessingPlan) -> Any:
        """Safely evaluate an AST node, handling constants, known calls, and paths."""
        if isinstance(node, ast.Constant): # Python 3.8+
            return node.value
        elif isinstance(node, (ast.Str, ast.Num, ast.NameConstant)): # Older Python constants
             try:
                 return ast.literal_eval(node)
             except ValueError:
                 # Handle potential string values that aren't quoted correctly in ast representation
                 if isinstance(node, ast.Str): return node.s
                 raise
        elif isinstance(node, ast.List):
             return [await self._evaluate_ast_node(el, current_plan) for el in node.elts]
        elif isinstance(node, ast.Dict):
             return {await self._evaluate_ast_node(k, current_plan): await self._evaluate_ast_node(v, current_plan)
                     for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Call):
            # Handle known safe/necessary nested calls explicitly
            call_str = unparse(node).strip() # Use unparse for better reconstruction (Py 3.9+)
            logfire.debug(f"Evaluating nested call: {call_str}")

            # Handle audio_tools.READ_AUDIO_NUMPY(...)
            if 'audio_tools.READ_AUDIO_NUMPY' in call_str:
                nested_kwargs = {}
                for kw in node.keywords:
                    if kw.arg in ['wav_path', 'sr']: # Only parse expected args
                         nested_kwargs[kw.arg] = await self._evaluate_ast_node(kw.value, current_plan)

                # Resolve relative path from wav_path
                if 'wav_path' in nested_kwargs:
                    nested_kwargs['wav_path'] = self._resolve_path(nested_kwargs['wav_path'])
                    logfire.debug(f"Executing nested READ_AUDIO_NUMPY: {nested_kwargs}")
                    return audio_tools.READ_AUDIO_NUMPY(**nested_kwargs)
                else:
                     raise ValueError(f"Nested READ_AUDIO_NUMPY call missing 'wav_path': {call_str}")

            # Handle os.path.join(...)
            elif 'os.path.join' in call_str:
                 # Assume arguments are strings or evaluated nodes returning strings
                 join_args = [str(await self._evaluate_ast_node(arg, current_plan)) for arg in node.args]
                 # Prepend working dir if the path isn't already absolute
                 joined_path = os.path.join(*join_args)
                 logfire.debug(f"Evaluated os.path.join: {joined_path}")
                 return joined_path # Return the string path

            else:
                 logfire.warning(f"Unknown or disallowed nested call: {call_str}. Treating as string.")
                 # Fallback: return the string representation, might fail later
                 return call_str
        elif isinstance(node, ast.Name):
             # Allow specific safe names like 'True', 'False', 'None' (handled by NameConstant anyway)
             # Or potentially pass simple variables if we introduce state later
             if node.id in ['True', 'False', 'None']:
                 return ast.literal_eval(node) # Safe
             else:
                 logfire.warning(f"Disallowed variable name used in expression: {node.id}. Treating as string.")
                 return node.id # Treat as string, likely incorrect
        else:
            # For other types, attempt literal_eval or return string representation
            try:
                return ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError):
                node_str = unparse(node).strip() # Py 3.9+
                logfire.warning(f"Could not evaluate AST node type {type(node)}, treating as string: {node_str}")
                return node_str # Return string representation

    def _resolve_path(self, path_str: str) -> str:
        """Resolve a path string relative to the working directory."""
        if not isinstance(path_str, str):
            logfire.warning(f"Path expected to be string, got {type(path_str)}. Returning as is.")
            return path_str # Or raise error?
        abs_path = os.path.abspath(path_str)
        abs_working_dir = os.path.abspath(self.working_dir)
        # If it's already absolute and *within* the working dir, keep it
        if os.path.isabs(path_str) and abs_path.startswith(abs_working_dir):
            return path_str
        # If it's relative, join with working dir
        elif not os.path.isabs(path_str):
            return os.path.join(self.working_dir, path_str)
        # If it's absolute but *outside* working dir, this is suspicious
        else:
            logfire.warning(f"Path '{path_str}' is absolute and outside working directory '{self.working_dir}'. Using absolute path but this may be unintended.")
            return path_str


    async def _execute_step(self, code_string: str, step_index: int, plan: AudioProcessingPlan) -> StepExecutionResult:
        """Execute the generated code for a step using AST parsing."""
        start_time = time()
        with logfire.span("execute_step", step=step_index+1, code=code_string):
            try:
                # Parse the code string into an Abstract Syntax Tree (AST)
                try:
                    tree = ast.parse(code_string)
                    if not tree.body or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
                        raise SyntaxError("Code string is not a single function call expression.")
                    call_node = tree.body[0].value
                except SyntaxError as e:
                    logfire.error(f"AST parsing failed for code: {code_string}", error=str(e))
                    raise ValueError(f"Invalid Python syntax in generated code: {code_string}") from e

                # --- Extract Function Name ---
                func_name = None
                if isinstance(call_node.func, ast.Attribute) and \
                   isinstance(call_node.func.value, ast.Name) and \
                   call_node.func.value.id == 'audio_tools':
                    func_name = call_node.func.attr
                elif isinstance(call_node.func, ast.Name):
                    # Allow calling without 'audio_tools.' prefix, but log it
                    func_name = call_node.func.id
                    logfire.debug(f"Executing tool '{func_name}' without 'audio_tools.' prefix.")
                else:
                    raise ValueError(f"Cannot determine function name from code: {code_string}")

                # --- Validate Function Name (Stricter: BEFORE parsing args) ---
                if func_name not in self.available_tools:
                    tool_names = list(self.available_tools.keys())
                    closest = difflib.get_close_matches(func_name, tool_names, n=1, cutoff=0.7)
                    suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
                    # Raise error immediately to force LLM correction in review
                    raise NameError(f"Tool '{func_name}' is not available.{suggestion} Available: {tool_names}")

                tool_func = self.available_tools[func_name]
                try:
                    sig = inspect.signature(tool_func)
                    valid_params = sig.parameters.keys()
                    required_params = {p.name for p in sig.parameters.values()
                                      if p.default is p.empty and p.name != 'self'}
                except ValueError: # Handles issues with signature inspection (less likely with explicit tools)
                    logfire.warning(f"Could not get signature for tool {func_name}. Parameter validation will be skipped.")
                    valid_params = None
                    required_params = set()

                # --- Parse Arguments using AST ---
                kwargs = {}
                provided_params = set()
                for kw in call_node.keywords:
                    param_name = kw.arg
                    provided_params.add(param_name)

                    # --- Validate Parameter Name (Stricter) ---
                    if valid_params and param_name not in valid_params:
                        # Raise error immediately if param is unknown for this tool
                        raise ValueError(f"Invalid parameter '{param_name}' provided for tool '{func_name}'. Valid parameters are: {list(valid_params)}")

                    # Evaluate the value node (handles constants, nested calls, paths)
                    try:
                        value = await self._evaluate_ast_node(kw.value, plan)
                        # Resolve paths relative to working dir *after* evaluation
                        # Check param name convention or type hint if available?
                        if 'path' in param_name or 'file' in param_name or 'dir' in param_name:
                            if isinstance(value, str):
                                value = self._resolve_path(value)
                            elif isinstance(value, list) and all(isinstance(p, str) for p in value):
                                value = [self._resolve_path(p) for p in value]

                        kwargs[param_name] = value
                    except Exception as eval_err:
                         logfire.error(f"Error evaluating parameter '{param_name}' for {func_name}: {eval_err}", node=ast.dump(kw.value))
                         raise ValueError(f"Error evaluating parameter '{param_name}' for {func_name}: {eval_err}") from eval_err


                # --- Check for Missing Required Parameters ---
                missing = required_params - provided_params
                if missing:
                    raise ValueError(f"{func_name}() missing required keyword argument(s): {', '.join(sorted(list(missing)))}")

                # --- Execute the Function ---
                logfire.info(f"Executing: {func_name}({kwargs})")
                result = tool_func(**kwargs) # Assume tools are synchronous for now

                # --- Process Result ---
                output_path = None
                output_paths = None
                message = f"Step {step_index+1} ({func_name}) executed successfully."
                duration = time() - start_time

                # Check for specific known output types or conventions
                if isinstance(result, str) and os.path.exists(result): # Tool returned a single file path
                    # Ensure path is absolute for consistency downstream
                    output_path = os.path.abspath(result)
                    # Verify it's within the working directory for safety
                    if not output_path.startswith(os.path.abspath(self.working_dir)):
                         logfire.warning(f"Output path {output_path} is outside the working directory {self.working_dir}.")
                    message += f" Output: {os.path.basename(output_path)}"
                elif isinstance(result, list) and result and all(isinstance(item, str) and os.path.exists(item) for item in result): # Tool returned multiple file paths
                    output_paths = [os.path.abspath(p) for p in result]
                    output_path = output_paths[0] # Use first as primary convention
                    # Verify paths
                    for p in output_paths:
                        if not p.startswith(os.path.abspath(self.working_dir)):
                             logfire.warning(f"Output path {p} is outside the working directory {self.working_dir}.")
                    message += f" Outputs: {[os.path.basename(p) for p in output_paths]}"
                elif hasattr(result, '__array__'): # Numpy array returned, write it automatically
                    temp_name = f"step_{step_index+1}_output_{audio_tools.generate_random_series()}.wav"
                    temp_out_path = os.path.join(self.working_dir, temp_name)
                    logfire.info(f"Tool returned raw audio data. Writing to temporary file: {temp_out_path}")
                    # Assume SAMPLE_RATE, could be smarter if tool provided it
                    audio_tools.WRITE_AUDIO(wav=result, name=temp_out_path, sr=getattr(audio_tools, 'SAMPLE_RATE', 44100))
                    if os.path.exists(temp_out_path):
                        output_path = temp_out_path
                        message += f" Raw data written to {os.path.basename(output_path)}."
                    else:
                         raise IOError(f"Failed to write temporary output file {temp_out_path}")
                elif func_name in ["AUDIO_QA", "AUDIO_DIFF"]: # Analysis tools return text/dict
                    analysis_str = str(result)
                    message += f" Analysis: {analysis_str[:200]}{'...' if len(analysis_str) > 200 else ''}"
                    # Analysis steps don't usually change the current_audio_path
                elif result is None and func_name.startswith("WRITE"):
                     # WRITE functions might return None on success, infer output from input name arg
                     if 'name' in kwargs and isinstance(kwargs['name'], str):
                         written_path = self._resolve_path(kwargs['name'])
                         if os.path.exists(written_path):
                             output_path = written_path
                             message += f" Output confirmed: {os.path.basename(output_path)}"
                         else:
                              logfire.warning(f"WRITE tool {func_name} returned None, and specified output name '{kwargs['name']}' not found.")
                     else:
                          logfire.warning(f"WRITE tool {func_name} returned None but 'name' argument missing or invalid.")
                # Add more specific result handling if needed based on tool behavior

                # Validate final output path exists if one was determined
                if output_path and not os.path.exists(output_path):
                    logfire.error(f"Output path '{output_path}' determined but file does not exist after execution.")
                    raise FileNotFoundError(f"Output file '{output_path}' not found after step {step_index+1} execution.")

                return StepExecutionResult(
                    success=True,
                    output_path=output_path,
                    output_paths=output_paths,
                    message=message,
                    duration=duration
                )

            except Exception as e:
                error_message = f"Error executing step {step_index+1} ({code_string}): {type(e).__name__}: {str(e)}"
                logfire.error(f"Step {step_index+1} execution failed.", error=str(e), code=code_string, exc_info=True)
                # Add traceback to the message for clarity in logs/review
                # error_message += f"\nTraceback:\n{traceback.format_exc()}"
                return StepExecutionResult(
                    success=False,
                    message=error_message,
                    duration=time() - start_time
                )

    async def _update_plan(self, plan: AudioProcessingPlan, step_index: int, result: StepExecutionResult) -> AudioProcessingPlan:
        """Update plan based on execution results and LLM review, with improved review prompt."""
        with logfire.span("update_plan", step_index=step_index, success=result.success):
            # --- Update Step Status in Plan ---
            if 0 <= step_index < len(plan.steps):
                step_text = plan.steps[step_index]
                if result.success:
                    # Mark as completed [x]
                    plan.steps[step_index] = step_text.replace("- [ ]", "- [x]", 1)
                    if step_index not in plan.completed_steps_indices:
                        plan.completed_steps_indices.append(step_index)
                else:
                    # Mark as failed [!] (or keep as [ ] for retry?) - Let's use [!]
                    plan.steps[step_index] = step_text.replace("- [ ]", "- [!]", 1).replace("- [x]", "- [!]", 1) # Mark failed even if previously complete
                    if step_index in plan.completed_steps_indices:
                        plan.completed_steps_indices.remove(step_index)
            else:
                logfire.warning(f"Attempted to update plan for invalid step_index: {step_index}")


            # --- Update Current Audio Path ---
            new_audio_path = None
            if result.success:
                if result.output_path:
                    new_audio_path = result.output_path
                # If multiple outputs, convention is to use the first? Or let LLM decide?
                # Let's stick to the first for now as `current_audio_path` implies singularity
                elif result.output_paths:
                    new_audio_path = result.output_paths[0]

            if new_audio_path and os.path.exists(new_audio_path):
                 # Check if path changed
                 if plan.current_audio_path != new_audio_path:
                     logfire.info(f"Current audio path updated to: {os.path.basename(new_audio_path)}")
                     plan.current_audio_path = new_audio_path
            elif not result.success:
                logfire.warning(f"Step {step_index+1} failed. Current audio path remains: {os.path.basename(plan.current_audio_path)}")
            # else: step succeeded but didn't produce a new audio path (e.g., AUDIO_QA)


            # --- Gather Context for LLM Review ---
            # Get list of available .wav files in the working directory
            try:
                available_files = [f for f in os.listdir(self.working_dir) if f.lower().endswith('.wav')]
            except OSError as e:
                logfire.error(f"Failed to list working directory {self.working_dir}: {e}")
                available_files = ["<Error listing files>"]

            # Context from step result
            step_result_summary = f"Step {step_index + 1} Result:\n- Success: {result.success}\n- Message: {result.message}"
            if result.success and result.output_path:
                 step_result_summary += f"\n- Output File(s): {[os.path.basename(p) for p in result.output_paths] if result.output_paths else os.path.basename(result.output_path)}"

            # Context from user feedback (if any)
            feedback_context = ""
            if plan.user_feedback:
                feedback_context = f"\n\nUser Feedback Provided: '{plan.user_feedback}'"
                plan.user_feedback = None  # Clear after use


            # --- Construct Improved Review Prompt ---
            review_prompt = f"""
You are reviewing an audio processing plan execution. Analyze the last step's result and update the plan accordingly.

Original Task: "{plan.task_description}"
Initial Input File: "{os.path.basename(plan.original_input_path)}"

Current Plan Status:
```markdown
{chr(10).join(plan.steps)}


{step_result_summary}

Current Audio File Considered for Next Step: "{os.path.basename(plan.current_audio_path)}"
Files Available in Working Directory: {available_files}
{feedback_context}

Review and Update Instructions:

Diagnose Failure: If the last step failed (Success: False or step marked [! ]), analyze the error message. What was the likely cause (e.g., wrong file path, incorrect parameters, tool limitation)?

Verify File Paths: Examine the wav_path (or other file path arguments) in all remaining - [ ] steps.

Do the paths exist in the 'Files Available'?

Should a step use the output from the last successful step ({os.path.basename(plan.current_audio_path)}) instead of an older or non-existent file?

Correct file paths in the updated_plan_steps if they are wrong. Use relative paths (just the filename).

Correct Failed/Incorrect Step: If the last step failed [! ] or produced an unexpected result (based on the message or analysis):

Modify the failed step itself in updated_plan_steps. Try correcting parameters, fixing file paths, or choosing a different, more appropriate tool from the available list if the original tool was wrong. Mark it as - [ ] to retry.

Do NOT just add new steps to compensate without fixing the root cause.

Refine Remaining Steps: Based on the result/analysis (especially from AUDIO_QA), refine parameters or arguments in subsequent - [ ] steps if necessary.

Add Steps Cautiously: Only add new steps if the diagnosis reveals a missing logical step is required to achieve the task, and only after attempting to correct existing issues.

Determine Completion: Is the original task {plan.task_description} fully achieved based on the executed steps and the final result/analysis? Set is_complete to true if yes.

Ask for Clarification: If you lack essential information required to proceed or make a correction (e.g., user preference needed, ambiguous task), set clarification_question to a specific question for the user. Otherwise, set it to null.

Output Format (JSON):
Provide your response ONLY in this JSON format. Do not include any explanatory text outside the JSON structure.

{{
    "analysis_of_result": "Your brief analysis of the last step's outcome and any diagnosis.",
    "updated_plan_steps": [
        "- [x] STEP 1...",
        "- [!] FAILED STEP (potentially modified for retry)...",
        "- [ ] CORRECTED/REFINED NEXT STEP...",
        "- [ ] ..."
    ],
    "is_complete": boolean,
    "clarification_question": "String question for the user or null"
}}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

"""

# --- Call LLM for Review ---
        response_json_str = await self._call_llm(review_prompt, model_name=self.model_name, expect_json=True) # Use default model for review

        # --- Process LLM Response ---
        try:
            response_data = json.loads(response_json_str)

            logfire.info("LLM Review Analysis:", analysis=response_data.get("analysis_of_result", "N/A"))

            # Update plan steps if valid
            if isinstance(response_data.get("updated_plan_steps"), list):
                updated_steps = response_data["updated_plan_steps"]
                # Basic validation of the updated steps format
                if all(isinstance(s, str) and re.match(r"^\s*-\s*\[[\sx!]\]", s) for s in updated_steps):
                    if plan.steps != updated_steps:
                         logfire.info("LLM updated the plan steps.", diff=list(difflib.unified_diff(plan.steps, updated_steps, lineterm='')))
                         plan.steps = updated_steps
                else:
                    logfire.warning("LLM provided 'updated_plan_steps' but the format was invalid. Keeping previous plan.", llm_steps=updated_steps)
            else:
                 logfire.warning("LLM response missing or invalid 'updated_plan_steps'. Keeping previous plan.", llm_response=response_data)

            # Update completion status
            is_complete = response_data.get("is_complete")
            if isinstance(is_complete, bool):
                if plan.is_complete != is_complete:
                    logfire.info(f"Plan completion status updated to: {is_complete}")
                    plan.is_complete = is_complete
            else:
                logfire.warning("LLM response missing or invalid 'is_complete' flag.", llm_response=response_data)


            # Update clarification question
            clarification = response_data.get("clarification_question")
            if isinstance(clarification, str) and clarification.strip():
                plan.clarification_question = clarification.strip()
                logfire.info(f"LLM asks for clarification: {plan.clarification_question}")
            else:
                plan.clarification_question = None # Ensure it's cleared if null or empty

            # Recalculate completed steps indices from the potentially updated plan
            plan.completed_steps_indices = [i for i, step in enumerate(plan.steps) if step.startswith("- [x]")]

            return plan

        except json.JSONDecodeError as e:
            logfire.error(f"Failed to decode JSON response from LLM review: {e}", llm_response=response_json_str)
            # Keep existing plan on JSON error, maybe ask for clarification?
            plan.clarification_question = f"I received an invalid internal response ({e}). Could you please clarify the next step or restate the goal?"
            return plan
        except Exception as e:
            logfire.error(f"Error processing LLM review response: {e}", llm_response=response_json_str, exc_info=True)
            # Keep existing plan on other errors
            plan.clarification_question = f"An internal error occurred while reviewing the plan ({e}). Please check the logs."
            return plan

async def process_audio(self, task: str, audio_file_path: str) -> str:
    """Process audio using the iterative planning and execution approach."""
    run_id = audio_tools.generate_random_series(4) # Short ID for this run
    with logfire.span("process_audio", task=task, input_file=audio_file_path, run_id=run_id):
        start_time = time()

        # --- Input File Handling ---
        abs_input_path = os.path.abspath(audio_file_path)
        if not os.path.exists(abs_input_path):
            logfire.error(f"Input audio file not found: {abs_input_path}")
            raise FileNotFoundError(f"Input audio file not found: {abs_input_path}")

        # Copy input file to working directory with a unique name to avoid conflicts
        input_basename = os.path.basename(abs_input_path)
        sanitized_basename = re.sub(r'[^\w.\-]', '_', input_basename) # Basic sanitization
        working_input_filename = f"input_{run_id}_{sanitized_basename}"
        working_input_path = os.path.join(self.working_dir, working_input_filename)

        try:
            import shutil
            shutil.copy(abs_input_path, working_input_path)
            logfire.info(f"Copied input '{input_basename}' to working directory as '{working_input_filename}'")
        except Exception as e:
             logfire.error(f"Failed to copy input file to working directory: {e}", exc_info=True)
             raise IOError(f"Failed to copy input file to {self.working_dir}: {e}") from e

        # --- Initial Plan ---
        try:
            plan = await self._generate_initial_plan(task, working_input_path)
        except Exception as e:
            logfire.error(f"Failed to generate initial plan: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate initial plan: {e}") from e

        max_steps = max(25, len(plan.steps) * 3) # Increased max steps slightly
        step_count = 0
        executed_step_indices = set()

        # --- Main Execution Loop ---
        while not plan.is_complete and step_count < max_steps:
            iteration = step_count + 1
            logfire.info(f"--- Iteration {iteration}/{max_steps} ---")
            print(f"\n--- Iteration {iteration}/{max_steps} ---") # User feedback

            # --- Handle Clarification ---
            if plan.clarification_question:
                print(f"\n🤔 AI asks: {plan.clarification_question}")
                logfire.info(f"Asking user for clarification: {plan.clarification_question}")
                try:
                    user_answer = input("Your answer: ")
                except EOFError: # Handle non-interactive environments
                     logfire.warning("EOFError reading user input. Proceeding without feedback.")
                     user_answer = ""
                plan.user_feedback = user_answer
                plan.clarification_question = None # Clear question

                # Trigger a plan review immediately with the feedback
                # Find the last relevant step index (failed or last completed)
                last_idx = -1
                for i in range(len(plan.steps) - 1, -1, -1):
                    if plan.steps[i].startswith("- [!]") or plan.steps[i].startswith("- [x]"):
                        last_idx = i
                        break
                dummy_result = StepExecutionResult(success=True, message="User clarification provided.", duration=0)
                try:
                    plan = await self._update_plan(plan, last_idx, dummy_result)
                    logfire.info("Plan updated after user clarification.")
                except Exception as e:
                     logfire.error(f"Failed to update plan after user clarification: {e}", exc_info=True)
                     # Decide how to proceed - maybe halt?
                     raise RuntimeError(f"Failed to update plan after user clarification: {e}") from e
                step_count += 1 # Count clarification as an iteration
                continue # Restart loop to find next step

            # --- Find Next Step ---
            next_step_index = -1
            for i, step in enumerate(plan.steps):
                # Find the first step that is not completed '[ ]' and hasn't been tried excessively
                # Let's allow retrying failed steps '[!]' based on LLM review
                if (step.startswith("- [ ]") or step.startswith("- [!]")) and i not in plan.completed_steps_indices:
                     # Basic check to prevent infinite loops on a persistently failing step
                     # Allow maybe 2 retries? Let LLM decide mostly via review prompt.
                     # We just need *a* step to try.
                     next_step_index = i
                     break

            if next_step_index == -1:
                # No steps left marked as [ ] or [!]. Check if LLM thinks it's complete.
                if not plan.is_complete:
                    logfire.warning("No incomplete steps found, but plan is not marked complete. Forcing review.")
                    # Trigger a final review
                    last_idx = plan.completed_steps_indices[-1] if plan.completed_steps_indices else -1
                    final_review_result = StepExecutionResult(success=True, message="All plan steps processed. Requesting final completion check.", duration=0)
                    try:
                        plan = await self._update_plan(plan, last_idx, final_review_result)
                        if not plan.is_complete:
                            logfire.error("Final review did not mark plan as complete. Potential loop or error.")
                            plan.is_complete = True # Force completion to exit loop
                    except Exception as e:
                        logfire.error(f"Failed final plan review: {e}", exc_info=True)
                        plan.is_complete = True # Force completion on error
                else:
                     logfire.info("No more steps to execute and plan marked complete.")
                break # Exit loop

            # Mark that we are attempting this step
            executed_step_indices.add(next_step_index)
            current_step_text = plan.steps[next_step_index]
            logfire.info(f"Selected Step {next_step_index + 1}: {current_step_text}")
            print(f"Executing Step {next_step_index + 1}/{len(plan.steps)}: {current_step_text}")

            # --- Generate Code and Execute ---
            result = None
            try:
                # Generate code (might correct step text in plan)
                code = await self._generate_code_for_step(plan, next_step_index)
                # Execute code
                result = await self._execute_step(code, next_step_index, plan)
            except NameError as e: # Specific handling for tool not found during generation/validation
                logfire.error(f"Tool Error processing step {next_step_index + 1}: {e}", step_text=current_step_text)
                result = StepExecutionResult(success=False, message=f"Tool Error: {str(e)}", duration=0)
            except ValueError as e: # Specific handling for parsing/parameter errors
                logfire.error(f"Value Error processing step {next_step_index + 1}: {e}", step_text=current_step_text)
                result = StepExecutionResult(success=False, message=f"Parameter/Value Error: {str(e)}", duration=0)
            except Exception as e: # Catch other unexpected errors during generation/execution
                logfire.error(f"Unexpected error processing step {next_step_index + 1}: {e}", step_text=current_step_text, exc_info=True)
                result = StepExecutionResult(success=False, message=f"Unexpected Error: {str(e)}", duration=0)

            # --- Update Plan via LLM Review ---
            if result is not None:
                try:
                    plan = await self._update_plan(plan, next_step_index, result)
                except Exception as e:
                    logfire.error(f"Failed to update plan after step {next_step_index + 1}: {e}", exc_info=True)
                    # If review fails, we should probably stop to avoid inconsistent state
                    raise RuntimeError(f"Failed to update plan after step {next_step_index + 1}: {e}") from e
            else:
                # This case should ideally not happen if exceptions are caught correctly
                logfire.critical(f"Step {next_step_index + 1} execution yielded no result object. Halting.")
                raise RuntimeError(f"Internal error: Step {next_step_index + 1} execution failed to produce a result.")


            step_count += 1
            # Optional: Short delay between steps?
            # await asyncio.sleep(0.5)

        # --- Final Status Check ---
        total_duration = time() - start_time
        logfire.info(f"Processing loop finished after {total_duration:.2f}s, {step_count} iterations.")
        print(f"\nProcessing finished in {total_duration:.2f}s.")

        if plan.is_complete and plan.current_audio_path and os.path.exists(plan.current_audio_path):
            final_path = os.path.abspath(plan.current_audio_path)
            logfire.info(f"Processing completed successfully. Final audio: {final_path}")
            return final_path
        elif step_count >= max_steps:
            logfire.error(f"Maximum steps ({max_steps}) reached without completion.")
            raise RuntimeError(f"Maximum steps ({max_steps}) reached without completion. The task may be too complex or stuck in a loop. Check logs in {self.working_dir}")
        elif not plan.current_audio_path or not os.path.exists(plan.current_audio_path):
             logfire.error("Processing finished, but the final audio path is missing or invalid.", final_path=plan.current_audio_path)
             raise RuntimeError(f"Processing finished, but the expected final audio file ('{plan.current_audio_path}') was not found. Check logs in {self.working_dir}")
        else:
            logfire.error("Processing finished but was not marked as complete by the AI.")
            # Return the last known path, but maybe with a warning?
            final_path = os.path.abspath(plan.current_audio_path)
            print("Warning: Processing finished, but AI did not explicitly confirm completion.")
            return final_path # Return best guess



async def main_async():
    parser = argparse.ArgumentParser(description="AI Audio Editor")
    parser.add_argument("--task", type=str, required=True, help="Description of the audio transformation task")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file (e.g., .wav)")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save final output audio")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model to use (e.g., gemini-1.5-flash, gemini-1.5-pro)")
    args = parser.parse_args()

    # Basic check for Gemini API Key before starting
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set. Cannot proceed.")
        logfire.critical("GEMINI_API_KEY not set at script start.")
        return 1

    exit_code = 1 # Default to error
    editor = None # Initialize editor to None

    try:
        # Validate input file existence
        if not os.path.exists(args.input):
            print(f"Error: Input file not found at {args.input}")
            logfire.error(f"Input file not found: {args.input}")
            return 1
        # Basic check for wav? Maybe relax later.
        if not args.input.lower().endswith(".wav"):
            print(f"Warning: Input file '{args.input}' is not a .wav file. Compatibility issues may occur.")
            logfire.warning(f"Input file is not a .wav: {args.input}")

        # Initialize editor
        editor = AIAudioEditor(model_name=args.model)
        print(f"Starting audio processing task: \"{args.task}\"")
        print(f"Input file: {os.path.abspath(args.input)}")
        print(f"Working directory: {editor.working_dir}")

        # Process audio
        final_path = await editor.process_audio(args.task, args.input)

        # --- Output Handling ---
        if final_path and os.path.exists(final_path):
            output_path = os.path.abspath(args.output)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            try:
                import shutil
                shutil.copy(final_path, output_path)
                print(f"\n✅ Success! Final audio saved to: {output_path}")
                print(f"(Working files are in: {editor.working_dir})")
                logfire.info(f"Successfully copied final audio {final_path} to {output_path}")
                exit_code = 0 # Success
            except Exception as e:
                print(f"\n❌ Error: Failed to copy final audio from {final_path} to {output_path}: {e}")
                print(f"The final result is available at: {final_path}")
                logfire.error(f"Failed to copy final audio to output path: {e}", final_path=final_path, output_path=output_path, exc_info=True)
                exit_code = 1 # Indicate copy error
        else:
            # This case should ideally be caught by exceptions in process_audio
            print("\n❌ Error: Processing finished, but final audio file was not found or generated.")
            logfire.error("Processing seemed to finish, but final_path was invalid.", final_path_received=final_path)
            exit_code = 1

    except FileNotFoundError as e:
        # Already handled logging/printing specific message usually
        print(f"\n❌ Error: A required file was not found: {e}")
        exit_code = 1
    except ConnectionError as e: # Catch LLM connection issues
        print(f"\n❌ Error: Could not connect to the AI service: {e}")
        logfire.critical(f"LLM Connection Error: {e}", exc_info=True)
        exit_code = 1
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user.")
        logfire.warning("Processing interrupted by KeyboardInterrupt.")
        exit_code = 130 # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        # Log the full traceback if not already done
        if not logfire.is_configured(): # Check if logging failed somehow
            traceback.print_exc()
        else:
            # Assuming process_audio or other parts logged it sufficiently
            logfire.error(f"Unhandled exception in main: {str(e)}", exc_info=True) # Ensure it's logged
        exit_code = 1
    finally:
        # Optional: Clean up working directory?
        # Be careful with this - might want to keep logs/intermediate files for debugging
        # if exit_code != 0 and editor and os.path.exists(editor.working_dir):
        #     print(f"Note: Working files kept for debugging in {editor.working_dir}")
        # elif exit_code == 0 and editor and os.path.exists(editor.working_dir):
        #     try:
        #         # shutil.rmtree(editor.working_dir)
        #         # print(f"Cleaned up working directory: {editor.working_dir}")
        #         pass # Keep files by default for now
        #     except Exception as e:
        #         print(f"Warning: Failed to clean up working directory {editor.working_dir}: {e}")
        pass

    return exit_code


if __name__ == "main":
# Run the async main function
    exit_status = asyncio.run(main_async())
    exit(exit_status)
