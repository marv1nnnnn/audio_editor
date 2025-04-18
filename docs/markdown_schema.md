# Workflow Markdown File Schema (`workflow.md`)

This document outlines the structure of the `workflow.md` file used by the multi-agent audio processing system. This file acts as the central state and communication hub.

**File Generation:** A unique `workflow_{timestamp}_{request_hash}.md` file is generated by the Coordinator for each new processing request.

---

## 1. Request Summary

*   **Original Request:**
    ```text
    {User's original request text multiline}
    ```
*   **Input Audio:** `{Path to the initial input audio file}`
*   **Timestamp:** `{Timestamp of request initiation}`
*   **Workflow ID:** `{Unique identifier for this workflow run}`

---

## 2. Product Requirements Document (PRD)

*This section is generated by the Planner Agent based on the original request.*

*   **Goal:** {Brief, high-level goal of the processing}
*   **Key Requirements:**
    *   {Requirement 1}
    *   {Requirement 2}
    *   ...
*   **Constraints:**
    *   {Constraint 1 (e.g., output format)}
    *   {Constraint 2 (e.g., max duration)}
    *   ...
*   **Success Criteria:**
    *   {Criterion 1 for successful completion}
    *   ...

---

## 3. Processing Plan & Status

*This section outlines the steps generated by the Planner Agent and is updated throughout the workflow.*

### Step 1: {Brief Step Title}

*   **ID:** `step_1`
*   **Description:** {Detailed description of what this step should accomplish. Generated by Planner.}
*   **Status:** {PENDING | READY | RUNNING | CODE_GENERATED | EXECUTING | DONE | FAILED}
*   **Input Audio:** `{Path to input audio for this step (can be the original input or $OUTPUT_STEP_X)}`
*   **Output Audio:** `{Pre-defined path for the output audio of this step, e.g., /path/to/output/step_1_action.wav}`
*   **Code:**
    ```python
    # Code generated by Code Generation Agent will appear here
    # Or: "N/A" if no code execution is required (e.g., manual review step)
    ```
*   **Execution Results:**
    ```text
    # Stdout/Stderr/Logs from Executor Agent will appear here
    # Or: Error details if FAILED
    # Or: "N/A"
    ```
*   **Timestamp Start:** `{Timestamp when execution started}`
*   **Timestamp End:** `{Timestamp when execution finished/failed}`
*   **QA Notes:** {Optional notes from the QA Agent}

---

### Step 2: {Brief Step Title}

*   **ID:** `step_2`
*   **Description:** {Description...}
*   **Status:** {PENDING | READY | RUNNING | CODE_GENERATED | EXECUTING | DONE | FAILED}
*   **Input Audio:** `$OUTPUT_STEP_1`
*   **Output Audio:** `{Pre-defined path for the output audio of this step, e.g., /path/to/output/step_2_another_action.wav}`
*   **Code:**
    ```python
    # Placeholder
    ```
*   **Execution Results:**
    ```text
    # Placeholder
    ```
*   **Timestamp Start:** `{Timestamp}`
*   **Timestamp End:** `{Timestamp}`
*   **QA Notes:** {Optional}

---

*... (Additional steps as needed) ...*

---

## 4. Final Output

*   **Overall Status:** {SUCCESS | FAILURE | REQUIRES_REVIEW}
*   **Final Audio Path:** `{Path to the final processed audio file (usually the output of the last 'DONE' step)}`
*   **Summary:** {Brief summary of the outcome generated by the Coordinator or QA Agent.}

---

## 5. Workflow Log

*This section contains high-level logs added by the Coordinator.*

*   `{Timestamp}`: Workflow Initiated.
*   `{Timestamp}`: Planner generated PRD and {N} steps.
*   `{Timestamp}`: Generating code for step_1...
*   `{Timestamp}`: Code generated for step_1.
*   `{Timestamp}`: Executing step_1...
*   `{Timestamp}`: Step_1 completed successfully. Output: {path}
*   `{Timestamp}`: Generating code for step_2...
*   `{Timestamp}`: Execution failed for step_2. Error: {Error summary}. Triggering Error Analyzer.
*   `{Timestamp}`: Error Analyzer proposed fix for step_2 code.
*   `{Timestamp}`: Re-executing step_2...
*   `{Timestamp}`: Step_2 completed successfully. Output: {path}
*   `{Timestamp}`: QA checks passed.
*   `{Timestamp}`: Workflow finished successfully. Final output: {path}

---

**Notes:**

*   **Statuses:**
    *   `PENDING`: Step planned but not yet ready.
    *   `READY`: Step is ready for the next action (e.g., code generation or execution).
    *   `RUNNING`: Agent is actively working on the step (e.g., Code Gen running).
    *   `CODE_GENERATED`: Code Gen finished, ready for Executor.
    *   `EXECUTING`: Executor is running the code.
    *   `DONE`: Step completed successfully.
    *   `FAILED`: Step encountered an error.
*   **Path Variables:** `$OUTPUT_STEP_X` refers to the actual file path defined in the `Output Audio:` field of the step with `ID: step_X`. The Coordinator resolves these.
*   **Parsing:** The Coordinator uses regex-based parsing to read and update the Markdown file. Each section and field is identified by its header and list markers. This approach allows for robust updates without corrupting the file structure.
*   **Code Generation Agent:** Takes the entire Markdown content and the step ID, analyzes the step description and context, and generates a single line of Python code to execute the step.
*   **Error Analyzer:** Analyzes the code and error message for failed steps and proposes fixed code that is then reapplied by the Coordinator.
*   **QA Agent:** Evaluates the final processed audio against the requirements in the PRD and returns a meets_requirements assessment. 