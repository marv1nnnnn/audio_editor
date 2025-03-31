"""
Unit tests for Markdown utilities in the Coordinator.
"""
import pytest
import os
import re
from pathlib import Path

from audio_editor.agents.coordinator import AudioProcessingCoordinator


def test_markdown_file_creation(temp_workspace):
    """Test creation of Markdown workflow file."""
    coordinator = AudioProcessingCoordinator(str(temp_workspace))
    workflow_id = "test_workflow"
    task_description = "Test task"
    audio_path = "/path/to/audio.wav"
    workflow_file = os.path.join(temp_workspace, "docs", "test_workflow.md")
    
    # Initialize the workflow Markdown file
    coordinator._initialize_workflow_markdown(
        workflow_file,
        workflow_id,
        task_description,
        audio_path
    )
    
    # Check that the file was created
    assert os.path.exists(workflow_file)
    
    # Read the file content
    with open(workflow_file, "r") as f:
        content = f.read()
    
    # Check that the expected sections are present
    assert "# Audio Processing Workflow: test_workflow" in content
    assert "## 1. Request Summary" in content
    assert f"* **Original Request:**\n    ```text\n    {task_description}\n    ```" in content
    assert f"* **Input Audio:** `{audio_path}`" in content
    assert "## 2. Product Requirements Document (PRD)" in content
    assert "## 3. Processing Plan & Status" in content
    assert "## 4. Final Output" in content
    assert "## 5. Workflow Log" in content


def test_append_workflow_log(temp_workspace):
    """Test appending to the workflow log."""
    coordinator = AudioProcessingCoordinator(str(temp_workspace))
    workflow_id = "test_workflow"
    task_description = "Test task"
    audio_path = "/path/to/audio.wav"
    workflow_file = os.path.join(temp_workspace, "docs", "test_workflow.md")
    
    # Initialize the workflow Markdown file
    coordinator._initialize_workflow_markdown(
        workflow_file,
        workflow_id,
        task_description,
        audio_path
    )
    
    # Append to the workflow log
    log_entry = "Test log entry"
    coordinator._append_workflow_log(workflow_file, log_entry)
    
    # Read the file content
    with open(workflow_file, "r") as f:
        content = f.read()
    
    # Check that the log entry was added
    assert re.search(r"\* `[^`]+`: Test log entry", content) is not None


def test_update_workflow_section(temp_workspace):
    """Test updating a section in the workflow."""
    coordinator = AudioProcessingCoordinator(str(temp_workspace))
    workflow_id = "test_workflow"
    task_description = "Test task"
    audio_path = "/path/to/audio.wav"
    workflow_file = os.path.join(temp_workspace, "docs", "test_workflow.md")
    
    # Initialize the workflow Markdown file
    coordinator._initialize_workflow_markdown(
        workflow_file,
        workflow_id,
        task_description,
        audio_path
    )
    
    # Update a section
    coordinator._update_workflow_section(
        workflow_file,
        "Final Output",
        {
            "Overall Status": "SUCCESS",
            "Final Audio Path": "`/path/to/output.wav`",
            "Summary": "Test summary"
        }
    )
    
    # Read the file content
    with open(workflow_file, "r") as f:
        content = f.read()
    
    # Check that the section was updated
    assert "* **Overall Status:** SUCCESS" in content
    assert "* **Final Audio Path:** `/path/to/output.wav`" in content
    assert "* **Summary:** Test summary" in content


def test_find_step_by_id(temp_workspace):
    """Test finding a step by ID."""
    coordinator = AudioProcessingCoordinator(str(temp_workspace))
    
    # Create a test markdown content with steps
    markdown_content = """# Audio Processing Workflow: test_workflow

## 3. Processing Plan & Status

### Step 1: Test Step

* **ID:** `step_1`
* **Description:** Test step description
* **Status:** PENDING
* **Input Audio:** `/path/to/input.wav`
* **Output Audio:** `/path/to/output.wav`
* **Code:**
```python
# Test code
```
* **Execution Results:**
```text
# Test results
```

### Step 2: Another Step

* **ID:** `step_2`
* **Description:** Another step description
* **Status:** READY
"""
    
    # Find step_1
    step = coordinator._find_step_by_id(markdown_content, "step_1")
    
    # Check that the step was found and fields were extracted
    assert step is not None
    assert step["ID"] == "step_1"
    assert step["Description"] == "Test step description"
    assert step["Status"] == "PENDING"
    assert step["Input Audio"] == "`/path/to/input.wav`"
    assert step["Output Audio"] == "`/path/to/output.wav`"
    assert step["Code"] == "# Test code"
    assert step["Execution Results"] == "# Test results"
    
    # Find step_2
    step = coordinator._find_step_by_id(markdown_content, "step_2")
    
    # Check that the step was found
    assert step is not None
    assert step["ID"] == "step_2"
    
    # Find non-existent step
    step = coordinator._find_step_by_id(markdown_content, "step_3")
    
    # Check that the step was not found
    assert step is None 