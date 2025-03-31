# AI Audio Editor

An intelligent audio processing tool powered by Gemini AI and Pydantic AI that creates and executes step-by-step plans for audio transformations based on natural language instructions. The system employs a multi-agent architecture with a Markdown-centric workflow for improved planning, execution, and transparency.

## Features

- Markdown-centric workflow for full transparency and traceability
- Multi-agent architecture with dedicated agents for planning, code generation, execution, error analysis, and QA
- Analyze audio files using AI
- Apply audio transformations (filters, effects, normalization, etc.)
- Generate AI-based audio content
- Mix and concatenate audio files
- Execute complex multi-step audio processing workflows
- Automatic error recovery and replanning
- Quality verification with dedicated QA Agent
- Interactive user feedback for ambiguous instructions or decisions

## Project Structure

The project consists of three main components:

1. **Multi-agent system** (`audio_editor/agents`): Implements the Planner, Code Generator, Executor, Error Analyzer, and QA agents using Pydantic AI
2. **Modular audio tools** (`audio_editor/audio_tools`): A comprehensive library of audio processing functions
3. **Workflow Markdown** (`docs/`): Structured Markdown files that track the entire processing pipeline

The audio tools package is organized into the following modules:
- `audio_tools/config.py`: Configuration constants
- `audio_tools/utils.py`: Utility functions
- `audio_tools/io.py`: File I/O functions
- `audio_tools/manipulation.py`: Basic audio manipulation functions
- `audio_tools/effects.py`: Audio effects processing
- `audio_tools/volume.py`: Volume adjustment functions
- `audio_tools/ai.py`: AI-powered analysis and generation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio_editor.git
   cd audio_editor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install uv (if not already installed):
   ```bash
   # Install uv (faster Python package installer)
   curl -sSf https://astral.sh/uv/install.sh | bash
   # On Windows PowerShell:
   # (Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing).Content | powershell -c -
   ```

4. Install dependencies with uv and pyproject.toml:
   ```bash
   # Install in development mode
   uv pip install -e .
   
   # Or install with all dependencies
   uv pip install .
   ```

5. Set up environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"  # On Windows: set GEMINI_API_KEY=your-gemini-api-key
   ```

## Usage

### Multi-agent System (Recommended)

Run the editor with a task description and input file:

```bash
audio-editor --task "Enhance vocals and increase bass" --input input.wav --output enhanced.wav
```

### Direct Script Usage

You can also run the main script directly using `uv run` without installing the package:

```bash
uv run audio_editor/agents/main.py \
  --task "Enhance vocals and increase bass" \
  --input ./my_audio.wav \
  --output ./enhanced.wav
```

This method supports all the same arguments as the installed version:
- `--task`, `-t`: Description of what you want to do with the audio
- `--input`, `-i`: Path to your input WAV file
- `--output`, `-o`: Where to save the processed file
- `--model`, `-m`: Gemini model to use (default: "gemini-2.0-flash")
- `--working-dir`, `-w`: Directory for temporary files
- `--transcript`: Transcript of the audio, if available

You can view all available options with:
```bash
uv run audio_editor/agents/main.py --help
```

Note: Make sure you have set up your environment variables (like `GEMINI_API_KEY`) before running the script.

### Command Line Arguments

- `--task`, `-t`: Description of the audio transformation task (required)
- `--input`, `-i`: Path to the input audio file (required)
- `--output`, `-o`: Path for the output audio file (optional)
- `--model`, `-m`: Gemini model to use (default: "gemini-2.0-flash")
- `--working-dir`, `-w`: Directory for intermediate files (optional)
- `--transcript`: Transcript of the audio file, if available (optional)
- `--non-interactive`: Disable interactive user feedback, use default responses
- `--disable-error-analyzer`: Disable the Error Analyzer for fixing failed steps
- `--disable-qa`: Disable the QA Agent for output quality verification
- `--log-level`: Set logging level (choices: debug, info, warning, error)
- `--workflow-file`: Path to an existing workflow file to resume processing (optional)

## How It Works

### Markdown-Centric Workflow

The system uses a central Markdown file as the single source of truth for the entire audio processing pipeline. This provides full transparency and traceability of the processing steps and decisions.

1. **Coordinator**:
   - Creates a unique `workflow_{timestamp}_{hash}.md` file for each processing request
   - Orchestrates the agent calls based on the current state in the Markdown file
   - Updates the Markdown file with agent outputs and state changes
   - Tracks the workflow log for debugging and auditing

2. **Workflow Markdown Structure**:
   - **Request Summary**: The original user request and input details
   - **Product Requirements Document (PRD)**: Generated by the Planner Agent
   - **Processing Plan & Status**: Steps with detailed status tracking
   - **Final Output**: Overall results and QA assessment
   - **Workflow Log**: Timestamped log of all actions taken

3. **Step Structure**:
   - Each step includes ID, description, status, input/output paths, code, and execution results
   - Steps transition through statuses: PENDING → READY → RUNNING → CODE_GENERATED → EXECUTING → DONE or FAILED
   - Variable references like `$OUTPUT_STEP_1` connect outputs from one step to inputs of the next

This approach allows for:
- Complete transparency of the entire process
- Easy debugging and resumability
- Better error recovery
- Clear tracking of intermediate files
- Simplified agent interactions

### Multi-agent System

The multi-agent system uses specialized agents working together:

1. **Planner Agent**:
   - Creates the PRD and initial step-by-step plans directly in the Markdown file
   - Defines specific, concrete steps with detailed descriptions
   - Sets the initial input/output paths for each step

2. **Code Generation Agent**:
   - Reads the Markdown content and the specific step ID
   - Generates a single line of Python code to implement that step
   - Uses the appropriate tool based on the step description and PRD
   - Resolves input/output path variables correctly

3. **Executor**:
   - Executes the generated code in a controlled environment
   - Updates the Markdown with execution results
   - Changes step status to DONE or FAILED

4. **Error Analyzer**:
   - Analyzes failed steps by reading the code and error message
   - Generates fixed code that is updated in the Markdown
   - Allows the step to be retried

5. **QA Agent**:
   - Verifies the processed audio meets requirements defined in the PRD
   - Updates the Final Output section with quality assessment
   - Recommends further processing if needed

### Interactive Feedback

The system can interact with you during processing to:

1. **Request clarification** for ambiguous instructions
2. **Ask for confirmation** before applying fixes to errors
3. **Provide choices** between alternative approaches
4. **Allow custom code input** when automatic fixes fail
5. **Request feedback** on quality assessment results

This interactive mode can be disabled with the `--non-interactive` flag for automated batch processing.

## Available Audio Tools

The system includes various audio processing functions organized into modules:

### File Operations
- Read/write audio files
- Get audio length
- Clip audio segments

### Audio Effects
- Apply high-pass, low-pass, and band-pass filters
- Add noise (Gaussian, colored)
- Time stretching
- Pitch shifting
- Normalize audio loudness
- Increase/decrease volume

### Audio Manipulation
- Split audio at specified breakpoints
- Mix multiple audio files with offsets
- Concatenate audio files
- Clip audio to specific time ranges

### AI Features
- Analyze audio content with Gemini
- Compare multiple audio files
- Generate audio from text descriptions

## Advanced Use Cases

The AI Audio Editor can handle complex multi-step tasks like:

- "Clean up my podcast recording by removing background noise and normalizing volume"
- "Make this guitar recording sound like it's being played in a large cathedral"
- "Create a lofi hip-hop version of this piano sample"
- "Add rain sounds to this nature recording and enhance the bird calls"
- "Analyze my band's mix, identify mastering issues, and fix them"
- "Remove background conversation from my voice memo while preserving audio quality"

## Using the Audio Tools Library

You can use the audio tools library directly in your code:

```python
from audio_editor.audio_tools import CLIP, AUDIO_QA, LOUDNESS_NORM, ADD_NOISE

# Analyze audio content
analysis = AUDIO_QA("input.wav", "Describe the audio quality and content")
print(analysis)

# Extract a segment from 10s to 30s
clipped = CLIP("input.wav", offset=30.0, onset=10.0, out_wav="segment.wav")

# Normalize loudness to -14 LUFS (standard for streaming)
normalized = LOUDNESS_NORM(clipped, volume=-14.0)

# Add subtle noise
final = ADD_NOISE(normalized, min_amplitude=0.001, max_amplitude=0.005)
```

## Development

### Dependency Management

This project uses modern Python packaging with `pyproject.toml` and [uv](https://github.com/astral-sh/uv) for dependency management.

#### Adding Dependencies

To add new dependencies:

```bash
# Add a dependency and update pyproject.toml
uv pip install package-name --update-pyproject
```

#### Development Installation

For development:

```bash
# Install in development mode
uv pip install -e .
```

#### Creating a Distribution

To build the package:

```bash
uv pip build
```

### Testing

The project uses pytest for testing. The test suite includes:

1. Unit tests for individual components
2. Integration tests for agent interactions
3. End-to-end tests for complete workflows

To run the tests:

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=audio_editor tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run tests in parallel
pytest -n auto

# Run with detailed output
pytest -v
```

The test suite provides comprehensive coverage of:
- Model validation and constraints
- Agent interaction patterns
- Error handling and recovery
- Checkpoint system
- Concurrent processing
- End-to-end workflows

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures and utilities
├── unit/                 # Unit tests
│   ├── test_models.py    # Pydantic model tests
│   ├── test_dependencies.py  # Dependency class tests
│   └── test_mcp.py      # Master Control Program tests
├── integration/          # Integration tests
│   └── test_agent_interactions.py  # Agent interaction tests
└── e2e/                 # End-to-end tests
    └── test_audio_processing.py  # Complete workflow tests
```

### Writing Tests

When adding new features, please ensure:
1. Unit tests cover the new functionality
2. Integration tests verify interaction with existing components
3. End-to-end tests demonstrate the feature in a complete workflow
4. All tests are properly documented
5. Test fixtures are reusable and maintainable

## Requirements

- Python 3.9+
- Gemini API key (https://ai.google.dev/)
- Dependencies listed in pyproject.toml

## License

[Include license information]