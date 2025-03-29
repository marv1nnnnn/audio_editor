# AI Audio Editor

An intelligent audio processing tool powered by Gemini AI and Pydantic AI that creates and executes step-by-step plans for audio transformations based on natural language instructions. The system employs a multi-agent architecture for improved planning and execution.

## Features

- Multi-agent architecture with dedicated Planner and Executor agents
- Analyze audio files using AI
- Apply audio transformations (filters, effects, normalization, etc.)
- Generate AI-based audio content
- Mix and concatenate audio files
- Execute complex multi-step audio processing workflows
- Automatic error recovery and replanning
- Quality verification with dedicated QA Agent
- Plan and code critique with specialized Critique Agent
- Interactive user feedback for ambiguous instructions or decisions

## Project Structure

The project consists of four main components:

1. **Multi-agent system** (`audio_editor/agents`): Implements the Planner, Executor, Critique, and QA agents using Pydantic AI
2. **Modular audio tools** (`audio_editor/audio_tools`): A comprehensive library of audio processing functions
3. **ai_audio_editor_v2.py**: Legacy orchestrator for audio processing using monolithic LLM planning
4. **audio_tools.py**: Legacy wrapper for backward compatibility

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
- `--disable-critique`: Disable the Critique Agent for plan and code review
- `--disable-qa`: Disable the QA Agent for output quality verification
- `--log-level`: Set logging level (choices: debug, info, warning, error)

## How It Works

### Multi-agent System

The multi-agent system uses four specialized agents working together:

1. **Planner Agent**:
   - Creates initial step-by-step plans
   - Monitors execution progress
   - Updates step statuses based on execution results
   - Handles recovery and replanning when steps fail
   - Sets checkpoints after successful steps for recovery points

2. **Executor Agent**:
   - Generates Python code for each plan step
   - Refines code when execution errors occur
   - Manages retries for failed steps
   - Validates code before execution

3. **Critique Agent**:
   - Reviews plans for completeness, efficiency, and robustness
   - Evaluates generated code for correctness and best practices
   - Suggests improvements for both plans and code
   - Helps maintain high quality across the workflow

4. **QA Agent**:
   - Verifies the processed audio meets requirements
   - Computes quantitative metrics for audio quality
   - Identifies issues in the final output
   - Recommends further processing if needed

5. **Error Analyzer**:
   - Analyzes error traces to determine root cause
   - Suggests specific fixes for code errors
   - Determines if errors require code fixes or replanning
   - Provides confidence scores for its error analysis

6. **MCP (Master Control Program)**:
   - Executes the Python code in a controlled environment
   - Parses and validates the generated code
   - Returns detailed execution results

7. **Coordinator**:
   - Orchestrates the workflow between agents
   - Handles file management and dependencies
   - Manages the interactive feedback loop with users
   - Tracks overall progress of the plan

This architecture provides better error handling, more robust code generation, and clearer separation of concerns compared to the legacy system.

### Interactive Feedback

The system can interact with you during processing to:

1. **Request clarification** for ambiguous instructions
2. **Ask for confirmation** before applying fixes to errors
3. **Provide choices** between alternative approaches
4. **Allow custom code input** when automatic fixes fail
5. **Request feedback** on quality assessment results

This interactive mode can be disabled with the `--non-interactive` flag for automated batch processing.

### Legacy System

1. **Planning**: Analyzes your request and creates a step-by-step audio processing plan using Gemini AI.
2. **Execution**: Each step is executed sequentially, with results from one step feeding into the next.
3. **Clarification**: If needed, the AI will ask for clarification on specific parameters.
4. **Output**: The final processed audio is saved to the specified output location.

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