# AI Audio Editor

An intelligent audio processing tool powered by Gemini AI that creates and executes step-by-step plans for audio transformations based on natural language instructions.

## Features

- Analyze audio files using AI
- Apply audio transformations (filters, effects, normalization, etc.)
- Generate AI-based audio content
- Mix and concatenate audio files
- Execute complex multi-step audio processing workflows

## Project Structure

The project consists of two main components:

1. **ai_audio_editor_v2.py**: Orchestrates the audio processing workflow using LLM planning
2. **audio_tools.py**: Contains the audio processing functions that are called by the main module

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

Run the editor with a task description and input file:

```bash
python ai_audio_editor_v2.py --task "Enhance vocals and increase bass" --input input.wav --output enhanced.wav
```

Alternatively, if installed as a package:

```bash
audio-editor --task "Enhance vocals and increase bass" --input input.wav --output enhanced.wav
```

### Command Line Arguments

- `--task`: Description of the audio transformation task (required)
- `--input`: Path to the input audio file (required)
- `--output`: Path for the output audio file (default: "output.wav")
- `--model`: Gemini model to use (default: "gemini-1.5-flash")

## How It Works

1. **Planning**: The system analyzes your request and creates a step-by-step audio processing plan using Gemini AI.
2. **Execution**: Each step is executed sequentially, with results from one step feeding into the next.
3. **Clarification**: If needed, the AI will ask for clarification on specific parameters.
4. **Output**: The final processed audio is saved to the specified output location.

## Available Audio Tools

The system includes various audio processing functions:

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

## Requirements

- Python 3.9+
- Gemini API key (https://ai.google.dev/)
- Dependencies listed in pyproject.toml

## License

[Include license information]