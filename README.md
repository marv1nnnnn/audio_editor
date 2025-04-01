# AI Audio Editor

An intelligent audio processing tool powered by Gemini AI that creates and executes step-by-step plans for audio transformations based on natural language instructions.

## Quick Start

1. **Install**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/audio_editor.git
   cd audio_editor

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -e .
   ```

2. **Set Environment Variables**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

3. **Run**
   ```bash
   audio-editor --task "Enhance vocals and increase bass" --input input.wav --output enhanced.wav
   ```

## How It Works

The system uses a Markdown-centric workflow with multiple specialized AI agents:

1. **Planner Agent**
   - Creates a detailed plan based on your request
   - Breaks down complex tasks into simple steps
   - Generates a Product Requirements Document (PRD)

2. **Code Generation Agent**
   - Converts each step into executable Python code
   - Uses built-in audio processing tools
   - Ensures proper input/output handling
   - Accesses audio content directly for better understanding (new!)

3. **Executor Agent**
   - Runs the generated code safely
   - Handles errors and retries
   - Tracks progress and results

4. **QA Agent**
   - Verifies output quality
   - Suggests improvements
   - Ensures requirements are met
   - Compares original and processed audio directly (new!)

### Workflow Example

For a request like "Enhance vocals and increase bass":

1. **Planning Phase**
   - Analyzes audio content
   - Creates step-by-step plan
   - Sets quality requirements

2. **Execution Phase**
   - Isolates vocal frequencies
   - Applies enhancement filters
   - Boosts bass frequencies
   - Balances overall mix

3. **Quality Check**
   - Verifies vocal clarity
   - Checks bass levels
   - Ensures no distortion

All progress is tracked in a Markdown file (`workflow_{timestamp}_{hash}.md`) containing:
- Original request and requirements
- Step-by-step progress
- Generated code and results
- Quality verification results

## Available Audio Tools

- **File Operations**: Read/write audio, get length, clip segments
- **Effects**: Filters, noise reduction, time stretching, pitch shifting
- **Volume**: Normalization, dynamic adjustment
- **Mixing**: Combine audio, adjust balance
- **Analysis**: AI-powered content analysis

## New Feature: Direct Audio Content

The system now passes audio directly to AI models using Pydantic AI's audio input capabilities, allowing:
- Better understanding of audio characteristics
- More accurate code generation
- Improved quality assessment
- Enhanced error analysis

This feature enables the model to "hear" the audio rather than just seeing file paths, resulting in more accurate processing.

## Command Line Options

```bash
audio-editor \
  --task "Description of what to do" \
  --input input.wav \
  --output output.wav \
  [--model MODEL] \
  [--working-dir DIR] \
  [--transcript TEXT] \
  [--non-interactive] \
  [--disable-qa] \
  [--disable-audio-content] \
  [--log-level {debug,info,warning,error}]
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run with debug logging
audio-editor --task "..." --input ... --log-level debug
```

## Requirements

- Python 3.11+
- Gemini API key (or other supported model provider)
- pydantic-ai >= 0.2.0 (for audio content support)
- Dependencies listed in pyproject.toml

## License

[Include license information]