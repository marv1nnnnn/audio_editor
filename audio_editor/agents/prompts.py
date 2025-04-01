"""
Default prompts for AI-powered audio analysis and generation.
"""

# Default prompts for AUDIO_QA
DEFAULT_AUDIO_QUALITY_PROMPT = """
Analyze this audio file and provide a detailed assessment of its quality:

1. Audio Quality
   - Clarity and definition
   - Presence of artifacts or noise
   - Dynamic range
   
2. Frequency Balance
   - Low-end (bass) quality and amount
   - Mid-range clarity
   - High-frequency detail and presence
   
3. Technical Issues
   - Identify clipping, distortion, or compression artifacts
   - Point out frequency imbalances
   - Highlight noise problems (hiss, hum, room noise)
   
4. Professional Standards
   - How does this compare to professional standards?
   - What improvements would be needed for broadcast or commercial release?
   
5. Specific Recommendations
   - Suggest 3-5 precise adjustments that would improve the sound
   - Rank them by priority

Provide the assessment in a detailed, technical manner that could guide an audio engineer.
"""

# Default prompt for comparing original vs processed audio
DEFAULT_AUDIO_COMPARISON_PROMPT = """
Compare these two audio files in detail:

1. Sonic Differences
   - What are the most noticeable differences between the files?
   - How has the frequency response changed?
   - How has the dynamic range changed?

2. Improvements
   - What specific aspects sound better in the processed file?
   - Which technical issues have been resolved?
   - How has clarity, definition, or balance improved?

3. Potential Regressions
   - Has anything gotten worse in the processing?
   - Are there any new artifacts or issues introduced?
   - Has anything important from the original been lost?

4. Overall Assessment
   - Which file sounds better overall and why?
   - Is the processing appropriate for the content?
   - What is the most successful aspect of the processing?

5. Next Steps
   - What further processing would you recommend?
   - Are there any specific techniques that should be applied next?
   - What should be prioritized in the next processing step?

Please be specific and technical in your analysis, focusing on audio engineering concepts.
"""

# Default prompt for comparing multiple processing approaches
DEFAULT_MULTI_COMPARISON_PROMPT = """
Compare these audio files representing different processing approaches:

1. Approach Characteristics
   - What seems to be the focus of each processing approach?
   - How do the approaches differ in their handling of frequency balance?
   - How do they differ in dynamic processing?

2. Strengths and Weaknesses
   - What are the unique strengths of each approach?
   - What technical weaknesses does each approach have?
   - Which aspects of the audio are best handled by each approach?

3. Technical Assessment
   - How do the approaches differ in terms of clarity and definition?
   - How do they handle problematic frequencies or artifacts?
   - Which approach maintains the most natural sound?

4. Suitability for Purpose
   - Which approach would be best for general listening?
   - Which would be best for professional or broadcast use?
   - Which preserves the original artistic intent best?

5. Recommendation
   - Which approach is overall most successful?
   - Could elements from different approaches be combined?
   - What final adjustments would you recommend to the best approach?

Be detailed and objective, focusing on the technical qualities rather than subjective preference.
"""

# Default prompt for generating reference audio
DEFAULT_AUDIO_GENERATION_PROMPT = """
Create audio that has the following characteristics:

1. Overall sound profile:
   - Clear, balanced frequency spectrum
   - Professional production quality
   - Appropriate dynamic range
   
2. Low frequency characteristics:
   - Tight, defined bass without muddiness
   - Controlled sub-bass that doesn't overpower
   
3. Mid-range characteristics:
   - Clear vocal presence (if applicable)
   - Balanced instrument separation
   - No boxy or honky qualities
   
4. High frequency characteristics:
   - Detailed but not harsh treble
   - Air and sparkle without sibilance
   - Natural cymbal and high-frequency decay
   
5. Spatial characteristics:
   - Appropriate stereo width
   - Depth and dimension in the mix
   - Balanced positioning of elements

Generate audio that would serve as an excellent reference for professional audio production.
"""

# Prompt templates for iterative improvement
ITERATIVE_IMPROVEMENT_TEMPLATE = """
Based on the current audio state and analysis, suggest the next processing step with precise parameters:

1. Current Audio Assessment:
{current_assessment}

2. Remaining Issues:
{remaining_issues}

3. Recommended Next Processing:
   - What specific processing technique should be applied next?
   - What exact parameters should be used? (frequency values, dB amounts, ratio values, etc.)
   - What specific improvement will this achieve?

4. Expected Outcome:
   - What measurable improvement should result?
   - How will the subjective quality change?
   - What should be verified after this processing?

Be extremely specific with technical parameters - use exact frequency values, dB amounts, threshold settings, etc.
"""

# Prompt for final quality verification
FINAL_QUALITY_VERIFICATION_PROMPT = """
Provide a final quality assessment of this processed audio:

1. Professional Standards
   - Does this meet broadcast or commercial release standards?
   - How close is it to professional reference material?
   - What is the overall production quality level?

2. Technical Quality
   - Frequency balance assessment
   - Dynamic range and loudness assessment
   - Noise floor and artifact assessment
   - Stereo image assessment

3. Improvements Made
   - What were the most significant improvements from the original?
   - Which technical issues were successfully addressed?
   - How has the overall listenability improved?

4. Remaining Room for Improvement
   - What minor issues still remain?
   - What would be needed for truly professional quality?
   - What advanced techniques could take this to the next level?

5. Verification Statement
   - Is this audio ready for its intended purpose?
   - Does it meet the requirements specified in the task description?
   - What is your final verdict on the quality?

Provide this assessment with the detail and specificity of a professional audio engineer's final report.
""" 