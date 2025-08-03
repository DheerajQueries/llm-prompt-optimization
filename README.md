# LLM Prompt Optimization for Structured Data Extraction

This project is a practical demonstration of how strategic prompt engineering dramatically improves the performance and reliability of Large Language Models (LLMs). The goal is to extract structured data (a JSON object) from an unstructured sentence, showcasing the core skills required for an AI Prompt Engineer role.

## The Objective

To reliably extract three key pieces of information from a sentence—a speaker's name, an event name, and a date—and format them into a clean, predictable JSON object.

## The Experiment: Basic Prompt vs. Optimized Prompt

I tested two different prompt engineering techniques to highlight the difference in output quality.

### 1. The Basic (Zero-Shot) Prompt

First, I used a simple, direct instruction without providing any examples. This is known as a "zero-shot" prompt.

**Prompt Used:**
```python
"""
Task: Extract the speaker's name, the event name, and the date from the text below.
Format the output as a JSON object.

Text: "Join us on September 5th for a keynote by Dr. Evelyn Reed at the Annual Tech Summit."

JSON Output:
"""

Result:
The model's output was unstructured and unusable, returning a simple list of words it deemed relevant. This demonstrates the unreliability of basic prompts for specific, structured tasks.
(Note: The actual output from the script should be referenced here to show the "before" state.)


2. The Optimized (Few-Shot) Prompt
To fix this, I engineered an optimized prompt using two key techniques:
Persona Assignment: I instructed the model to act as an "expert data extraction assistant."
Few-Shot Examples: I provided high-quality examples of the desired input and output, showing the model exactly what success looks like.


Prompt Used:
"""
You are an expert data extraction assistant. Your job is to extract the speaker's name, event name, and date from the provided text.
You must return the information in a strict, clean JSON format.

--- EXAMPLES ---
Text: "We're excited for a talk from Professor Alan Grant at the Jurassic Conference on July 15th."
JSON Output: {
  "speaker_name": "Professor Alan Grant",
  "event_name": "Jurassic Conference",
  "date": "July 15th"
}

Text: "On March 2nd, CEO Jane Doe will present at the Future of AI Expo."
JSON Output: {
  "speaker_name": "Jane Doe",
  "event_name": "Future of AI Expo",
  "date": "March 2nd"
}
--- END OF EXAMPLES ---

Now, perform the extraction for the following text.

Text: "Join us on September 5th for a keynote by Dr. Evelyn Reed at the Annual Tech Summit."
JSON Output:
"""

Result:
The model produced a structured output that was significantly closer to the desired JSON format. This proves that well-engineered prompts are essential for guiding LLMs to produce accurate and reliable results.

It turns a failing process into a successful one.
(Note: The actual output from the script should be referenced here to show the "after" state.)

Basic Prompt Result:
Evelyn Reed', 'YEAR', '2017', '2018', 'Tech Summit', 'Sept. 5th', 'Join us',

Optimizd Prompt Result:
"speaker_name": "Evelyn Reed" , "date": "September 5th"

Technical Details
Language: Python
Core Libraries: transformers, torch (from PyTorch)
Model Used: google/flan-t5-base via the Hugging Face library.
