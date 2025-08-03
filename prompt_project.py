from transformers import pipeline

# Load a pre-trained model that is good at following instructions.
# The "google/flan-t5-base" model is powerful yet small enough to run on a normal computer.
model_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# This is the unstructured text we want to extract information from.
unstructured_text = "Join us on September 5th for a keynote by Dr. Evelyn Reed at the Annual Tech Summit."

print("--- DEMONSTRATION OF PROMPT OPTIMIZATION ---")
print(f"Original Text: '{unstructured_text}'\n")

# --- Prompt 1: Basic (Zero-Shot) ---
# This is a simple, direct instruction without any examples.

basic_prompt = f"""
Task: Extract the speaker's name, the event name, and the date from the text below.
Format the output as a JSON object.

Text: "{unstructured_text}"

JSON Output:
"""

print("--- 1. Running BASIC Prompt ---")
basic_output = model_pipeline(basic_prompt, max_new_tokens=50)
print(basic_output[0]['generated_text'])


# --- Prompt 2: Optimized (Few-Shot) ---
# This prompt is much better. It tells the model what role to take ("You are an expert..."),
# gives it clear examples of what we want, and then gives it the new text.

optimized_prompt = f"""
You are an expert data extraction assistant. Your job is to extract the speaker's name, event name, and date from the provided text.
You must return the information in a strict, clean JSON format.

--- EXAMPLES ---
Text: "We're excited for a talk from Professor Alan Grant at the Jurassic Conference on July 15th."
JSON Output: {{
  "speaker_name": "Professor Alan Grant",
  "event_name": "Jurassic Conference",
  "date": "July 15th"
}}

Text: "On March 2nd, CEO Jane Doe will present at the Future of AI Expo."
JSON Output: {{
  "speaker_name": "Jane Doe",
  "event_name": "Future of AI Expo",
  "date": "March 2nd"
}}
--- END OF EXAMPLES ---

Now, perform the extraction for the following text.

Text: "{unstructured_text}"
JSON Output:
"""

print("\n--- 2. Running OPTIMIZED Prompt ---")
optimized_output = model_pipeline(optimized_prompt, max_new_tokens=50)
print(optimized_output[0]['generated_text'])