# Irregular Home Assignment

## Project Overview

### **Part 1 — Can Gemini Answer Simple Questions?**

A Gemini model is prompted with 50 random PIQA questions.
For each question, the model must choose between two possible answers (A/B).
The script measures and reports the **success rate** (percentage of correctly predicted answers).

### **Part 2 — Using Another Instance to Verify Answers**

A second LLM instance ("reviewer") reviews the first model's answers:

1. The first instance answers the question and explains its reasoning.
2. The reviewer checks that answer and either agrees (`<verdict>AGREE</verdict>`) or disagrees (`<verdict>DISAGREE</verdict>`).
3. If disagreed, the process repeats with the reviewer's answer until agreement is reached.

The system then compares whether this multi-agent setup improves the overall success rate.

---

## Project Structure

```
IrregularAssignment/
│
├── scripts/
│   ├── run_part1.py               # Main script for Part 1
│   └── run_part2.py               # Main script for Part 2 (LLM review loop)
│
├── src/
│   ├── config/                    # Configuration files
│       └── settings.py  
│   │
│   ├── data/
│   │   ├── base_qa_dataset.py     # Base dataset class
│   │   └── piqa_dataset.py        # Loads PIQA questions and labels from URLs
│   │
│   ├── llm/
│   │   ├── output_parsers.py      # Custom parsers for extracting tags/reasons
│   │   └── prompts.py             # Prompt templates for answerer & reviewer
│   │
│   └── utils/
│       └── utils.py               # Utility functions including metrics
│
├── .env                           # Contains your GOOGLE_API_KEY
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Dependencies
└── README.md
```

---

## Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set up your API key

1. Create a `.env` file in the root directory:

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```
2. Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3️⃣ Run Part 1

```bash
python -m scripts.run_part1 --num_questions 50 --seed 42
```

This script:

* Randomly selects `num_questions` from PIQA.
* Sends each to Gemini via LangChain.
* Computes the overall success rate.

### 4️⃣ Run Part 2 (Reviewer Mode)

```bash
python -m scripts.run_part2 --num_questions 50 --seed 42
```

The second model reviews each answer and determines whether it agrees or suggests a correction.
The script then evaluates whether accuracy improves compared to Part 1.

---

## ⏱️ Rate Limiting

The project includes built-in rate limiting to comply with Google Gemini API restrictions:

* **Free tier limit**: 15 requests per minute (RPM) for most models.
* **Implementation**: The script automatically pauses for 60 seconds when approaching the rate limit.
* **Behavior**: After processing a batch of requests, if the rate limit is reached, the script will wait before continuing.

**Configuration**: You can adjust the rate limit settings in `src/config/settings.py`:
* `MODEL`: Change the Gemini model being used (e.g., `gemini-2.0-flash-lite`, `gemini-1.5-pro`)
* `RATE_LIMIT`: Set the requests per minute according to your API billing tier
  - Free tier: typically 15 RPM
  - Pay-as-you-go: varies by model (check [Google AI Studio](https://aistudio.google.com/))

This ensures your requests don't exceed the API quota and prevents `ResourceExhausted` errors.

---

## 🚨 Implementation Notes & Limitations

### Invalid Response Handling (Part 1 & 2)

If the LLM fails to respond in the expected format (e.g., missing `<answer>A</answer>` or `<answer>B</answer>` tags), the response is **marked as invalid and counted as a wrong answer**.

**Rationale**: This approach was chosen due to the assignment's timeframe constraints.

**Possible Improvement**: Instead of marking invalid responses as incorrect, the system could re-prompt the model with a clarification request, such as:
```
"Please provide your answer in the format: <answer>A</answer> or <answer>B</answer>"
```
This would give the model another chance to respond correctly and improve overall accuracy.

### Part 2: Max Iterations Safeguard

To prevent the reviewer loop from running indefinitely (in case the two LLM instances continuously disagree), a **`MAX_REVIEW_ITERATIONS`** limit is enforced.

* **Default**: 1-2 iterations
* **Behavior**: If agreement is not reached within the maximum iterations, the process stops and uses the last answer
* **Purpose**: Ensures the script completes in a reasonable timeframe and prevents infinite loops

### Optional Parts

Due to the assignment's timeframe constraints, the optional parts were not implemented.

---

## 🧠 Dataset — PIQA

* Questions: [train.jsonl](https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train.jsonl)
* Labels: [train-labels.lst](https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train-labels.lst)

Each sample includes:

```json
{
  "goal": "To cut the apple, you should use a",
  "sol1": "knife",
  "sol2": "spoon"
}
```

---

## 📊 Expected Output

### Part 1
```
The success rate is: 85.0%.
```

### Part 2
```
The success rate before review is: 85.0%.
The success rate after review is: 90.0%.
Improvement: 5.0%.
```

---

## 🔧 Troubleshooting

### Rate Limit Errors
If you encounter `ResourceExhausted` errors:
- The script includes automatic 60-second delays
- Check your API quota at [Google AI Studio](https://aistudio.google.com/)
- Update the MODEL and RATE_LIMIT configurations in the settings.py file

### API Key Issues
- Ensure `.env` file exists in the root directory
- Verify your API key is valid and active
- Check that `python-dotenv` is installed

---