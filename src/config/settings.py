from dotenv import load_dotenv
import os

load_dotenv()

# Dynamic configurations

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Static configurations

DATASET_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train.jsonl"
LABELS_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train-labels.lst"


ANS_START_TAG = "<answer>"
ANS_END_TAG = "</answer>"
LLM_OUTPUT_TAGS: dict = {"final_answer": ("<answer>", "</answer>")}


PART_1_SYSTEM_PROMPT = f"""I will give you a question or sentence to complete and two possible answers. Please answer either A or B, depending on which answer is better. You may write down your reasoning but please write your final answer (either A or B) between the {ANS_START_TAG} and {ANS_END_TAG} tags"""
PART_2_SYSTEM_PROMPT = """You are a careful reviewer. Your goal is to decide whether a prior multiple-choice answer (A/B) is correct. \nAlways return your result in the exact requested format."""

MODEL = "gemini-2.5-flash-lite"
RATE_LIMIT = 15
MAX_REVIEW_ITERATIONS = 50

JUDGE_OUTPUT_TAGS: dict = {"verdict": ("<verdict>", "</verdict>"),
                "final_answer": ("<final_answer>", "</final_answer>"),
                "reason": ("<reason>", "</reason>")}
                

