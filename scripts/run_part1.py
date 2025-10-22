from src.config.settings import GOOGLE_API_KEY, MODEL, LLM_OUTPUT_TAGS, RATE_LIMIT
from langchain_google_genai import ChatGoogleGenerativeAI
from src.data.piqa_dataset import PIQADataset
from src.llm.prompts import get_part1_prompt_template
from src.llm.output_parsers import AnsewerOutputParser
from argparse import ArgumentParser
from src.utils.utils import get_n_random_samples, get_success_rate
import time

def main(args):

    num_questions = args.num_questions
    seed = args.seed

    piqa_dataset = PIQADataset()

    samples = get_n_random_samples(piqa_dataset, num_questions, seed)

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    prompt_template = get_part1_prompt_template()

    output_parser = AnsewerOutputParser(tags=LLM_OUTPUT_TAGS)

    chain = prompt_template | llm | output_parser

    for i in range(0, len(samples), RATE_LIMIT):
        batch = samples[i:i+RATE_LIMIT]
        batch_results = chain.batch(batch, return_exceptions=True)
        if i == 0:
            results = batch_results
        else:
            results.extend(batch_results)
        time.sleep(60)
    sucess_rate = get_success_rate(samples=samples, llm_results=results)

    return sucess_rate
    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    succes_rate = main(args)
    print(f"The success rate is: {succes_rate * 100}%.")