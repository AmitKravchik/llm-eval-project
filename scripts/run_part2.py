from src.config.settings import GOOGLE_API_KEY, MODEL, MAX_REVIEW_ITERATIONS, JUDGE_OUTPUT_TAGS, LLM_OUTPUT_TAGS, RATE_LIMIT
from langchain_google_genai import ChatGoogleGenerativeAI
from src.data.piqa_dataset import PIQADataset
from src.llm.prompts import get_part1_prompt_template
from src.llm.output_parsers import AnsewerOutputParser
from argparse import ArgumentParser
from src.llm.prompts import get_part2_llm_judge_prompt_template
from src.utils.utils import get_n_random_samples, get_success_rate
import time



def validate_answer_with_llm(sample, llm_result, model, google_api_key, max_iterations=50, request_count=0, rate_limit=15):

    prev_answer = llm_result.get("final_answer")
    prev_full_output = llm_result.get("raw_answer", "")

    last_result = {"reason": prev_full_output, "final_answer": prev_answer}

    for _ in range(max_iterations):
        if request_count == rate_limit:
            time.sleep(60)
            request_count = 0
        judge = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key)
        prompt_template = get_part2_llm_judge_prompt_template()
        output_parser = AnsewerOutputParser(tags=JUDGE_OUTPUT_TAGS)
        chain = prompt_template | judge | output_parser
        request_count += 1
        input_data = {
            "goal": sample.get("goal"),
            "sol1": sample.get("sol1"),
            "sol2": sample.get("sol2"),
            "prev_full_output": prev_full_output,
            "prev_answer": prev_answer,
        }

        last_result = chain.invoke(input_data)
        curr_final = last_result.get("final_answer")

        # If judge agrees with the previous answer -> accept
        if curr_final == prev_answer:
            return last_result, request_count

        # Otherwise adopt the judge's answer+reason and continue with a new judge
        prev_answer = curr_final
        prev_full_output = last_result.get("reason", prev_full_output)

    return last_result, request_count



def main(args):

    num_questions = args.num_questions
    seed = args.seed

    piqa_dataset = PIQADataset()

    samples = get_n_random_samples(piqa_dataset, num_questions, seed)

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
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
    original_sucess_rate = get_success_rate(samples=samples, llm_results=results)

    validated_results = []
    request_count = 0
    for sample, llm_result in zip(samples, results):
        if request_count == RATE_LIMIT:
            time.sleep(60)
            request_count = 0
        validated_result, request_count = validate_answer_with_llm(
            sample=sample,
            llm_result=llm_result,
            model=MODEL,
            google_api_key=GOOGLE_API_KEY,
            max_iterations=MAX_REVIEW_ITERATIONS,
            request_count=request_count,
            rate_limit=RATE_LIMIT,
        )
        validated_results.append(validated_result)

    reviewed_sucess_rate = get_success_rate(samples=samples, llm_results=validated_results)

    return original_sucess_rate, reviewed_sucess_rate
    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    succes_rate = main(args)
    print(f"The success rate before review is: {succes_rate[0] * 100}%.")
    print(f"The success rate after review is: {succes_rate[1] * 100}%.")
    print(f"Improvement: {(succes_rate[1] - succes_rate[0]) * 100}%.")