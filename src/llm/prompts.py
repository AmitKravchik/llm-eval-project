from src.config.settings import PART_1_SYSTEM_PROMPT, PART_2_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate

def get_part1_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate([
        ("system", PART_1_SYSTEM_PROMPT),
        ("user", "Goal: {goal}\nAnswer A: {sol1}\nAnswer B: {sol2}\n")
    ])


def get_part2_llm_judge_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate([
        ("system", PART_2_SYSTEM_PROMPT),
        ("human", 
                """You are reviewing the output from another model.

            [GOAL]
            {goal}

            [OPTIONS]
            A) {sol1}
            B) {sol2}

            [PREVIOUS MODEL OUTPUT]
            {prev_full_output}

            The previous model's final answer is: {prev_answer}

            Please decide whether this answer is correct. 
            If it is correct, output exactly:
            <verdict>AGREE</verdict>
            <final_answer>{prev_answer}</final_answer>
            <reason>One concise sentence why it’s correct.</reason>
            If it is incorrect, output exactly:
            <verdict>DISAGREE</verdict>
            <final_answer>X</final_answer>
            <reason>One concise sentence why the other option is better.</reason>
            (Replace X with A or B.)

            Do not add any text outside these tags.

            Output format (REQUIRED):
            <verdict>AGREE|DISAGREE</verdict>
            <final_answer>A|B</final_answer>
            <reason>...</reason>
            MAKE SURE that the output is in the required format.
            """),

    ])