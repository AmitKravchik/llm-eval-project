from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.exceptions import OutputParserException

from src.config.settings import LLM_OUTPUT_TAGS
from pydantic import Field
from typing import List, Dict, Any

class AnsewerOutputParser(BaseOutputParser):

    tags: Dict = Field(default_factory=dict)

    def parse(self, text: str) -> dict:

        res = {"raw_answer": text}
        for tag_name in self.tags.keys():
            res[tag_name] = self._extract_part_from_tags(text=text, part=tag_name)

        res["predicted_label"] = self._convert_final_answer(final_answer=res["final_answer"])

        return res

    def _extract_part_from_tags(self, text: str, part: str = "final_answer") -> str:

        start_index = text.find(self.tags[part][0])
        end_index = text.find(self.tags[part][1])

        if start_index == -1 or end_index == -1:
            return "invalid" # Indicate invalid if tags not found

        extracted = text[start_index + len(self.tags[part][0]):end_index].strip()
        return extracted

    def _convert_final_answer(self, final_answer: str) -> str:
        if final_answer == "A":
            return 0
        elif final_answer == "B":
            return 1
        elif final_answer == "invalid":
            return -1
        else:
            raise OutputParserException(f"{final_answer} is not a valid final answer.")


