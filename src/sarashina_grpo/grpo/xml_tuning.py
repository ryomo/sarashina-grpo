import re
from typing import Callable, List, Optional

from transformers import PreTrainedTokenizerBase


class XMLTag:
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    RESPONSE_OPEN = "<response>"
    RESPONSE_CLOSE = "</response>"


class XMLTuning:

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.tokenizer = tokenizer

    def get_reward_functions(self) -> list[Callable[..., List[float]]]:
        """
        Return all reward functions as a list.
        * See about reward functions:
            https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
        """
        return [
            self.xmlcount_reward_func,
            self.soft_format_reward_func,
            self.strict_format_reward_func,
        ]

    def xmlcount_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function based on XML tag counting."""

        def _count_xml(text) -> float:
            """Count XML tags to evaluate format compliance."""
            count = 0.0
            if text.count(XMLTag.THINK_OPEN) == 1:
                count += 0.125
            if text.count(XMLTag.THINK_CLOSE) == 1:
                count += 0.125
            if text.count(XMLTag.RESPONSE_OPEN) == 1:
                count += 0.125
                count -= len(text.split(XMLTag.RESPONSE_CLOSE)[-1]) * 0.001
            if text.count(XMLTag.RESPONSE_CLOSE) == 1:
                count += 0.125
                count -= (len(text.split(XMLTag.RESPONSE_CLOSE)[-1]) - 1) * 0.001
            return count

        contents = [completion[0]["content"] for completion in completions]
        return [_count_xml(c) for c in contents]

    def soft_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has approximate expected format."""
        pattern = rf"{XMLTag.THINK_OPEN}.*?{XMLTag.THINK_CLOSE}\s*{XMLTag.RESPONSE_OPEN}.*?{XMLTag.RESPONSE_CLOSE}"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def strict_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has the exact expected format."""
        pattern = rf"^{XMLTag.THINK_OPEN}.*?{XMLTag.THINK_CLOSE}\n{XMLTag.RESPONSE_OPEN}.*?{XMLTag.RESPONSE_CLOSE}$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def _extract_xml_response(self, text: str) -> str:
        """Extract response from XML format response."""
        response = text.split(XMLTag.RESPONSE_OPEN)[-1]
        response = response.split(XMLTag.RESPONSE_CLOSE)[0]
        response = response.strip()
        print(f"Extracted response: {response}")
        return response
