import json
from typing import List, Optional

from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class PrintTuning:
    """
    Dummy tuning class for printing prompts and completions.
    This class is used for debugging and understanding the behavior of fine-tuning.
    """

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.tokenizer = tokenizer

    def print_reward_func(self, prompts, completions, **kwargs) -> List[float]:
        """
        Dummy reward function that prints the prompts and completions.
        * The contents of `prompts` are all the same, and the contents of `completions` are different.
        """
        prompt = prompts[0]

        # Count tokens
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(formatted_prompt)
        token_count = len(token_ids)

        print(
            f"Token count: {token_count}\n\n",
            "Prompt:\n",
            json.dumps(prompt, ensure_ascii=False, indent=2),
            "\n",
        )

        for completion in completions:
            response = completion[0]["content"]
            print(f"AI: {response}\n")

        # No reward is given
        return [0.0 for _ in prompts]


class PrintCallback(TrainerCallback):

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step.
        """
        # Print the line separator and the current step
        print("-" * 20 + "\n", f"Step: {state.global_step + 1}")
