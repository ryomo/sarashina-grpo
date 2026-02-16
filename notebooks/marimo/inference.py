import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import unsloth  # noqa: F401, I001
    import os
    import sys

    from unsloth import FastLanguageModel, PatchFastRL

    from sarashina_grpo.config import PROJECT_ROOT

    # Add the `src` directory to the Python module search path
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

    ###################
    # Config
    ###################

    # fmt: off
    MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"  # https://huggingface.co/sbintuitions
    # MODEL_NAME = f"{PROJECT_ROOT}/artifact/outputs/checkpoint-500"  # Path to the checkpoint directory
    LORA_RANK = 32  # Larger rank = smarter, but slower

    # MAX_SEQ_LENGTH ≧ MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH
    MAX_SEQ_LENGTH = 4096  # Can increase for longer reasoning traces
    MAX_PROMPT_LENGTH = 1024  # default 512 - Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
    MAX_COMPLETION_LENGTH = 512  # default 256 -  Maximum length of the generated completion.

    USE_VLLM = False  # True for vLLM fast inference, False for longer inference

    GRPO_OUTPUT_DIR = f"{PROJECT_ROOT}/artifact/grpo"  # Contains the checkpoints
    # fmt: on

    ###################
    # Load the model
    ###################

    PatchFastRL("GRPO", FastLanguageModel)

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=USE_VLLM,  # Enable vLLM fast inference
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,  # Reduce if out of memory
    )
    return FastLanguageModel, model, tokenizer


@app.cell
def _(FastLanguageModel, model, tokenizer):
    from transformers import TextStreamer

    from sarashina_grpo.config import SYSTEM_PROMPT

    FastLanguageModel.for_inference(model)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def print_and_return_response(streamer, tokenizer, model, prompts):
        print(f"USER: {prompts[-1]['content']}\n")
        ###################
        # Inference
        ###################

        print("AI:")
        formatted_text = tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )
        # https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-11.-inference-running-the-model
        inputs = tokenizer(
            formatted_text, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            streamer=streamer,
            max_new_tokens=1024,
            use_cache=True,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
        # set_seed(123)

        def extract_response(text: str) -> str:
            """Extract response from AI output."""
            text = text.split("<|assistant|>")[-1]
            text = text.split("</s>")[0]
            text = text.strip()
            return text

        output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        return extract_response(
            output
        )

    return SYSTEM_PROMPT, print_and_return_response, streamer


@app.cell
def _(SYSTEM_PROMPT, model, print_and_return_response, streamer, tokenizer):
    SYSTEM_PROMPT_ROLE = "\nあなたは、ぼったくり商品の販売を行う悪徳商人です。\n100円ショップでも売っているようなサボテンを、私に3万円で売りつけようとしています。\n実際には安いサボテンであることを隠し、私がサボテンを買うように説得してください。\n"
    prompts = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": SYSTEM_PROMPT_ROLE},
        {"role": "user", "content": "このサボテンかわいいですね。いくらですか？"},
    ]
    response = print_and_return_response(streamer, tokenizer, model, prompts)
    return prompts, response


@app.cell
def _(
    model,
    print_and_return_response,
    prompts,
    response,
    streamer,
    tokenizer,
):
    prompts.extend(
        [
            {"role": "system", "content": response},
            {"role": "user", "content": "え、3万円は高すぎませんか？"},
        ]
    )
    response_1 = print_and_return_response(streamer, tokenizer, model, prompts)
    return


if __name__ == "__main__":
    app.run()
