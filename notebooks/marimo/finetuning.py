import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import unsloth  # noqa: F401, I001
    import os
    import sys
    from datetime import datetime

    from unsloth import FastLanguageModel, PatchFastRL

    from sarashina_grpo.config import PROJECT_ROOT

    # Add the `src` directory to the Python module search path
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

    ###################
    # Config
    ###################

    # fmt: off
    RESUME_FROM_CHECKPOINT = False  # Set to True to resume training. Default False
    MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"  # https://huggingface.co/sbintuitions
    MAX_STEPS = 500  # Number of training steps. Note that if you set RESUME_FROM_CHECKPOINT to True, this value needs to be larger than the number of steps already trained.
    LORA_RANK = 32  # Larger rank = smarter, but slower

    # MAX_SEQ_LENGTH ≥ MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH
    MAX_SEQ_LENGTH = 4096  # Can increase for longer reasoning traces
    MAX_PROMPT_LENGTH = 1024  # default 512 - Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
    MAX_COMPLETION_LENGTH = 512  # default 256 -  Maximum length of the generated completion.

    USE_VLLM = False  # True for vLLM fast inference, False for longer inference

    TRAINER_OUTPUTS_DIR = f"{PROJECT_ROOT}/artifact/outputs"  # Directory to save the model predictions and checkpoints
    TRAINER_LOGS_DIR = os.path.join(PROJECT_ROOT, "artifact/logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  # Directory to save the training logs for TensorBoard
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
    return (
        FastLanguageModel,
        LORA_RANK,
        MAX_COMPLETION_LENGTH,
        MAX_PROMPT_LENGTH,
        MAX_STEPS,
        RESUME_FROM_CHECKPOINT,
        TRAINER_LOGS_DIR,
        TRAINER_OUTPUTS_DIR,
        USE_VLLM,
        model,
        tokenizer,
    )


@app.cell
def _(
    FastLanguageModel,
    LORA_RANK,
    MAX_COMPLETION_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_STEPS,
    TRAINER_LOGS_DIR,
    TRAINER_OUTPUTS_DIR,
    USE_VLLM,
    model,
):
    ###################
    # Training
    ###################

    from trl import GRPOConfig, GRPOTrainer
    from unsloth import is_bfloat16_supported

    model_1 = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    # Apply LoRA (Low-Rank Adaptation) for Parameter-Efficient Fine-Tuning (PEFT)
    # https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
    training_args = GRPOConfig(
        use_vllm=USE_VLLM,
        learning_rate=5e-06,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="tensorboard",
        output_dir=TRAINER_OUTPUTS_DIR,
        logging_dir=TRAINER_LOGS_DIR,
    )
    return GRPOTrainer, model_1, training_args


@app.cell
def _(GRPOTrainer, RESUME_FROM_CHECKPOINT, model_1, tokenizer, training_args):
    from sarashina_grpo.dataset_loader import get_dataset
    from sarashina_grpo.grpo.xml_tuning import XMLTuning
    from sarashina_grpo.print import PrintCallback, PrintTuning

    dataset = get_dataset()

    xml_tuning = XMLTuning(tokenizer)
    reward_funcs = xml_tuning.get_reward_functions()
    print_tuning = PrintTuning(tokenizer)
    reward_funcs.insert(0, print_tuning.print_reward_func)

    # Add print class & function to show the debug information
    trainer = GRPOTrainer(
        model=model_1,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[PrintCallback()],
    )
    # NOTE: The training loss is not reliable for GRPO. See: https://github.com/huggingface/trl/issues/2703
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    return


if __name__ == "__main__":
    app.run()
