"""
Mock data layer for the ArchForge LLM-builder wizard prototype.
In production, DATASETS would come from the live HF Hub / Ollama library
APIs, and MODEL_CATALOG would be a maintained, benchmarked lookup table.
"""

# ---------------------------------------------------------------------------
# Task types used across the whole wizard to tie dataset -> params -> models
# ---------------------------------------------------------------------------
TASK_TYPES = [
    ("code", "Code generation / dev assistant"),
    ("chat", "General chat & reasoning"),
    ("domain_qa", "Domain Q&A (legal, medical, finance, etc.)"),
    ("summarization", "Summarization"),
    ("instruction", "Custom instruction-following"),
]

VRAM_TIERS = [4, 8, 16, 24]  # GB, "24" really means 24+

# ---------------------------------------------------------------------------
# Datasets (mocked, tagged by task_type + source)
# ---------------------------------------------------------------------------
DATASETS = [
    # ---- code ----
    dict(id="bigcode/the-stack-smol", source="Hugging Face", task_type="code",
         rows="~500K", size="2.1 GB", license="OpenRail-M (commercial ok)",
         tags=["code", "multi-language"],
         desc="A distilled slice of The Stack — real permissively-licensed source code across 30+ languages.",
         plain="Teaches the model to read and write real code in many languages."),
    dict(id="sahil2801/CodeAlpaca-20k", source="Hugging Face", task_type="code",
         rows="20K", size="12 MB", license="Apache 2.0",
         tags=["code", "instruction"],
         desc="Instruction/response pairs for code generation, in the Alpaca format.",
         plain="Teaches the model to follow coding instructions like 'write a function that...'."),
    dict(id="codellama-finetune-pack", source="Ollama", task_type="code",
         rows="8K", size="9 MB", license="Community",
         tags=["code", "chat-style"],
         desc="Curated coding Q&A pairs packaged as an Ollama Modelfile training set.",
         plain="A ready-made coding-conversation pack tuned for local Ollama models."),

    # ---- chat / reasoning ----
    dict(id="HuggingFaceH4/ultrachat_200k", source="Hugging Face", task_type="chat",
         rows="200K", size="1.4 GB", license="MIT",
         tags=["chat", "multi-turn"],
         desc="High-quality multi-turn synthetic conversations covering everyday topics.",
         plain="Teaches the model to hold natural back-and-forth conversations."),
    dict(id="Open-Orca/OpenOrca", source="Hugging Face", task_type="chat",
         rows="4.2M", size="3.9 GB", license="MIT",
         tags=["chat", "reasoning"],
         desc="Augmented FLAN reasoning traces for stronger step-by-step answers.",
         plain="Teaches the model to reason through problems step by step."),
    dict(id="general-chat-pack", source="Ollama", task_type="chat",
         rows="15K", size="18 MB", license="Community",
         tags=["chat", "lightweight"],
         desc="Lightweight conversational pack for quick local fine-tunes.",
         plain="A small, fast conversation set for quick local experiments."),

    # ---- domain qa ----
    dict(id="medalpaca/medical_meadow_medqa", source="Hugging Face", task_type="domain_qa",
         rows="10K", size="8 MB", license="CC-BY 4.0",
         tags=["medical", "qa"],
         desc="Medical exam-style question/answer pairs for clinical reasoning tasks.",
         plain="Teaches the model to answer medical exam-style questions."),
    dict(id="pile-of-law/pile-of-law", source="Hugging Face", task_type="domain_qa",
         rows="~256K docs", size="17 GB", license="Mixed (check subsets)",
         tags=["legal", "documents"],
         desc="Large collection of legal documents, filings, and case law.",
         plain="Teaches the model legal language and case reasoning."),
    dict(id="finance-qa-pack", source="Ollama", task_type="domain_qa",
         rows="6K", size="5 MB", license="Community",
         tags=["finance", "qa"],
         desc="Finance-focused Q&A pack for local domain fine-tunes.",
         plain="Teaches the model finance terminology and common questions."),

    # ---- summarization ----
    dict(id="cnn_dailymail", source="Hugging Face", task_type="summarization",
         rows="300K", size="1.3 GB", license="Apache 2.0",
         tags=["summarization", "news"],
         desc="News articles paired with human-written highlight summaries.",
         plain="Teaches the model to condense long articles into short summaries."),
    dict(id="samsum", source="Hugging Face", task_type="summarization",
         rows="16K", size="10 MB", license="CC-BY-NC 4.0",
         tags=["summarization", "dialogue"],
         desc="Chat-log style dialogues paired with short summaries.",
         plain="Teaches the model to summarize conversations and chat logs."),

    # ---- instruction ----
    dict(id="tatsu-lab/alpaca", source="Hugging Face", task_type="instruction",
         rows="52K", size="24 MB", license="CC-BY-NC 4.0",
         tags=["instruction", "general"],
         desc="The original Alpaca instruction-following dataset.",
         plain="Teaches the model to follow general instructions well."),
    dict(id="custom-instruct-pack", source="Ollama", task_type="instruction",
         rows="12K", size="14 MB", license="Community",
         tags=["instruction", "lightweight"],
         desc="Compact instruction pack tuned for fast local iteration.",
         plain="A small instruction-following set for quick local fine-tunes."),
]


def get_datasets(task_type=None, source=None, query=None):
    results = DATASETS
    if task_type:
        results = [d for d in results if d["task_type"] == task_type]
    if source and source != "All":
        results = [d for d in results if d["source"] == source]
    if query:
        q = query.lower()
        results = [d for d in results if q in d["id"].lower() or q in d["desc"].lower()
                   or any(q in t for t in d["tags"])]
    return results


# ---------------------------------------------------------------------------
# Model catalog (mocked) keyed by task_type, each entry has an est. VRAM
# ---------------------------------------------------------------------------
MODEL_CATALOG = {
    "code": [
        dict(name="Phi-3-mini (3.8B, Q4)", source="Ollama", params="3.8B",
             quant="Q4_K_M", vram=4, why="Lightweight coding help, runs on modest laptops."),
        dict(name="CodeLlama-7B-Instruct", source="Ollama", params="7B",
             quant="Q4_K_M", vram=8, why="Great balance of coding accuracy and local speed."),
        dict(name="StarCoder2-7B", source="Hugging Face", params="7B",
             quant="fp16", vram=16, why="Strong multi-language code completion, full precision."),
        dict(name="DeepSeek-Coder-33B-Instruct", source="Hugging Face", params="33B",
             quant="Q4_K_M", vram=24, why="Top-tier coding accuracy for serious dev workloads."),
    ],
    "chat": [
        dict(name="TinyLlama-1.1B-Chat", source="Ollama", params="1.1B",
             quant="Q4_K_M", vram=4, why="Fast, tiny footprint for basic conversational needs."),
        dict(name="Llama-3-8B-Instruct", source="Ollama", params="8B",
             quant="Q4_K_M", vram=8, why="Well-rounded general chat quality on consumer GPUs."),
        dict(name="Mistral-7B-Instruct-v0.3", source="Hugging Face", params="7B",
             quant="fp16", vram=16, why="Crisp reasoning and instruction-following at full precision."),
        dict(name="Mixtral-8x7B-Instruct", source="Hugging Face", params="47B (MoE)",
             quant="Q4_K_M", vram=24, why="Near frontier-level chat quality, MoE efficiency."),
    ],
    "domain_qa": [
        dict(name="BioMistral-7B (Q4)", source="Ollama", params="7B",
             quant="Q4_K_M", vram=8, why="Medical-tuned base, cheap to fine-tune further."),
        dict(name="Meditron-7B", source="Hugging Face", params="7B",
             quant="fp16", vram=16, why="Purpose-built medical reasoning at full precision."),
        dict(name="Law-LLM-13B", source="Hugging Face", params="13B",
             quant="fp16", vram=24, why="Deeper legal/finance context handling for complex docs."),
    ],
    "summarization": [
        dict(name="Phi-3-mini (3.8B, Q4)", source="Ollama", params="3.8B",
             quant="Q4_K_M", vram=4, why="Handles short-doc summarization cheaply."),
        dict(name="Llama-3-8B-Instruct", source="Ollama", params="8B",
             quant="Q4_K_M", vram=8, why="Reliable general-purpose summarizer."),
        dict(name="BART-Large-CNN", source="Hugging Face", params="406M",
             quant="fp16", vram=4, why="Purpose-built summarization encoder-decoder, very light."),
        dict(name="Mistral-7B-Instruct-v0.3", source="Hugging Face", params="7B",
             quant="fp16", vram=16, why="Higher-quality abstractive summaries, more context."),
    ],
    "instruction": [
        dict(name="Gemma-2B-Instruct", source="Ollama", params="2B",
             quant="Q4_K_M", vram=4, why="Small, fast instruction-follower for quick tests."),
        dict(name="Llama-3-8B-Instruct", source="Ollama", params="8B",
             quant="Q4_K_M", vram=8, why="Dependable general instruction-following."),
        dict(name="Mistral-7B-Instruct-v0.3", source="Hugging Face", params="7B",
             quant="fp16", vram=16, why="Higher fidelity instruction-following at full precision."),
        dict(name="Mixtral-8x7B-Instruct", source="Hugging Face", params="47B (MoE)",
             quant="Q4_K_M", vram=24, why="Best-in-class instruction quality if hardware allows."),
    ],
}


def recommend_models(task_type, vram_tier):
    """
    Return up to 3 models for a task type, sorted low->high VRAM.
    If vram_tier is given, prefer models that fit within it; otherwise
    just return the 3 lowest-VRAM options as a safe default.
    """
    pool = sorted(MODEL_CATALOG.get(task_type, []), key=lambda m: m["vram"])
    if vram_tier:
        fitting = [m for m in pool if m["vram"] <= vram_tier]
        if len(fitting) >= 3:
            return fitting[-3:] if len(fitting) > 3 else fitting
        # not enough fitting models -> pad with next cheapest above tier
        rest = [m for m in pool if m not in fitting]
        return (fitting + rest)[:3]
    return pool[:3]


FINE_TUNE_METHODS = [
    dict(id="full", label="Full fine-tune",
         blurb="Updates every model weight. Best quality, needs the most VRAM and data."),
    dict(id="lora", label="LoRA",
         blurb="Trains small adapter layers. Great quality/cost trade-off, widely used."),
    dict(id="qlora", label="QLoRA",
         blurb="LoRA on a quantized model. Lowest VRAM footprint, ideal for local GPUs."),
    dict(id="prompt", label="Prompt-tuning",
         blurb="Learns soft prompts only. Cheapest and fastest, more limited gains."),
]

PIPELINES = {
    "Hugging Face": ["transformers + peft", "Axolotl", "LLaMA-Factory"],
    "Ollama": ["Ollama Modelfile customization", "Axolotl (export to GGUF)", "LLaMA-Factory (export to GGUF)"],
}

NEXT_STEPS = [
    dict(title="Data preparation",
         body="Clean and deduplicate the dataset, verify tokenizer compatibility, "
              "and set aside a train / validation / test split before training starts."),
    dict(title="Compute & cost",
         body="Decide between renting cloud GPUs or running locally, and get a rough "
              "estimate of training time and cost for the chosen VRAM tier."),
    dict(title="Evaluation plan",
         body="Pick how you'll measure quality: perplexity, task-specific benchmarks, "
              "or structured human evaluation before calling the fine-tune 'done'."),
    dict(title="Licensing check",
         body="Confirm the base model and dataset licenses allow your intended use, "
              "especially for commercial or redistributed products."),
    dict(title="Safety review",
         body="Run a bias/toxicity check and add guardrails or a system prompt policy "
              "before the model is exposed to real users."),
    dict(title="Deployment format",
         body="Choose a serving path: quantize for local inference (Ollama/GGUF) or "
              "serve via vLLM / Text Generation Inference for higher-throughput APIs."),
    dict(title="Monitoring & feedback",
         body="Plan for drift detection and a feedback loop so future fine-tunes can "
              "correct issues found after deployment."),
]
