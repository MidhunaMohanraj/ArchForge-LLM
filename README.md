# ArchForge — Custom LLM Builder (Prototype)

A GUI-only (no chatbot) wizard for configuring a custom LLM fine-tune, built
with Streamlit. All data is mocked in `data.py` so the flow can be reviewed
end-to-end before wiring up live Hugging Face Hub / Ollama APIs.

## Run it

```bash
pip install streamlit
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Flow

1. **Dataset** — pick a use case (code / chat / domain QA / summarization /
   instruction), then browse and select a Hugging Face or Ollama dataset.
2. **Parameters** — deployment target, response style, speed/accuracy
   priority, and available VRAM, all via sliders/radios (no free text
   required). An "Advanced" expander holds technical extras.
3. **Model** — 3 recommended models for the chosen task + hardware, sorted
   low → high VRAM, pulled from `MODEL_CATALOG` in `data.py`.
4. **Fine-tune** — method (Full / LoRA / QLoRA / Prompt-tuning), pipeline
   (auto-filtered by the model's source), epochs, batch size, learning rate,
   validation split — all slider/dropdown controlled.
5. **Summary** — read-only recap, downloadable JSON config, and a
   "What to consider next" panel (data prep, compute/cost, evaluation,
   licensing, safety, deployment format, monitoring).

## Swapping in real data

- Replace `get_datasets()` in `data.py` with live calls to the HF Hub API
  (`huggingface_hub.list_datasets`) and the Ollama library API.
- Replace `MODEL_CATALOG` / `recommend_models()` with a maintained,
  benchmarked table (or a small scoring model) once real VRAM/quality
  numbers are available.
- The exported JSON in Step 5 is shaped to be consumable directly by a
  training kickoff script.
