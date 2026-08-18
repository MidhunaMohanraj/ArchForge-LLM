import json
import streamlit as st
from data import (
    TASK_TYPES, VRAM_TIERS, get_datasets, recommend_models,
    FINE_TUNE_METHODS, PIPELINES, NEXT_STEPS,
)
 
st.set_page_config(page_title="ArchForge — Build a Custom LLM", page_icon="🛠️", layout="wide")

# ---------------------------------------------------------------------------
# Global state 
# ---------------------------------------------------------------------------
DEFAULTS = {
    "step": 1,
    "dataset": None,  
    "task_type": None,
    "deployment": None,
    "vram_choice": "Not sure — recommend for me",
    "style_slider": 50,
    "priority": "Balanced",
    "selected_model": None,   
    "ft_method": "lora",
    "epochs": 3,
    "batch_size": 4,
    "lr_exp": -4,
    "val_split": 15,
    "pipeline": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEP_NAMES = ["Dataset", "Parameters", "Model", "Fine-tune", "Summary"]


def goto(step):
    st.session_state.step = step


def progress_bar():
    cols = st.columns(len(STEP_NAMES))
    for i, (col, name) in enumerate(zip(cols, STEP_NAMES), start=1):
        with col:
            if i < st.session_state.step:
                st.markdown(f"✅ **{i}. {name}**")
            elif i == st.session_state.step:
                st.markdown(f"🟦 **{i}. {name}**")
            else:
                st.markdown(f"⬜ {i}. {name}")
    st.progress((st.session_state.step - 1) / (len(STEP_NAMES) - 1))
    st.divider()


st.title("🛠️ ArchForge — Custom LLM Builder")
st.caption("Prototype wizard · GUI-based, no chat interface · dev-team internal tool")
progress_bar()

# ---------------------------------------------------------------------------
# STEP 1 — Dataset selection
# ---------------------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("Step 1 · Choose a training dataset")
    st.write("Pick what you want the model to get good at, then browse matching datasets.")

    task_labels = {label: key for key, label in TASK_TYPES}
    chosen_label = st.radio(
        "What should the model be good at?",
        list(task_labels.keys()),
        index=0 if not st.session_state.task_type else
        [k for k, v in TASK_TYPES].index(st.session_state.task_type),
        horizontal=False,
    )
    st.session_state.task_type = task_labels[chosen_label]

    c1, c2 = st.columns([1, 2])
    with c1:
        source = st.selectbox("Source", ["All", "Hugging Face", "Ollama"])
    with c2:
        query = st.text_input("Search datasets (optional)", placeholder="e.g. medical, chat, code")

    results = get_datasets(task_type=st.session_state.task_type, source=source, query=query)

    st.write(f"**{len(results)} dataset(s) found**")
    for d in results:
        with st.container(border=True):
            top = st.columns([4, 1, 1])
            top[0].markdown(f"### {d['id']}")
            top[1].markdown(f"`{d['source']}`")
            selected = st.session_state.dataset and st.session_state.dataset["id"] == d["id"]
            if top[2].button("✅ Selected" if selected else "Select", key=f"ds_{d['id']}"):
                st.session_state.dataset = d
                st.rerun()
            st.caption(d["plain"])
            st.write(d["desc"])
            meta = st.columns(4)
            meta[0].metric("Rows", d["rows"])
            meta[1].metric("Size", d["size"])
            meta[2].write(f"**License**\n\n{d['license']}")
            meta[3].write("**Tags**\n\n" + ", ".join(d["tags"]))

    st.divider()
    disabled = st.session_state.dataset is None
    if disabled:
        st.info("Select a dataset above to continue.")
    if st.button("Next → Parameters", type="primary", disabled=disabled):
        goto(2)
        st.rerun()

# ---------------------------------------------------------------------------
# STEP 2 — Non-technical + technical parameter questions
# ---------------------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("Step 2 · Tell us about your use case")
    st.caption(f"Dataset selected: **{st.session_state.dataset['id']}** "
               f"({st.session_state.dataset['source']})")

    st.markdown("#### For everyone")
    st.session_state.deployment = st.radio(
        "Where will this model run?",
        ["Local / offline", "Cloud API", "Edge device (phone, embedded)"],
        horizontal=True,
    )

    st.session_state.style_slider = st.slider(
        "Response style", 0, 100, st.session_state.style_slider,
        help="0 = concise answers, 100 = detailed answers")
    st.caption("Concise " + "◀" * 3 + " " * 20 + "▶" * 3 + " Detailed")

    st.session_state.priority = st.select_slider(
        "What matters most?",
        options=["Speed", "Balanced", "Accuracy"],
        value=st.session_state.priority,
    )

    st.markdown("#### Hardware")
    vram_options = ["Not sure — recommend for me"] + [f"{v} GB" for v in VRAM_TIERS] + ["24+ GB"]
    st.session_state.vram_choice = st.select_slider(
        "Available VRAM", options=vram_options, value=st.session_state.vram_choice)

    with st.expander("🔧 Advanced (technical users)"):
        st.checkbox("I plan to quantize the model for inference", value=True, key="adv_quantize")
        st.checkbox("Multi-GPU training available", value=False, key="adv_multigpu")
        st.number_input("Max context length needed (tokens)", min_value=512,
                         max_value=131072, value=4096, step=512, key="adv_ctx")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        goto(1); st.rerun()
    if col2.button("Next → Model suggestions", type="primary"):
        goto(3); st.rerun()

# ---------------------------------------------------------------------------
# STEP 3 — Model recommendations
# ---------------------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("Step 3 · Recommended models")
    st.caption("Based on your dataset's task type and hardware, ranked lowest → highest VRAM usage.")

    vram_map = {"Not sure — recommend for me": None}
    for v in VRAM_TIERS:
        vram_map[f"{v} GB"] = v
    vram_map["24+ GB"] = 24
    vram_tier = vram_map[st.session_state.vram_choice]

    recs = recommend_models(st.session_state.task_type, vram_tier)

    cols = st.columns(3)
    for col, m in zip(cols, recs):
        with col:
            with st.container(border=True):
                st.markdown(f"### {m['name']}")
                st.markdown(f"`{m['source']}` · {m['params']} · {m['quant']}")
                st.metric("Est. VRAM", f"{m['vram']} GB")
                st.write(m["why"])
                selected = st.session_state.selected_model and st.session_state.selected_model["name"] == m["name"]
                if st.button("✅ Selected" if selected else "Choose this model", key=f"m_{m['name']}"):
                    st.session_state.selected_model = m
                    st.rerun()

    if st.session_state.selected_model:
        st.success(f"Selected: **{st.session_state.selected_model['name']}** "
                   f"({st.session_state.selected_model['source']})")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        goto(2); st.rerun()
    if col2.button("Next → Fine-tuning setup", type="primary",
                   disabled=st.session_state.selected_model is None):
        goto(4); st.rerun()

# ---------------------------------------------------------------------------
# STEP 4 — Fine-tuning parameters & pipeline
# ---------------------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("Step 4 · Fine-tuning setup")
    st.caption(f"Model: **{st.session_state.selected_model['name']}**  ·  "
               f"Dataset: **{st.session_state.dataset['id']}**")

    st.markdown("#### Method")
    method_labels = {m["label"]: m["id"] for m in FINE_TUNE_METHODS}
    default_label = [m["label"] for m in FINE_TUNE_METHODS if m["id"] == st.session_state.ft_method][0]
    chosen = st.radio("Fine-tuning method", list(method_labels.keys()),
                       index=list(method_labels.keys()).index(default_label), horizontal=True)
    st.session_state.ft_method = method_labels[chosen]
    st.caption(next(m["blurb"] for m in FINE_TUNE_METHODS if m["id"] == st.session_state.ft_method))

    st.markdown("#### Pipeline / tooling")
    pipeline_options = PIPELINES.get(st.session_state.selected_model["source"], [])
    st.session_state.pipeline = st.selectbox("Recommended pipeline", pipeline_options)

    st.markdown("#### Training parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.epochs = st.slider("Epochs", 1, 10, st.session_state.epochs)
        st.session_state.batch_size = st.slider("Batch size", 1, 32, st.session_state.batch_size)
        if st.session_state.selected_model["vram"] <= 8 and st.session_state.batch_size > 8:
            st.warning("Batch size may be too high for the selected model's VRAM tier.")
    with c2:
        st.session_state.lr_exp = st.slider(
            "Learning rate (log scale)", -5, -3, st.session_state.lr_exp,
            help="Slider represents 10^x")
        lr_value = 10 ** st.session_state.lr_exp
        st.caption(f"≈ {lr_value:.0e}  (recommended: 1e-4 for LoRA/QLoRA, 2e-5 for full fine-tune)")
        st.session_state.val_split = st.slider("Validation split (%)", 5, 30, st.session_state.val_split)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        goto(3); st.rerun()
    if col2.button("Next → Summary", type="primary"):
        goto(5); st.rerun()

# ---------------------------------------------------------------------------
# STEP 5 — Summary, export, and "what's next"
# ---------------------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("Step 5 · Review & export")

    config = {
        "dataset": {
            "id": st.session_state.dataset["id"],
            "source": st.session_state.dataset["source"],
            "task_type": st.session_state.task_type,
        },
        "use_case": {
            "deployment": st.session_state.deployment,
            "response_style_0_concise_100_detailed": st.session_state.style_slider,
            "priority": st.session_state.priority,
            "vram_choice": st.session_state.vram_choice,
        },
        "model": st.session_state.selected_model,
        "fine_tuning": {
            "method": st.session_state.ft_method,
            "pipeline": st.session_state.pipeline,
            "epochs": st.session_state.epochs,
            "batch_size": st.session_state.batch_size,
            "learning_rate": 10 ** st.session_state.lr_exp,
            "validation_split_pct": st.session_state.val_split,
        },
    }

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Configuration summary")
        st.json(config)
        st.download_button("⬇️ Download config (JSON)", data=json.dumps(config, indent=2),
                            file_name="archforge_config.json", mime="application/json",
                            type="primary")

    with c2:
        st.markdown("#### What to consider next")
        for step_info in NEXT_STEPS:
            with st.expander(step_info["title"]):
                st.write(step_info["body"])

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        goto(4); st.rerun()
    if col2.button("🔄 Start over"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
