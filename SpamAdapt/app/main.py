"""
SpamAdapt: Interactive Web Application (Stage 6).

Streamlit dashboard for SMS Spam Detection, Model Benchmarking,
and Parameter-Efficient Fine-Tuning Architectural Walkthrough.
"""

import datetime
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

import app.inference
import app.model_registry

# Diagnostic output on startup / module import
print("[DIAGNOSTICS] ==================================================")
print("[DIAGNOSTICS] SpamAdapt Server Startup / Rerun")
print(f"[DIAGNOSTICS] Working Directory : {os.getcwd()}")
print(f"[DIAGNOSTICS] Main Script       : {Path(__file__).resolve()}")
print(f"[DIAGNOSTICS] Inference Module  : {Path(app.inference.__file__).resolve()}")
print(f"[DIAGNOSTICS] Model Registry    : {Path(app.model_registry.__file__).resolve()}")
print(f"[DIAGNOSTICS] Python Executable : {sys.executable} (PID: {os.getpid()})")
print("[DIAGNOSTICS] ==================================================")

# Must be the very first Streamlit command
st.set_page_config(
    page_title="SpamAdapt — Spam Detection & PEFT Benchmarks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.styles import DARK_THEME_CSS
from app.inference import (
    SAMPLE_MESSAGES,
    get_inference_device,
    load_cached_model,
    predict_message,
)
from app.metrics_loader import (
    get_comparison_summary_table,
    load_all_metrics,
)
from app.model_registry import (
    get_models_for_version,
    get_model_spec,
    get_ordered_selectable_models,
    get_active_ui_models,
    validate_model_files,
    REGISTERED_MODELS,
)


def init_session_state():
    """Initializes session-only prediction history and UI state."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "sms_input" not in st.session_state:
        st.session_state["sms_input"] = ""


def render_header(device_str: str, cuda_available: bool):
    """Renders dashboard header with title and status badges."""
    cuda_status = "CUDA Active" if cuda_available else "CPU Execution"
    badge_color_class = "tech-badge-primary" if cuda_available else "tech-badge"

    st.markdown(
        f"""
        <div class="main-header">
            <h1 class="main-title">🛡️ SpamAdapt Research Dashboard</h1>
            <p class="sub-title">
                Parameter-Efficient vs. Full Fine-Tuning for Message Spam Detection on Pretrained BERT
            </p>
            <div class="badge-container">
                <span class="tech-badge tech-badge-primary">google-bert/bert-base-uncased</span>
                <span class="tech-badge tech-badge-purple">Hugging Face PEFT 0.21.2</span>
                <span class="tech-badge {badge_color_class}">Device: {device_str.upper()} ({cuda_status})</span>
                <span class="tech-badge">Dataset: UCI SMS Spam Collection</span>
                <span class="tech-badge">Sequence Length: 128</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Inject custom styling
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    init_session_state()

    device = get_inference_device()
    cuda_avail = device.type == "cuda"
    project_root_str = str(project_root)

    render_header(str(device), cuda_avail)

    # Sidebar: Global Model Configuration (Single Unified Model Dropdown)
    st.sidebar.markdown("### ⚙️ Model Selection")

    # ONE central dropdown driven by registered models: exactly 3 active models
    selectable_specs = get_active_ui_models()
    valid_specs = []
    validation_failures = {}

    for s in selectable_specs:
        try:
            validate_model_files(s.model_id, project_root)
            valid_specs.append(s)
        except Exception as e:
            validation_failures[s.model_id] = str(e)
            print(f"[DIAGNOSTICS] Model '{s.model_id}' failed validation: {e}")

    # Build options mapping: display_name -> model_id
    model_options = {s.display_name: s.model_id for s in valid_specs}

    # Verify LoRA v3 status: set as initial default ONLY if valid, else explicit error without fallback
    lora_v3_valid = any(s.model_id == "lora_v3" for s in valid_specs)
    if not lora_v3_valid:
        st.sidebar.error(
            f"❌ LoRA v3 Candidate Failed Validation: {validation_failures.get('lora_v3', 'Missing files')}. "
            "Cannot set as default model."
        )

    # Find default index for selectbox (LoRA v3 if available)
    default_idx = 0
    if lora_v3_valid:
        default_idx = [s.model_id for s in valid_specs].index("lora_v3")

    selected_display_label = st.sidebar.selectbox(
        "Active Model / Strategy:",
        options=list(model_options.keys()),
        index=default_idx,
        help="Select one of the three active models (LoRA v3, DoRA v2, or Full Fine-Tuning v2) for real-time sequence classification.",
    )
    selected_method = model_options[selected_display_label]
    spec = get_model_spec(selected_method)

    # Reference get_models_for_version to maintain invariant compatibility
    _v3_models = get_models_for_version("v3")

    # Model info card in sidebar directly from verified registry metadata
    st.sidebar.markdown(
        f"""
        <div class="dash-card">
            <div class="card-title">{spec.display_name}</div>
            <p style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 8px;">{spec.description}</p>
            <div style="font-size: 0.75rem; color: #94a3b8;">
                <b>Identifier:</b> <code>{spec.model_id}</code><br>
                <b>Generation:</b> {spec.version.upper()}<br>
                <b>Checkpoint Path:</b> <code>{spec.checkpoint_subpath}</code><br>
                <b>Adapter / Weights Size:</b> {spec.adapter_size_mb:.2f} MB<br>
                <b>Trainable Params:</b> {spec.trainable_parameters:,} ({spec.trainable_parameters/spec.total_parameters*100:.2f}%)<br>
                <b>Decision &tau;:</b> {spec.default_threshold:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("🔧 Runtime Diagnostics", expanded=False):
        st.caption(f"**CWD:** `{os.getcwd()}`")
        st.caption(f"**Main:** `{Path(__file__).resolve()}`")
        st.caption(f"**Inference:** `{Path(app.inference.__file__).resolve()}`")
        st.caption(f"**Registry:** `{Path(app.model_registry.__file__).resolve()}`")
        st.caption(f"**PID:** `{os.getpid()}`")
        st.caption(f"**Active Model:** `{spec.model_id}` ({spec.display_name})")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Empirical Guarantee")
    st.sidebar.info(
        "All models use fresh `google-bert/bert-base-uncased` base weights and were evaluated on the held-out "
        "test split ($N=774$). Models are cached in local memory."
    )

    # Main Navigation Tabs
    tab_detect, tab_benchmarks, tab_arch = st.tabs([
        "🔍 Live SMS Detection",
        "📊 Model Comparison & Benchmarks",
        "🧠 Architecture & Methodology",
    ])

    # -------------------------------------------------------------
    # TAB 1: LIVE SMS DETECTION
    # -------------------------------------------------------------
    with tab_detect:
        col_input, col_info = st.columns([2, 1])

        with col_input:
            st.markdown("#### Input Text Message")

            # Sample message quick selector
            sample_key = st.selectbox(
                "Or choose a curated test example:",
                list(SAMPLE_MESSAGES.keys()),
                index=0,
            )
            if sample_key != "Select a pre-configured sample...":
                st.session_state["sms_input"] = SAMPLE_MESSAGES[sample_key]

            # Text area for user input
            sms_text = st.text_area(
                "Enter or paste the SMS text below:",
                value=st.session_state.get("sms_input", ""),
                height=130,
                placeholder="e.g. URGENT! You have won a 1000 cash prize. Call 09066364589 now to claim...",
            )

            col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
            with col_btn1:
                analyze_clicked = st.button("🚀 Analyze Message", use_container_width=True)
            with col_btn2:
                clear_clicked = st.button("🗑️ Clear Input", use_container_width=True)

            if clear_clicked:
                st.session_state["sms_input"] = ""
                st.rerun()

        with col_info:
            st.markdown("#### Detection Guidelines")
            st.markdown(
                """
                - **Character Limit**: Input up to 2,000 characters (tokenized to max length 128).
                - **Preserved Signals**: URLs, phone numbers, punctuation, and uppercase text are critical spam signals.
                - **Latency Profile**: Warm model inference takes ~115–150 ms on CPU (~1.6 s initial cold load). GPU acceleration available when CUDA is present.
                """
            )

        # Execution and Result Rendering
        if analyze_clicked:
            if not sms_text or not sms_text.strip():
                st.error("⚠️ Validation Error: Message text cannot be empty. Please enter an SMS message.")
            else:
                with st.spinner(f"Running inference with {selected_display_label}..."):
                    try:
                        result = predict_message(
                            text=sms_text,
                            method=selected_method,
                            project_root_str=project_root_str,
                        )

                        loaded_name = result.get("loaded_model_type", spec.display_name)
                        loaded_id = result.get("loaded_model_id", spec.model_id)

                        # Record in session history with actual loaded model
                        history_entry = {
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "model": f"{loaded_name} ({loaded_id})",
                            "preview": (sms_text[:60] + "...") if len(sms_text) > 60 else sms_text,
                            "decision": result.get("decision", result["predicted_label"]),
                            "predicted": result["predicted_label"],
                            "confidence": f"{result['confidence']:.1f}%",
                            "latency": f"{result['latency_ms']:.1f} ms",
                        }
                        st.session_state["history"].insert(0, history_entry)

                        decision = result.get("decision", "SUSPECTED_SPAM" if result["is_spam"] else "HAM")
                        thresh_used = result.get("threshold_used", 0.50)
                        tau_low_str = f"&tau;_low = {result['tau_low']:.2f}" if result.get("tau_low") is not None else ""
                        tau_high_str = f"&tau;_high = {result['tau_high']:.2f}" if result.get("tau_high") is not None else ""
                        policy_meta = f"&nbsp;|&nbsp; Abstention Band: [{result['tau_low']:.2f}, {result['tau_high']:.2f})" if result.get("tau_low") is not None else ""

                        # Display prediction banner based on three-tier decision outcome
                        if decision == "SUSPECTED_SPAM":
                            st.markdown(
                                f"""
                                <div class="result-banner-spam">
                                    <div class="result-tag-spam">
                                        🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE
                                    </div>
                                    <p style="color: #fca5a5; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0;">
                                        Confidence: <b>{result['confidence']:.2f}%</b> &nbsp;|&nbsp; 
                                        Inference Latency: <b>{result['latency_ms']:.1f} ms</b> &nbsp;|&nbsp; 
                                        Loaded Model: <b>{loaded_name} (<code>{loaded_id}</code>)</b> &nbsp;|&nbsp;
                                        Decision Threshold: <b>&tau; = {thresh_used:.2f}</b>{policy_meta}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        elif decision == "REVIEW_REQUIRED":
                            st.markdown(
                                f"""
                                <div class="result-banner-review">
                                    <div class="result-tag-review">
                                        ⚠️ REVIEW REQUIRED / UNCERTAIN DECISION BOUNDARY
                                    </div>
                                    <p style="color: #fde68a; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0;">
                                        Model Output: <b>P(Spam) = {result['spam_probability']*100:.2f}%</b> &nbsp;|&nbsp; 
                                        Inference Latency: <b>{result['latency_ms']:.1f} ms</b> &nbsp;|&nbsp; 
                                        Loaded Model: <b>{loaded_name} (<code>{loaded_id}</code>)</b> &nbsp;|&nbsp;
                                        Decision Threshold: <b>&tau; = {thresh_used:.2f}</b>{policy_meta}
                                    </p>
                                    <p style="color: #fef08a; font-size: 0.85rem; margin-top: 4px; margin-bottom: 0;">
                                        Notice: Score falls within the uncertainty zone between &tau;_low and &tau;_high. Manual review or secondary verification is recommended.
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="result-banner-ham">
                                    <div class="result-tag-ham">
                                        ✅ HAM / LIKELY LEGITIMATE MESSAGE
                                    </div>
                                    <p style="color: #6ee7b7; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0;">
                                        Confidence: <b>{result['confidence']:.2f}%</b> &nbsp;|&nbsp; 
                                        Inference Latency: <b>{result['latency_ms']:.1f} ms</b> &nbsp;|&nbsp; 
                                        Loaded Model: <b>{loaded_name} (<code>{loaded_id}</code>)</b> &nbsp;|&nbsp;
                                        Decision Threshold: <b>&tau; = {thresh_used:.2f}</b>{policy_meta}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Softmax Probability Distribution Breakdown
                        st.markdown("##### Uncalibrated Model Scores (Raw Softmax Output)")
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.write(f"**Legitimate (Ham) Uncalibrated Score:** {result['ham_probability']*100:.2f}%")
                            st.progress(float(result['ham_probability']))
                        with col_p2:
                            st.write(f"**Spam Uncalibrated Score:** {result['spam_probability']*100:.2f}%")
                            st.progress(float(result['spam_probability']))

                        st.markdown(
                            """
                            <div class="security-notice">
                                ⚠️ <b>Research Transparency Notice</b>: Scores shown reflect raw uncalibrated softmax outputs on BERT representations and must NOT be treated as calibrated real-world fraud odds or proof of absolute safety/fraud. Always independently verify financial transactions, sensitive account requests, and unsolicited URL links.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except FileNotFoundError as fnf_err:
                        st.error(f"❌ Checkpoint Loading Failure: {str(fnf_err)}. Ensure the specified model checkpoint directory exists.")
                    except ValueError as val_err:
                        st.error(f"⚠️ Input Validation Error: {str(val_err)}")
                    except Exception as e:
                        st.error(f"❌ Model Inference Failure: {str(e)}")

        # Session Prediction History Section
        st.markdown("---")
        col_hist_title, col_hist_clear = st.columns([4, 1])
        with col_hist_title:
            st.markdown("#### 📜 Session Prediction History")
        with col_hist_clear:
            if st.button("Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        if st.session_state["history"]:
            history_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(
                history_df,
                column_config={
                    "timestamp": "Time",
                    "model": "Model Used",
                    "preview": "Message Preview",
                    "decision": "Outcome",
                    "predicted": "Class",
                    "confidence": "Confidence",
                    "latency": "Latency",
                },
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "🔒 Session privacy: Predictions are processed in local volatile RAM and are never written to disk or sent to external servers."
            )
        else:
            st.info("No messages analyzed yet in this session. Submit a message above to see history.")

    # -------------------------------------------------------------
    # TAB 2: MODEL COMPARISON & BENCHMARKS
    # -------------------------------------------------------------
    with tab_benchmarks:
        st.markdown("### 🔬 Empirical Benchmarking Comparison")
        st.markdown(
            "Comparative analysis of Full Fine-Tuning, LoRA, and DoRA evaluated on the held-out "
            "UCI SMS Spam test split ($N = 774$ examples) across predictive performance, storage, and runtime efficiency."
        )

        all_metrics = load_all_metrics(project_root_str)
        summary_rows = get_comparison_summary_table(all_metrics)

        # High-level KPI Cards
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-value" style="color: #60a5fa;">96.40%</div>
                    <div class="stat-label">Full FT Macro-F1</div>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 4px 0 0 0;">109.5M Trainable Params | 417.7 MB</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_kpi2:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-value" style="color: #34d399;">96.12%</div>
                    <div class="stat-label">LoRA Macro-F1 (-0.28%)</div>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 4px 0 0 0;">296.5k Params (0.27%) | 1.14 MB</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_kpi3:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-value" style="color: #c084fc;">95.84%</div>
                    <div class="stat-label">DoRA Macro-F1 (-0.56%)</div>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin: 4px 0 0 0;">314.9k Params (0.29%) | 1.21 MB</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Comprehensive Verification Table")
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        st.markdown("#### 📈 Benchmark Visualizations")
        col_plot1, col_plot2 = st.columns(2)

        plots_dir = project_root / "results" / "plots"
        lat_plot = plots_dir / "inference_latency_comparison.png"
        tp_plot = plots_dir / "throughput_comparison.png"
        size_plot = plots_dir / "model_size_comparison.png"
        param_plot = plots_dir / "parameter_efficiency_comparison.png"

        with col_plot1:
            if lat_plot.exists():
                st.image(str(lat_plot), caption="Per-Message Inference Latency Distribution (ms)")
            else:
                st.warning("Latency comparison plot not found.")

            if size_plot.exists():
                st.image(str(size_plot), caption="Model Storage & Deployed Footprint (MB)")
            else:
                st.warning("Model size comparison plot not found.")

        with col_plot2:
            if tp_plot.exists():
                st.image(str(tp_plot), caption="Batched Inference Throughput (msgs/sec)")
            else:
                st.warning("Throughput comparison plot not found.")

            if param_plot.exists():
                st.image(str(param_plot), caption="Trainable Parameter Footprint (Log Scale)")
            else:
                st.warning("Parameter efficiency plot not found.")

    # -------------------------------------------------------------
    # TAB 3: ARCHITECTURE & METHODOLOGY
    # -------------------------------------------------------------
    with tab_arch:
        st.markdown("### 🧠 Transformer Architecture & Adaptation Methodology")

        st.markdown(
            """
            #### 1. Pretrained BERT Sequence Classification
            **Architecture**: `google-bert/bert-base-uncased` is a bidirectional transformer encoder comprising:
            - **12 Transformer layers**, 12 attention heads, hidden dimension $d = 768$.
            - **109.5 Million total parameters**.
            - **Sequence Classification**: Unlike generative autoregressive language models (e.g. GPT), BERT processes 
              the entire input sequence simultaneously. The special classification token `[CLS]` aggregates cross-attention representations, 
              which are passed through a pooled dropout layer and projected into two logits ($0 = \\text{Ham}$, $1 = \\text{Spam}$).

            ---

            #### 2. Fine-Tuning Strategies Compared

            ##### A. Full Fine-Tuning Baseline
            - Updates all 109.5 million parameters across embeddings, self-attention, feed-forward layers, and classifier head.
            - **Trade-off**: Yields the highest Macro-F1 (**96.40%**), but requires saving full 418 MB checkpoints and high GPU/CPU gradient compute.

            ##### B. LoRA (Low-Rank Adaptation)
            - Freezes the entire base BERT transformer backbone.
            - Injects trainable rank decomposition matrices into the query and value attention projections:
              $$\\mathbf{W} = \\mathbf{W}_0 + \\frac{\\alpha}{r} \\mathbf{B} \\mathbf{A}$$
              where $\\mathbf{W}_0 \\in \\mathbb{R}^{d \\times k}$ is frozen, $\\mathbf{B} \\in \\mathbb{R}^{d \\times r}$ (initialized to zero), and $\\mathbf{A} \\in \\mathbb{R}^{r \\times k}$ (Gaussian).
            - **Parameters**: Only **296,450 parameters (0.27%)** are trained ($r=8, \\alpha=16$).
            - **Performance**: Retains **99.7% of Full Fine-Tuning performance (96.12% Macro-F1)** while reducing weights by **366x** (1.14 MB).
            - **Inference Speed**: Fastest inference (**92.05 ms per message**), outperforming Full Fine-Tuning due to cache-friendly localized weights.

            ##### C. DoRA (Weight-Decomposed Low-Rank Adaptation)
            - Decouples weight updates into directional updates and learnable magnitude vectors:
              $$\\mathbf{W} = \\mathbf{m} \\frac{\\mathbf{W}_0 + \\frac{\\alpha}{r}\\mathbf{B}\\mathbf{A}}{\\|\\mathbf{W}_0 + \\frac{\\alpha}{r}\\mathbf{B}\\mathbf{A}\\|_c}$$
              where $\\mathbf{m} \\in \\mathbb{R}^{1 \\times k}$ is a learnable magnitude vector matching projection dimension.
            - **Parameters**: **314,882 parameters (0.29%)** ($r=8, \\alpha=16$, `use_dora=True`).
            - **Findings**: Achieved slightly lower test loss (**0.0779 vs 0.0783**) and faster epoch convergence, but registered slightly higher per-message latency (**110.0 ms**) due to the runtime matrix normalization step.

            ---

            #### 3. Dataset & Scientific Split Verification
            - **UCI SMS Spam Collection**: 5,160 clean, deduplicated messages (4,518 Ham, 642 Spam).
            - **Stratified Split (`seed=42`)**:
              - **Train (70%)**: 3,612 samples (3,163 Ham, 449 Spam)
              - **Validation (15%)**: 774 samples (678 Ham, 96 Spam)
              - **Test (15%)**: 774 samples (677 Ham, 97 Spam)
            - **Zero Contamination**: The test set was held out and evaluated only once per method. Cryptographic SHA-256 hashes guarantee split integrity.

            ---

            #### 4. Limitations & Research Notes
            - **1-Epoch Setup**: Due to CPU hardware constraints, models were trained for 1 epoch. While sufficient for 96%+ Macro-F1 on BERT, multi-epoch convergence on DoRA may reveal additional gains.
            - **Single Dataset Scope**: Findings reflect SMS spam patterns and may exhibit domain shift on email, web, or social media spam.
            """
        )


if __name__ == "__main__":
    main()
