# SpamAdapt: Parameter-Efficient vs. Full Fine-Tuning for Message Spam Detection

**SpamAdapt** is an empirical machine learning research project that experimentally evaluates and compares three transformer adaptation paradigms for SMS and message spam classification:

1. **Full Fine-Tuning**: Updating all 110M parameters of the pretrained transformer.
2. **LoRA (Low-Rank Adaptation)**: Freezing base weights and injecting trainable low-rank decomposition matrices ($A$ and $B$).
3. **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Decomposing weights into directional and magnitude components, applying LoRA solely to the directional update.

The objective is to systematically benchmark **predictive performance** (Accuracy, Precision, Recall, F1, Macro-F1) against **computational efficiency** (trainable parameter ratio, memory footprint, training throughput, and latency).

---

## 📌 Experimental Results & Milestone Status

### Completed Stages
- **Stage 1: Dataset Pipeline & Preprocessing**: Automated retrieval, signal-preserving cleaning, deduplication, and stratified 70/15/15 splits with SHA-256 integrity checks.
- **Stage 2: Full Fine-Tuning Baseline**: 100% parameters trained (109.5M), achieving 96.40% Macro-F1 on test set.
- **Stage 3: LoRA Fine-Tuning (PEFT)**: Freezing backbone and training query/value low-rank adapters ($r=8, \alpha=16$) + classifier (296.5k parameters, 0.27%), achieving 96.12% Macro-F1.
- **Stage 4: DoRA Fine-Tuning (PEFT use_dora=True)**: Directional adapters + learnable magnitude vectors (314.9k parameters, 0.29%), achieving 95.84% Macro-F1 with lower test loss (0.0779).
- **Stage 5: Computational & Efficiency Benchmarking**: Systematic inference latency profiling, batched throughput benchmarking, memory tracking via `psutil`, storage footprint comparison, and 100% metric cross-verification across all three models on held-out test split ($N=774$).
- **Stage 6: Interactive Web Application & Verification**: Polished, multi-tab Streamlit web application featuring real-time SMS spam analysis across all three models, model comparison dashboards with verified benchmark plots, architecture & methodology deep dives, ephemeral session history, and a 7/7 automated testing suite.

### 🔬 Empirical Comparison Table (Held-Out Test Split, N=774)

| Metric / Attribute | Baseline (Full Fine-Tuning) | Stage 3: LoRA (PEFT) | Stage 4: DoRA (PEFT) | Trade-Off & Efficiency Insights |
| :--- | :--- | :--- | :--- | :--- |
| **Base Model** | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` | Identical base weights |
| **Total Parameters** | 109,483,778 | 109,780,228 | 109,798,660 | +18.4k DoRA magnitude params |
| **Trainable Parameters** | 109,483,778 (100.0%) | **296,450 (0.27%)** | 314,882 (0.29%) | **~350x-370x parameter reduction** |
| **Adapter Storage on Disk** | 417.67 MB (`model.safetensors`) | **1.14 MB (`adapter_model.safetensors`)** | **1.21 MB (`adapter_model.safetensors`)** | **345x-366x storage saving** |
| **Total Deployed Footprint** | 417.67 MB | 439.10 MB | 439.17 MB | Base model (437.9 MB) + adapter |
| **Training Duration** | 24m 24s (CPU) | 21m 43s (CPU) | **19m 35s (CPU)** | Fastest training convergence |
| **Model Loading Time** | **0.514 s** | 1.268 s | 1.254 s | Full FT avoids adapter injection |
| **Process Memory Delta (RSS)**| 419.43 MB | **417.82 MB** | 426.99 MB | Measured via psutil (CPU execution) |
| **Peak GPU Memory** | N/A (CPU execution) | N/A (CPU execution) | N/A (CPU execution) | Honest hardware reporting |
| **Mean Inference Latency** | 99.48 ms | **92.05 ms** | 110.00 ms | **LoRA is 7.5% faster than Full FT** |
| **Median Inference Latency** | 97.55 ms | **90.67 ms** | 107.86 ms | LoRA achieves lowest median latency |
| **P95 Inference Latency** | 120.36 ms | **102.08 ms** | 124.72 ms | LoRA maintains tightest latency tail |
| **Throughput (Batch Size=2)**| 12.71 msg/s | **12.74 msg/s** | 11.32 msg/s | Consistent batched throughput |
| **Test Accuracy** | **98.45%** (762/774) | 98.32% (761/774) | 98.19% (760/774) | Spread within 0.26% |
| **Spam Precision** | **95.70%** (89/93) | 94.68% (89/94) | 93.68% (89/95) | 1 extra FP per method |
| **Spam Recall** | **91.75% (89/97)** | **91.75% (89/97)** | **91.75% (89/97)** | **Identical recall (89/97 caught)** |
| **Spam F1-Score** | **93.68%** | 93.19% | 92.71% | High consistency across methods |
| **Macro F1-Score** | **96.40%** | 96.12% | 95.84% | All retain >99.4% of FT |
| **Test Loss** | **0.0701** | 0.0783 | 0.0779 | DoRA achieves lower loss than LoRA |
| **Confusion Matrix** | TN: 673, FP: 4, FN: 8, TP: 89 | TN: 672, FP: 5, FN: 8, TP: 89 | TN: 671, FP: 6, FN: 8, TP: 89 | Exactly 8 FN across all 3 |

### 💡 Accuracy vs. Efficiency Trade-off Insights
- **Inference Latency Advantage of LoRA**: LoRA demonstrated the lowest mean latency (**92.05 ms**) and tightest P95 tail (**102.08 ms**), performing faster than Full Fine-Tuning. DoRA exhibited slightly higher latency (**110.00 ms**) due to the runtime computational overhead of directional matrix normalization by magnitude vectors.
- **Recall Invariance**: All three adaptation strategies detected the exact same 89 out of 97 test spam messages (91.75% recall with 8 false negatives).
- **Parameter & Storage Efficiency**: LoRA and DoRA reduced trainable parameters by **369.3x** and **347.7x** respectively, while shrinking standalone weights from **417.7 MB** down to **1.14 MB** and **1.21 MB**, retaining over **99.4%** of Full Fine-Tuning's Macro F1.

---

## 🏗️ Project Architecture

```
SpamAdapt/
│
├── data/
│   ├── raw/
│   │   └── raw_sms_spam.csv          # Cached raw UCI SMS Spam Collection
│   └── processed/
│       ├── train.csv                 # 70% Stratified Training split (3,612 samples)
│       ├── val.csv                   # 15% Stratified Validation split (774 samples)
│       ├── test.csv                  # 15% Stratified Test split (774 samples)
│       └── split_metadata.json       # SHA-256 checksums, proportions, and seed
│
├── models/                           # Base models & cached weights
│
├── src/                              # Modular research package
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # Centralized dataclass configuration system
│   ├── data_loader.py                # Dataset fetching, caching, and immutable split loading
│   ├── preprocessing.py              # Signal-preserving cleaning, deduplication, stratified split
│   ├── tokenization.py               # AutoTokenizer & SpamClassificationDataset (preserves raw text)
│   ├── model.py                      # Pretrained BERT sequence classification model loader
│   ├── evaluate.py                   # Reusable metrics computation and confusion matrix plotting
│   ├── benchmark.py                  # Systematic latency, throughput, and memory benchmarking
│   └── utils.py                      # Seed setting, hardware probing, and formatting
│
├── experiments/
│   ├── full_finetuning/              # Baseline track (checkpoint: 417.7 MB)
│   ├── lora/                         # LoRA PEFT track (checkpoint: 1.14 MB)
│   ├── dora/                         # DoRA PEFT track (checkpoint: 1.21 MB)
│   └── run_benchmark.py              # Master benchmark execution runner
│
├── results/
│   ├── metrics/                      # Evaluation output JSONs & benchmark metrics
│   ├── plots/                        # Confusion matrices, training curves & benchmark plots
│   └── reports/                      # Dataset summaries and ablation reports
│
├── app/                              # Interactive Web Application (Stage 6)
│   ├── __init__.py                   # App package initialization
│   ├── main.py                       # Streamlit multi-tab application UI & flow
│   ├── inference.py                  # Cached model loader, device detection & inference pipeline
│   ├── metrics_loader.py             # Safe JSON metric ingestion & verification tables
│   └── styles.py                     # Dark navy, blue/violet custom CSS design system
│
├── PROJECT_STATUS.md                 # Current research project status
├── verify_milestone.py               # Master milestone verification runner
├── test_app.py                       # Automated verification test suite for Stage 6 (7/7 suites)
├── prepare_data.py                   # Dataset preparation script
├── requirements.txt                  # Pinned dependencies (including Streamlit)
└── README.md                         # Project documentation
```

---

## 📊 Dataset & Splitting Methodology

- **Source**: UCI SMS Spam Collection (5,574 original records)
- **Deduplication**: 414 duplicate messages identified and dropped.
- **Cleaned Dataset**: 5,160 unique messages (4,518 Ham / 642 Spam; 7.04 : 1 imbalance).
- **Stratified Splits (`seed=42`)**:
  - **Train (70%)**: 3,612 samples (3,163 Ham [87.57%], 449 Spam [12.43%])
  - **Val (15%)**: 774 samples (678 Ham [87.60%], 96 Spam [12.40%])
  - **Test (15%)**: 774 samples (677 Ham [87.47%], 97 Spam [12.53%])
- **Reproducibility Guarantee**: The exact same processed splits are consumed by all experiments via `src/data_loader.py` with cryptographic SHA-256 validation.

---

## 🤖 Pretrained Model Configuration

- **Model**: `google-bert/bert-base-uncased`
- **Task**: Binary sequence classification (`0: ham`, `1: spam`)
- **Hidden Size**: 768
- **Transformer Layers**: 12
- **Attention Heads**: 12
- **Max Sequence Length**: 128
- **Total Parameters**: 109,483,778 (~110M)

---

## 🌐 Interactive Web Application (Stage 6)

The web dashboard is built using **Streamlit** with a custom dark navy aesthetic (`#090d16`), electric blue and violet accents, and pure Python inference.

### Features
1. **Live SMS Detection**: Enter any text or select curated samples from the UCI test distribution. Run inference across Full Fine-Tuning, LoRA, and DoRA with softmax confidence scores, millisecond latency metrics, and safety disclaimers.
2. **Model Comparison & Benchmarks**: Real-time KPI summaries, side-by-side metric verification tables, and empirical benchmark plots directly rendered from the saved evaluation artifacts.
3. **Architecture & Methodology**: Detailed explanations of BERT encoder representations, Full Fine-Tuning vs. LoRA ($W_0 + \frac{\alpha}{r}BA$) vs. DoRA ($m \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c}$), dataset splits, and experimental limitations.
4. **Session-Only History**: Ephemeral prediction log stored only in memory during the browser session. No disk logging, database persistence, or external API transmission.

### 🚀 Launching the Web App Locally

From the project root:
```bash
streamlit run SpamAdapt/app/main.py
```

Or from inside the `SpamAdapt/` directory:
```bash
cd SpamAdapt
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`.

### 🧪 Running the Stage 6 Verification Suite

An automated verification script (`test_app.py`) validates the complete application workflow:
1. Empty message validation
2. Real inference on HAM and SPAM test samples
3. Model switching (Full Fine-Tuning, LoRA, DoRA)
4. Confidence calculation & softmax probability calibration
5. Model in-memory caching (`@st.cache_resource`)
6. Missing-checkpoint error handling
7. Metrics JSON ingestion and comparison table generation

```bash
python SpamAdapt/test_app.py
```
