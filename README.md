# SpamAdapt: Parameter-Efficient vs. Full Fine-Tuning for Message Spam Detection

**SpamAdapt** is an empirical machine learning research project that systematically benchmarks and compares three transformer adaptation paradigms for SMS and message spam classification:

1. **Full Fine-Tuning**: Updating all 109.5M parameters of the pretrained transformer.
2. **LoRA (Low-Rank Adaptation)**: Freezing base weights and injecting trainable low-rank decomposition matrices ($r=8, \alpha=16$).
3. **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Decomposing weights into directional and magnitude components, applying LoRA solely to the directional update.

The objective is to systematically benchmark **predictive performance** (Accuracy, Precision, Recall, F1, Macro-F1) against **computational efficiency** (trainable parameter ratio, memory footprint, training throughput, and latency).

---

## 📌 Repository Organization

All implementation code, models, datasets, benchmarks, test suites, and interactive applications are organized within the [`SpamAdapt/`](SpamAdapt/) directory:

```
Scam_Detection/
├── SpamAdapt/
│   ├── app/                          # Interactive Streamlit Web Application (Stage 6)
│   │   ├── main.py                   # Multi-tab Streamlit dashboard
│   │   ├── inference.py              # In-memory cached PyTorch inference engine
│   │   ├── metrics_loader.py         # Benchmark & JSON metric ingestion
│   │   └── model_registry.py         # Multi-version candidate model registry
│   │
│   ├── src/                          # Core research package
│   │   ├── model.py                  # BERT, LoRA, and DoRA model builders
│   │   ├── tokenization.py           # AutoTokenizer & PyTorch Datasets
│   │   ├── evaluate.py               # Scientific evaluation & metrics module
│   │   ├── benchmark.py              # Latency, memory, and throughput profiling
│   │   ├── preprocessing.py          # Signal-preserving text normalization
│   │   └── data_loader.py            # Stratified split loading with SHA-256 checks
│   │
│   ├── experiments/                  # Fine-tuning tracks & checkpoints
│   │   ├── full_finetuning/          # Baseline Full Fine-Tuning
│   │   ├── lora/                     # LoRA PEFT adapters
│   │   ├── dora/                     # DoRA PEFT adapters
│   │   └── run_benchmark.py          # Benchmark runner
│   │
│   ├── data/                         # Processed datasets and external benchmarks
│   │   ├── processed/                # Preserved UCI SMS Spam Collection (70/15/15)
│   │   └── external_v2/              # Out-of-distribution benchmark (220 real-world samples)
│   │
│   ├── results/                      # Empirical outputs
│   │   ├── metrics/                  # JSON performance reports across all models
│   │   └── plots/                    # High-res confusion matrices & benchmark curves
│   │
│   ├── tests/                        # Comprehensive test & audit suites
│   ├── submission/                   # Final academic report, slides & viva notes
│   ├── requirements.txt              # Pinned dependencies
│   └── test_app.py                   # End-to-end application verification suite
│
├── .gitignore                        # Standardized repository ignore rules
└── README.md                         # Main repository entry point
```

---

## 🔬 Empirical Results Summary (Held-Out Test Split, N=774)

| Metric / Attribute | Baseline (Full Fine-Tuning) | LoRA (PEFT, $r=8$) | DoRA (PEFT, $r=8$) | Trade-Off & Efficiency Insights |
| :--- | :---: | :---: | :---: | :--- |
| **Base Model** | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` | Identical base weights |
| **Total Parameters** | 109,483,778 | 109,780,228 | 109,798,660 | +18.4k DoRA magnitude parameters |
| **Trainable Parameters** | 109,483,778 (100.0%) | **296,450 (0.27%)** | 314,882 (0.29%) | **~350x–370x parameter reduction** |
| **Adapter Storage on Disk** | 417.67 MB | **1.14 MB** | **1.21 MB** | **345x–366x storage saving** |
| **Mean Inference Latency** | 99.48 ms | **92.05 ms** | 110.00 ms | **LoRA is 7.5% faster than Full FT** |
| **P95 Latency Tail** | 120.36 ms | **102.08 ms** | 124.72 ms | LoRA achieves tightest latency distribution |
| **Test Accuracy** | **98.45%** (762/774) | 98.32% (761/774) | 98.19% (760/774) | Spread within 0.26% |
| **Spam Recall** | **91.75%** (89/97) | **91.75%** (89/97) | **91.75%** (89/97) | **Identical recall (89/97 detected)** |
| **Macro F1-Score** | **96.40%** | 96.12% | 95.84% | All retain >99.4% of Full FT Macro F1 |

---

## 🚀 Quick Start & Execution

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/akash9313/Scam_Detection.git
cd Scam_Detection
pip install -r SpamAdapt/requirements.txt
```

### 2. Run Verification Suite (7/7 Automated Tests)
Verify models, inference pipelines, and metric loaders:
```bash
python SpamAdapt/test_app.py
```

### 3. Launch the Interactive Web Application
Start the Streamlit application for real-time spam detection and interactive benchmarking:
```bash
python -m streamlit run SpamAdapt/app/main.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Benchmarks
Execute the computational and efficiency benchmarking suite:
```bash
python SpamAdapt/experiments/run_benchmark.py
```

---

## 📄 Academic Deliverables & Reports

Academic documentation, defense materials, and conference report sources are available inside [`SpamAdapt/submission/SpamAdapt_Final_Submission/`](SpamAdapt/submission/SpamAdapt_Final_Submission/):
- **`Project_Report.md`**: Formal Technical Defense Report & Thesis
- **`SpamAdapt_Final_Presentation.pptx`**: 14-Slide Defense Presentation
- **`Evaluation_Summary.md`**: Empirical Evaluation Reference & In-Distribution vs Out-of-Distribution Breakdown
- **`Viva_Quick_Revision.md`**: High-Yield Viva Voce Defense Sheet
- **`SpamAdapt_Report.tex`**: Complete IEEE Conference Paper LaTeX Source
