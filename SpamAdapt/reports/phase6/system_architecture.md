# SpamAdapt: System Architecture & Mathematical Foundations

**Author:** Senior Machine Learning Research Team & Systems Architects  
**Date:** October 2026  
**Artifact Directory:** `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`  
**Base Architecture:** `google-bert/bert-base-uncased`  

---

## 1. Architectural Overview

SpamAdapt is built upon a modular, clean-architecture Python pipeline designed for sequence classification of mobile text messages. The architecture decouples data ingestion, preprocessing, parameter-efficient adaptation, validation-constrained thresholding, and web-based demonstration.

```
       [ Input SMS Message ]
                 │
                 ▼
     [ Text Sanitization & Safety ] (Max 2,000 chars)
                 │
                 ▼
     [ WordPiece Tokenizer ] (bert-base-uncased, max_len=128)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│               Transformer Backbone                     │
│        (12 Transformer Layers, 768-dim, 12 Heads)       │
│                                                        │
│  ┌───────────────────────┬──────────────────────────┐  │
│  │   Full Fine-Tuning    │       PEFT Adapters      │  │
│  │ (109.5M dense update) │ (LoRA / DoRA on Q, V)    │  │
│  └───────────────────────┴──────────────────────────┘  │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
     [ [CLS] Token Pooling & Dropout ]
                 │
                 ▼
     [ Linear Classification Head ] (768 -> 2 logits)
                 │
                 ▼
     [ Softmax Probability Layer ] -> P(HAM), P(SPAM)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│            Three-Tier Decision Engine                  │
│                                                        │
│   P(SPAM) < tau_low           -> HAM                   │
│   tau_low <= P(SPAM) < tau_hi -> REVIEW_REQUIRED       │
│   P(SPAM) >= tau_hi           -> SUSPECTED_SPAM        │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
     [ Streamlit Web Presentation Layer ]
```

---

## 2. Pretrained Transformer Backbone

SpamAdapt employs **BERT (Bidirectional Encoder Representations from Transformers)** base uncased:
- **Parameters:** 109,483,778 weights.
- **Layers:** 12 bidirectional transformer encoder blocks.
- **Hidden Dimension ($d_{\text{model}}$):** 768.
- **Attention Heads:** 12 heads per layer ($d_k = 64$).
- **Feed-Forward Dimension:** 3,072.
- **Vocabulary:** 30,522 WordPiece subwords.
- **Input Representation:** Given token sequence $x_1, \dots, x_T$, the input vector is constructed by summing token embedding, positional embedding, and segment embedding:
  $$E = E_{\text{token}} + E_{\text{pos}} + E_{\text{seg}}$$
- **Bidirectional Self-Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

## 3. Mathematical Formulations of Fine-Tuning Strategies

### 3.1 Strategy 1: Full Fine-Tuning
In full fine-tuning, every pre-trained weight matrix $W \in \mathbb{R}^{d \times k}$ across all 12 transformer encoder layers and the final classification head is updated via backpropagation:
$$W_{\text{updated}} = W_0 + \Delta W$$
- **Gradient Computation:** $\nabla_W \mathcal{L}$ is computed for all 109.5 million parameters.
- **Optimizer State:** AdamW optimizer maintains two state tensors per parameter (first moment $m_t$ and second moment $v_t$), requiring $\sim 1.3\text{ GB}$ of optimizer memory during training.
- **Capacity:** Maximum expressive capacity; can adapt both low-level and high-level representations to out-of-domain structures.

---

### 3.2 Strategy 2: Low-Rank Adaptation (LoRA)
Introduced by Hu et al. (2021), LoRA hypothesizes that parameter updates during downstream adaptation have a low intrinsic dimension. Instead of updating the full dense matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update into two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} (B \times A)$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$, and $\alpha$ is a scaling hyperparameter.

- **Initialization:** $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$, ensuring $\Delta W = 0$ at training onset.
- **Forward Pass:**
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B (A x)$$
- **Target Modules in SpamAdapt:** Injected exclusively into the multi-head self-attention **Query** and **Value** projection matrices across all 12 layers:
  - Base dimensions: $d = 768, k = 768$.
  - Rank: $r = 8$.
  - Scaling: $\alpha = 16$.
  - Dropout: $0.1$.
- **Trainable Parameters:**
  $$\text{Per layer} = 2 \times (768 \times 8 + 8 \times 768) = 24,576 \text{ params}$$
  $$\text{12 layers} = 294,912 \text{ params} + 1,538 \text{ head params} = 296,450 \text{ trainable weights (0.27%)}$$

---

### 3.3 Strategy 3: Weight-Decomposed Low-Rank Adaptation (DoRA)
Introduced by Liu et al. (2024), DoRA builds upon LoRA by addressing a fundamental divergence between Full Fine-Tuning and LoRA: Full Fine-Tuning exhibits independent variation in weight magnitude and directional orientation, whereas standard LoRA entangles magnitude and direction updates proportionally.

DoRA decomposes the pre-trained weight matrix into two decoupled components:
1. **Magnitude Vector ($m \in \mathbb{R}^{1 \times k}$):** Controls the scale of each column vector.
2. **Directional Matrix ($V \in \mathbb{R}^{d \times k}$):** Defines column unit vectors.

The weight matrix is formulated as:
$$W = m \frac{V}{\|V\|_c} = m \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c}$$
where $\|\cdot\|_c$ denotes the $L_2$ norm of each column vector, and $\Delta W = \frac{\alpha}{r} B A$.

- **Initialization:** $m = \|W_0\|_c$, and $\Delta W = 0$, exactly matching the pre-trained model at the start.
- **Training Dynamics:** The learnable magnitude vector $m$ independently scales column magnitudes, while the LoRA matrices $B$ and $A$ optimize subtle directional rotations.
- **Trainable Parameters in SpamAdapt:**
  $$296,450 \text{ (LoRA)} + 24 \times 768 \text{ (magnitude vectors)} = 314,882 \text{ trainable weights (0.29%)}$$

---

## 4. Decision Engine & Abstention Policy

Binary softmax outputs $p_1 = P(\text{SPAM} \mid x)$ are converted into three-tier operational decisions:

$$\text{Decision}(p_1) = \begin{cases} 
\text{HAM}, & \text{if } p_1 < \tau_{\text{low}} \\ 
\text{REVIEW\_REQUIRED}, & \text{if } \tau_{\text{low}} \le p_1 < \tau_{\text{high}} \\ 
\text{SUSPECTED\_SPAM}, & \text{if } p_1 \ge \tau_{\text{high}} 
\end{cases}$$

- **Threshold Derivation:** Calibrated strictly on the held-out augmented validation set (`augmented_val.csv`, $N=562$).
- **Full Fine-Tuning v2:** $\tau_{\text{base}} = 0.10, \tau_{\text{low}} = 0.05, \tau_{\text{high}} = 0.35$.
- **LoRA v2 & DoRA v2:** $\tau_{\text{base}} = 0.18, \tau_{\text{low}} = 0.08, \tau_{\text{high}} = 0.45$.

---

## 5. Software Architecture & In-Memory Isolation

- **In-Memory Caching:** `@st.cache_resource` in `app/inference.py` ensures model weights and tokenizers are loaded into memory once and reused across queries ($<0.5\text{ ms}$ cache lookup).
- **Zero Disk Persistence:** Input messages are evaluated in local volatile memory and discarded immediately; no persistent databases, logs, or external telemetry are utilized.
- **Explicit Failure Surfacing:** `load_cached_model()` strictly checks checkpoint directories and raises `FileNotFoundError` or `ValueError` upon failure, preventing silent fallback to arbitrary baseline models.
