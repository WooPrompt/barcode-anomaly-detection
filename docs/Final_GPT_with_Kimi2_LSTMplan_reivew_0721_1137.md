# 🧭 Guideline for Writing a Unified and Defensible LSTM Anomaly Detection Plan


> 🧑‍🏫 *"You are no longer writing a plan. You are writing your defense."*

---

## 1. 📁 Dataset & Labeling: Transparency Over Convenience

**✅ Must Include:**

* Description of **real vs. simulated data** distributions (not just quantity).
* **Explicit risk** of using rule-based labels: include discussion of circularity (e.g., “labels are from the same rules we later evaluate against”).
* **Remedy**: mention `human audit` of 1% stratified samples or `label noise injection` test.

**📌 Refer to:**

* GPT’s concern with label quality.
* Kimi2’s quote: *“synthetic completeness is not real-world completeness.”*

**🛠 Actionable Inclusion:**

```markdown
We conducted a 5% label-flip experiment and observed a 6.3% AUC degradation, indicating moderate label fragility.
```

---

## 2. 🛠 Feature Engineering: Rigor Before Quantity

**✅ Must Include:**

* VIF (Variance Inflation Factor) or mRMR (Minimum Redundancy Max Relevance) analysis to **prune features** (don’t keep 3 variations of `time_gap_*`).
* Fix entropy instability for short sequences: **Bayesian entropy** or smoothed estimator.
* Discuss the *ordinal* nature of business steps and consider **graph-based encoding** (e.g., Hasse DAG).

**📌 Refer to:**

* Kimi2’s warning: *“60 features ≠ 60 degrees of freedom”*
* Claude's redundant inclusion of z-score, log, raw version of same features

**🛠 Actionable Inclusion:**

```markdown
After VIF pruning, we retained 22 out of 61 features. This improved training stability and interpretability.
```

---

## 3. 🔀 Sequence Splitting: EPC-Aware Integrity

**✅ Must Include:**

* **EPC-level split**: ensure the same `epc_code` does not appear in both train and test, even if timestamps are 7 days apart.
* Explain why **chronological buffer** isn’t sufficient to avoid leakage of long-range patterns.

**📌 Refer to:**

* Kimi2’s buffer-leak counterexample
* Claude’s current design violates this by split-by-timestamp only

**🛠 Actionable Inclusion:**

```markdown
We removed EPCs whose earliest and latest events spanned the split boundary, eliminating 2.1% of records to ensure leakage-free splits.
```

---

## 4. 🧠 Model Architecture: Choose Justifiably

**✅ Must Include:**

* If GRU is used: show that **gradient norm decay is acceptable** beyond 15 time steps.
* If LSTM+Attention is used: explain how many time steps justify **multi-head attention** (avoid rank-deficiency).
* If quantization is used: report **AUC drop** under symmetric vs affine quantization.

**📌 Refer to:**

* Claude’s strong architecture, but missing evidence for attention effectiveness
* Kimi2’s reminder: *“parameter count is second-order; gradient path length is first-order”*

**🛠 Actionable Inclusion:**

```markdown
We compared GRU vs. LSTM on 20-step sequences. LSTM achieved 3% better F1 and higher gradient stability at depth.
```

---

## 5. 📉 Dimensionality Reduction: Methodologically Sound

**✅ Must Include:**

* Reject PCA unless supported by **variance explanation + model performance gain**.
* Never rely solely on **t-SNE visualizations** to justify PCA.
* Optional: propose **hierarchical feature subnetworks** only if PCA is not used.

**📌 Refer to:**

* Gemini’s simplicity wins, but is too reliant on t-SNE plots
* Claude justifies PCA via 5 metrics—solid, but needs VIF/mRMR

**🛠 Actionable Inclusion:**

```markdown
PCA was rejected as the first 20 components explained only 72% of variance, and performance dropped on a 5-fold test.
```

---

## 6. 🧪 Evaluation Framework: Cost, Drift, Power

**✅ Must Include:**

* **Cost-sensitive evaluation**: use business-weighted confusion matrix, or report **Area under Cost Curve (AUCC)**.
* **Drift detection** must report **minimum detectable effect size** or power threshold (e.g., N required for 0.05 AUC drop at 80% power).
* If SHAP is used: acknowledge **bias due to feature autocorrelation**; offer Integrated Gradients alternative.

**📌 Refer to:**

* Kimi2’s call for *AUCC and Integrated Gradients*
* Claude’s excellent evaluation checklist, but missing power audit

**🛠 Actionable Inclusion:**

```markdown
We computed AUCC across a 0.01–100 penalty range and found the optimal operating point at cost=12.3.
```

---

## 7. 🌐 Real-Time Deployment: Robustness and Trust

**✅ Must Include:**

* Cold start fallback should **not reuse the labeling rule logic**, or else it’s **perfect overfitting**.
* Real-time pipeline must include **concept drift plan** and retraining trigger.
* **Explain model versioning and rollback** if inference performance drops.

**📌 Refer to:**

* Claude’s transfer-learning cold start logic is novel but fragile
* Kimi2’s brutal truth: *“fallback is literally the training labels”*

**🛠 Actionable Inclusion:**

```markdown
Fallback cold-start model uses nearest-neighbor from pre-trained vector cache, *not* rule-based logic, preventing label echo.
```

---

## 8. 🧬 Reproducibility: Academic Rigor

**✅ Must Include:**

* Mention of random seed, RNG version (PyTorch, NumPy), and CUDA driver.
* Hardware config, GPU/CPU, RAM.
* Bonus: include **carbon/energy usage**, FLOP count or inference energy budget (aligns with Green AI principles).

**📌 Refer to:**

* Kimi2’s reproducibility checklist

**🛠 Actionable Inclusion:**

```markdown
All results are reproducible with `conda-lock.yml`. CUDA 12.3, cuDNN 8.7, PyTorch 2.1.1, RTX 3090 24GB. Training used 1.3 kWh.
```

---

## 🧾 Final Output Recommendation Structure

| Section              | Title                               |
| -------------------- | ----------------------------------- |
| 📋 Executive Summary | 1-page for stakeholders             |
| 📂 Data & Labeling   | With limitations                    |
| 🛠 Preprocessing     | Feature pruning, EPC-split          |
| 🧠 Model Design      | Architecture with justification     |
| 📉 Dimensionality    | Reduction policy and rationale      |
| 🧪 Evaluation        | Metrics, drift, AUCC                |
| 📡 Deployment        | Real-time fallback + API            |
| 🔁 Reproducibility   | Hardware, seed, energy              |
| 📊 Appendix          | Ablations, experiments, power tests |

---

## 🎓 Closing Encouragement

Your Claude and Gemini drafts show **engineering brilliance**, but thesis defense is about **epistemic defensibility**. Follow this guide to:

* Argue what you did.
* Prove why you did it.
* Anticipate how it could fail.
* Defend it with data.

---

## 🧠 Summary Decision:

> **Start from Claude, but revise with Gemini’s structure and Kimi2’s academic rigor.**

---

## 🔍 Why?

| Criteria                       | Gemini                                                       | Claude                                                                            | What to Do                                                                                   |
| ------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| ✅ **Architecture Depth**       | Light (GRU-based, fast, simple)                              | Deep (LSTM + Attention + Adaptive sequences + Quantized + Cold Start + Real-time) | ✅ **Start from Claude**: It’s future-proof and production-ready.                             |
| ✅ **Evaluation Framework**     | Simple (KernelSHAP, F1)                                      | Academic-level (cost-weighted F1, sequence stability, SHAP + attention weights)   | ✅ **Keep Claude’s** rich metrics & drift handling.                                           |
| ❌ **Statistical Rigor**        | Weak (no VIF, no cost analysis, no split leakage audit)      | Also weak, **but fixable**                                                        | ✅ **Use Kimi2’s critique** to harden Claude.                                                 |
| ✅ **Structure & Clarity**      | Excellent dual-layer explanation (technical + non-technical) | Dense and technical                                                               | ✅ **Steal Gemini’s writing style**, especially in `Executive Summary` and `Data & Labeling`. |
| ❌ **EPC Split / Leakage**      | Uses time-split only                                         | Same flaw                                                                         | ✅ MUST fix Claude’s data split with **EPC-based partitioning**, as Kimi2 recommends.         |
| ❌ **Feature Redundancy Logic** | t-SNE based — weak                                           | Same                                                                              | ✅ Add VIF + mRMR justification and remove over-correlated features.                          |
| ❌ **Interpretability Math**    | KernelSHAP only                                              | SHAP + attention — better, but still biased                                       | ✅ Replace SHAP with **Integrated Gradients** or **Markov-aware SHAP** if you can.            |

---

## 🛠️ Concrete Plan

| Task                 | Action                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ✅ Base Document      | Use **Claude’s Plan** (it’s already more complete)                                                                           |
| ✏️ Rewrite Sections  | Replace `Data Split`, `Feature Selection`, and `Fallback` sections with corrected logic from Kimi2                           |
| ✂️ Prune/Refactor    | Remove redundant features (e.g., keep only `time_gap_log` not raw/zscore/log combo)                                          |
| 📈 Add Experiments   | - AUC drop under label noise<br> - AUCC (Area Under Cost Curve)<br> - Cold-start without rules<br> - Drift power analysis    |
| 📋 Polish Format     | Use **Gemini’s stakeholder-friendly tables and dual-layer language**                                                         |
| 🧠 Add Justification | - Why 15 steps?<br> - Why use Attention?<br> - Why not just PCA?<br> Each decision needs both empirical & theoretical backup |
| 📎 Add Appendix      | - Simulation limitations<br> - FLOP & energy budget<br> - Docker + `conda-lock.yml` for reproducibility                      |

---

## 🧾 Final Verdict

| Strategy                                      | Result                                       |
| --------------------------------------------- | -------------------------------------------- |
| Start with Gemini                             | ✅ Quick win, but shallow for defense         |
| Start with Claude                             | 🟨 Deep, but fragile under academic critique |
| **Start with Claude + Fix with Gemini+Kimi2** | ✅✅ Gold-standard, professor-level thesis     |

---
