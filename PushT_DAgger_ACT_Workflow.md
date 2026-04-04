# PushT DAgger & ACT Project Workflow

Project method details:
We will be training two baseline policies that we reimplement from scratch (using python libraries as necessary) on the PushT dataset, to isolate the effect of chunking on task performance, these policies will be standard Behavior Cloning, BC with DAgger, and Chunked Behavior Cloning (using ACT). Our proposed methods are: (1) We adapt the human intervention logic in DAgger into ACT; (2) upon that, we implement our Residual Chunk Blending Algorithm. We will then measure how success rate changes under human intervention, comparing performance with and without our Residual Chunk Blending algorithm.

---

## 1. Data and Format

### 1.1. Pretraining Data Source
- Existing dataset LeRobot/pusht, ensure data compatibility (observation/action format, environment domain)

### 1.2. Data Format
- Canonical storage uses **per-step raw actions** in `.npz` files (LeRobot-style):
  - `observation.state`: `(N, 2)`
  - `action`: `(N, 2)`
  - `next.reward`, `next.done`, `next.success`, `frame_index`, `timestamp`
- ACT chunk targets are generated at load time via sliding window:
  - `action_chunk[t] = [a_t, a_{t+1}, ..., a_{t+H-1}]` with tail padding at episode end.

---

## 2. Pretraining

### 2.1. Standard BC (for DAgger)
- Train a policy to map state → action using the demonstration data.
- Use supervised learning (e.g., MSE loss for continuous actions).

### 2.2. Chunked BC (using ACT paper)
- Train a chunked policy to map state → action_chunk using chunked demonstration data.
- The model outputs a sequence of actions per input state.

### 2.3. Model Initialization
- We will have two separate pretrained models:
  - DAgger/BC: state → action
  - ACT/Chunked BC: state → action_chunk

---

## 3. Interactive Training

### 3.1. DAgger
- Roll out the current policy in the environment.
- Allow a human (or expert) to provide corrective actions when the policy is uncertain or makes mistakes.
- Aggregate new (state, expert action) pairs into the dataset.
- Retrain the policy using BC training on the aggregated dataset after each iteration.

### 3.2. ACT (Chunked DAgger)
- Roll out the chunked policy (state → action_chunk) in the environment.
- Allow human intervention to provide corrective action chunks as needed.
- Aggregate new (state, expert action_chunk) pairs into the dataset.
- Retrain the old chunked BC policy after each iteration.

### 3.3. Residual Chunk Blending (Proposed Method)
- Implement the Residual Chunk Blending logic on top of ACT.
- $a_{blended}= (1 - \lambda) \cdot a_{base} + \lambda \cdot a_{human}$

---

## 4. Evaluation

- Measure and compare the success rate of:
  - Standard DAgger/BC
  - ACT/Chunked BC
  - ACT/Chunked BC + Residual Chunk Blending
- Evaluate under both autonomous and human intervention conditions.
- Analyze the effect of chunking and your proposed method on task performance.

---

## 5. Notes & Best Practices

- Always pretrain each model (BC for DAgger, chunked BC for ACT) before interactive training.
- Use the same demonstration data for both models where possible for fair comparison.
- Ensure data format and model architecture match (single action vs. action chunk).
- If using external datasets, write conversion scripts to ensure compatibility.

---

## 6. References
- DAgger: Ross et al., 2011 (https://arxiv.org/abs/1011.0686)
- ACT: Zhao et al., 2023 (https://arxiv.org/abs/2304.13705)

<!--
---

_This workflow is designed to help you systematically compare standard and chunked imitation learning methods, and to evaluate your proposed improvements in a controlled, reproducible way._
-->
