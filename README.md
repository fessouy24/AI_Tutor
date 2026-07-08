# AI Tutor: AKT Knowledge Tracing + RAG Tutoring

AI Tutor is a graduation project that combines an AKT-style attention-based knowledge tracing model with a RAG-grounded Socratic tutoring layer. The student model predicts next-response correctness from chronological EdNet interaction sequences, while the tutoring layer uses retrieval evidence, a lightweight student-twin state, answer-leakage checks, explanation artifacts, and ONNX export for deployment.

## Highlights

- Trained an AKT-style knowledge tracing model on multi-source EdNet logs, using question, skill, correctness, and position embeddings with causal multi-head attention.
- Converted raw EdNet answers into correctness labels by matching each response against official `correct_answer` metadata, then windowed student histories into masked length-200 prediction sequences.
- Added RAG retrieval, Socratic hint generation, answer-leakage detection, student-twin personalization, attention-based explainability outputs, efficiency benchmarks, and an ONNX deployment path.
- Final reported EdNet evaluation: AUC `0.7874`, accuracy `0.7213`, F1 `0.7774`, PR-AUC `0.8381`, over `32.27M` test tokens.

## Repository Structure

```text
AI Tutor/
  configs/          Experiment, local, Kaggle, and smoke-test configuration files
  notebooks/        Local orchestration notebook
  src/data/         Dataset parsing, EdNet preprocessing, sequence building
  src/models/       AKT model, training loop, checkpointing, optimizer setup
  src/rag/          Document loading, chunking, embeddings, and retrieval
  src/tutor/        Socratic tutor generation, leakage checks, student twin logic
  src/xai/          Attention and SHAP-style explanation helpers
  src/deployment/   ONNX export and inference
frontend/           Browser research console prototype
```

Generated datasets, checkpoints, logs, model outputs, and thesis/presentation files are intentionally excluded from Git.

## Setup

```bash
cd "AI Tutor"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the required datasets separately, then update the paths in `AI Tutor/configs/local.yaml`.

## Common Commands

Train the AKT model:

```bash
cd "AI Tutor"
python train_akt.py --config configs/local.yaml
```

Run the tutoring demo:

```bash
cd "AI Tutor"
python run_tutor_demo.py --config configs/local.yaml --question "I am stuck solving equations with variables on both sides"
```

Export a trained checkpoint to ONNX:

```bash
cd "AI Tutor"
python -m src.deployment.export_onnx --config configs/local.yaml
```

Run the frontend prototype:

```bash
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Notes

- The final measured student model is AKT-style knowledge tracing. Older DKT wording was used during early prototyping and should not be used to describe the final model.
- Keep dataset files, notebooks with private credentials, checkpoints, and generated outputs out of public commits.
