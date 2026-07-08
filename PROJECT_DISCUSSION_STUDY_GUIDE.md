# AI Tutor Project Discussion Study Guide

This guide explains what is behind the code in this project and how to defend it in a discussion. The cleanest way to describe the project is:

> The project builds an AI tutoring prototype whose core measured model is an AKT-style attention-based knowledge tracing model trained on EdNet. It predicts whether a student will answer the next question correctly. Around that model, the project adds a RAG-grounded Socratic tutoring layer, student-twin personalization state, explanation artifacts, evaluation graphs, and an ONNX deployment path.

Do not describe the final trained model as DKT. The public-facing frontend labels now use AKT wording, but the final backend model and metrics remain the source of truth.

## 1. Main Components

The code is organized around these parts:

- `AI Tutor/src/data/knowledge_tracing.py`: dataset detection, parsing, preprocessing, caching, splitting, and dataloader creation.
- `AI Tutor/src/models/akt.py`: the AKT-style attention model.
- `AI Tutor/src/models/training.py`: training loop, masked BCE loss, validation, checkpointing, test evaluation.
- `AI Tutor/train_akt.py`: command-line training entry point.
- `AI Tutor/src/eval/*`: final metrics, enhanced evaluation, plots, confusion matrix, ROC, PR curve, efficiency benchmarking.
- `AI Tutor/src/rag/*`: document loading, chunking, embedding, Chroma/TurboVec retrieval.
- `AI Tutor/src/tutor/*`: Socratic tutor generation, answer leakage detection, student twin storage.
- `AI Tutor/src/xai/*`: explanation artifacts using AKT attention and optional SHAP-style feature summaries.
- `AI Tutor/src/deployment/*`: ONNX export and ONNX inference.
- `frontend/`: prototype research console. It demonstrates intended interaction flow with simplified browser-side retrieval and prototype inference wiring.
- `AI Tutor/notebooks/AI_Tutor_Local.ipynb`: orchestration notebook for local execution.
- `AI Tutor/notebooks/data_handling.ipynb`: dataset discovery/organization notebook.

## 2. Dataset Organization

The project uses `Datasets (Organized)` as the cleaned dataset root.

Final EdNet sources used by the final AKT run:

- `EdNet_KT1`
- `EdNet_KT2`
- `EdNet_KT3`
- `EdNet_KT4`
- `EdNet_Contents/contents/questions.csv`

The metadata file `questions.csv` contains fields like:

- `question_id`
- `correct_answer`
- `tags`
- `bundle_id`
- `part`
- `deployed_at`

The student interaction files contain rows like:

- KT1: `timestamp, solving_id, question_id, user_answer, elapsed_time`
- KT2/KT3: `timestamp, action_type, item_id, source, user_answer, platform`
- KT4: same style as KT3 with extra fields such as `cursor_time`

The RAG source folder in the final local setup is `Datasets (Organized)/RAG_Datasets`; in the current artifact manifest it contains `0062670.pdf`, which is indexed into chunks.

## 3. Dataset Discovery And Preparation

The data handling notebook searches for local, Kaggle, and organized dataset folders. It copies datasets into `Datasets (Organized)` so the training code can rely on stable paths.

Important behavior:

- It can download or discover EdNet-related datasets.
- It promotes raw folders like `KT2` and `EdNet-Contents` into clean organized folders.
- It writes inventory files such as `dataset_inventory_manifest.json`, `dataset_inventory_summary.csv`, and `organized_paths.json`.
- It separates RAG material under `RAG_Datasets` instead of mixing tutoring documents with KT interaction files.

Security note: never commit Hugging Face, OpenAI, Kaggle, or other private access tokens. Set them through environment variables before running dataset download notebooks.

## 4. How Columns Are Parsed

The parser is designed to support multiple KT dataset formats. It has lists of possible names for student, question, skill, correctness, timestamp, and mask columns.

Examples:

- Student columns: `student_id`, `user_id`, `anon_student_id`
- Question columns: `question_id`, `item_id`, `problem_id`, `questions`
- Skill columns: `skill_id`, `concept_id`, `tags`, `KC(SubSkills)`
- Correct columns: `correct`, `is_correct`, `responses`
- Time columns: `timestamp`, `order_id`, `action_num`

If a dataset already stores whole sequences in one row, such as `questions`, `concepts`, and `responses`, the parser treats it as pre-sequenced. If every row is one interaction, it groups rows by student and sorts them by timestamp.

For EdNet, the project uses a special parser because EdNet student files do not directly say correct or incorrect. They contain the student's answer, so the parser must compare it with `questions.csv`.

## 5. EdNet Parsing In Detail

The EdNet parser does this:

1. Finds `questions.csv` from `EdNet_Contents`.
2. Reads `question_id`, `correct_answer`, and `tags`.
3. Builds a `correct_map`: question ID -> correct answer.
4. Builds a `skill_map`: question ID -> first tag.
5. Recursively scans student CSV files under each source folder.
6. For KT2/KT3/KT4, keeps only rows where `action_type == respond`.
7. Drops rows where the question ID or user answer is missing.
8. Sorts the student's rows by timestamp.
9. For each response row:
   - Reads the raw question ID.
   - Looks up the correct answer from metadata.
   - Lowercases the user's answer.
   - Converts correctness to `1` if user answer equals correct answer, otherwise `0`.
   - Maps question IDs and skill tags to integer IDs.
10. Creates one sequence per student file.

The label is therefore not guessed from `user_answer`. It is computed by comparing `user_answer` against the official EdNet `correct_answer`.

## 6. Important Preprocessing Choices

Questions and skills are mapped to integer IDs starting at 1. ID 0 is reserved for padding.

When multiple skills/tags exist, the current parser uses the first tag only. For example, tags like `1;2;179;181` become skill `1`. This simplifies the model input but loses multi-skill information.

Each source gets a prefix before merging, like:

- `ednet_kt1::q5012`
- `ednet_kt3::q5012`

This prevents collisions across datasets. The tradeoff is that the same EdNet question appearing in different KT levels is treated as different tokens. That is conservative for multi-source parsing, but it reduces shared learning across repeated EdNet question IDs.

The parser filters out sequences shorter than `min_seq_len + 1`. With `min_seq_len = 3`, a sequence needs at least 4 interactions, because predicting the next answer requires at least one target shifted from history.

Processed data is cached under `AI Tutor/outputs/processed`. Source-level caches avoid reparsing millions of files every time. The final processed EdNet cache is `kt_processed_ednet_only.pt`.

## 7. Final Dataset Scale

Final reported EdNet run:

- Dataset/run: `multi_source:ednet_kt3,ednet_kt4,ednet_kt2_uploaded,ednet_kt1`
- Questions: `46,948`
- Skills: `568`
- Total source sequences: `3,481,042`
- Final test tokens: `32,273,267`
- Dataloader build time: about `1,569` seconds in the final summary
- Epochs completed: `20`

Source sequence counts:

- `ednet_kt3`: `276,533`
- `ednet_kt4`: `276,533`
- `ednet_kt2_uploaded`: `2,212,264`
- `ednet_kt1`: `715,712`

Split counts by source:

- Train: `2,784,833` source sequences
- Validation: `348,104` source sequences
- Test: `348,105` source sequences

The split is approximately 80 percent train, 10 percent validation, and 10 percent test.

## 8. How Sequence Windows Become Model Inputs

The model predicts the next response. For a student sequence:

`(q1, s1, c1), (q2, s2, c2), ..., (qn, sn, cn)`

the dataset creates:

- input questions: `q1 ... q(n-1)`
- input skills: `s1 ... s(n-1)`
- input correctness: `c1 ... c(n-1)`
- target questions: `q2 ... qn`
- target skills: `s2 ... sn`
- labels: `c2 ... cn`

So at each position, the model uses past interaction information to predict whether the next target question/skill will be answered correctly.

`max_seq_len = 200`, so each training window contains up to 200 prediction positions. The original window includes up to 201 raw interactions because the model needs the next interaction as target.

Padding:

- Padded question and skill IDs are 0.
- Padded correctness is 0, but the model remaps valid correctness to `1` or `2`.
- The mask says which positions are real and which are padding.

Loss and metrics are computed only on masked real positions.

## 9. AKT Model Architecture

The final student model is an AKT-style attention-based Knowledge Tracing model.

Configuration:

- `d_model = 128`
- `n_heads = 8`
- `n_layers = 3`
- `dropout = 0.2`
- `ff_multiplier = 4`
- `max_seq_len = 200`
- Parameters: about `6,819,073`

Embeddings:

- Question embedding: `(n_questions + 1) x 128`
- Skill embedding: `(n_skills + 1) x 128`
- Correctness embedding: 3 tokens: padding, incorrect, correct
- Position embedding: 200 positions

For each past interaction, the model concatenates:

- question embedding
- skill embedding
- correctness embedding

That 384-dimensional vector is projected back to 128 dimensions with a linear layer, layer norm, and dropout. A positional embedding is added.

The target question and target skill are embedded separately, concatenated, and projected to a 128-dimensional target vector.

Each AKT block contains:

- LayerNorm
- Multi-head self-attention
- Causal mask so future positions cannot be seen
- Key padding mask so padded tokens are ignored
- Residual connection
- LayerNorm
- Feed-forward network: `128 -> 512 -> 128`
- GELU activation
- Dropout
- Residual connection

After 3 attention blocks, the model concatenates:

- the learned history state
- the target question/skill representation

Then a prediction head produces one logit per time step:

`Linear(256 -> 128) -> GELU -> Dropout -> Linear(128 -> 1)`

The logit is passed through sigmoid during evaluation to become a probability of correctness.

## 10. Why Attention And Causal Masking Matter

Attention lets the model decide which previous interactions are most relevant to the current target question. Causal masking prevents cheating: the model cannot attend to future interactions when predicting the next answer.

Without the causal mask, the model could accidentally use information from after the target, which would inflate metrics and not reflect a real tutoring scenario.

## 11. Training Pipeline

Training starts in `train_akt.py`:

1. Load config.
2. Set model type to AKT.
3. Create output folders.
4. Set random seed.
5. Select CUDA if available.
6. Build dataloaders, which triggers preprocessing or cache loading.
7. Build AKT model using detected `n_questions` and `n_skills`.
8. Train with validation after every epoch.
9. Save best and last checkpoints.
10. Evaluate on test split.
11. Save `akt_training_summary.json`.

Training settings:

- Epochs: `20`
- Learning rate: `0.001`
- Weight decay: `0.01`
- Scheduler: cosine decay
- Mixed precision: enabled on CUDA
- Gradient clipping: `1.0`
- Early stopping patience: `5`
- Optimizer: custom eager AdamW implementation

The custom AdamW avoids some Windows/Python/TorchDynamo issues while following decoupled AdamW update logic.

Loss:

- Binary cross entropy with logits
- Reduction is `none`
- Then the mask is applied
- Final loss is average BCE over valid tokens only

## 12. Training Behavior

The final training curve improved smoothly:

- Epoch 0 validation AUC: about `0.7593`
- Epoch 19 validation AUC: about `0.7865`
- Epoch 0 train loss: about `0.5754`
- Epoch 19 train loss: about `0.5350`
- Epoch 0 validation loss: about `0.5636`
- Epoch 19 validation loss: about `0.5386`

The best validation AUC was the last epoch, so early stopping did not stop the run early.

## 13. Final Test Results

Final AKT test metrics:

- AUC: `0.7874`
- Accuracy: `0.7213`
- Precision: `0.7392`
- Recall: `0.8197`
- F1-score: `0.7774`
- PR-AUC: `0.8381`
- BCE loss: `0.5378`
- Evaluated tokens: `32,273,267`

Confusion matrix at threshold 0.5:

- True negatives: `7,575,766`
- False positives: `5,540,500`
- False negatives: `3,454,174`
- True positives: `15,702,827`

Interpretation:

- The model is better at catching correct answers than incorrect answers, because recall is high.
- There are many false positives, meaning it sometimes predicts a student will be correct when they are actually wrong.
- AUC is the most important KT metric because it measures ranking quality across all thresholds, not just the fixed 0.5 threshold.

## 14. Evaluation Artifacts

Key output files:

- `AI Tutor/outputs/akt_training_summary.json`
- `AI Tutor/outputs/training_log.json`
- `AI Tutor/outputs/metrics.csv`
- `AI Tutor/outputs/final_eval.json`
- `AI Tutor/outputs/evaluation/final_test_metrics.csv`
- `AI Tutor/outputs/evaluation/akt_confusion_matrix.png`
- `AI Tutor/outputs/evaluation/akt_roc_curve.png`
- `AI Tutor/outputs/evaluation/akt_pr_curve.png`
- `AI Tutor/outputs/evaluation/model_metric_comparison.png`

The enhanced evaluation script can save prediction-level arrays in `akt_test_predictions.npz`, which allows ROC/PR curves and confusion matrix generation without re-running the whole evaluation.

## 15. About DKT Comparison

No final DKT training code, checkpoint, or metrics were found in `AI Tutor`. The final evaluation report correctly does not invent DKT results.

Say this clearly:

> The final implemented and evaluated student model is AKT. DKT was part of the earlier concept/frontend wording, but there is no compatible final DKT baseline checkpoint in this codebase. If a strict AKT-vs-DKT comparison is required, we would add a DKT model and train it on the same splits.

## 16. RAG Pipeline

The RAG layer is for grounding tutor responses in educational material.

It supports:

- `.txt`
- `.md`
- `.csv`
- `.json`
- `.jsonl`
- `.pdf`
- `.docx`
- `.pptx`
- `.svg`

Steps:

1. Load documents from `rag_docs_dir`.
2. Extract text depending on file type.
3. Normalize whitespace.
4. Split text into chunks.
5. Default chunk size: `900` characters.
6. Default overlap: `150` characters.
7. Embed chunks using SentenceTransformers if available.
8. Fall back to HashingVectorizer if embeddings are unavailable.
9. Store chunks in Chroma or TurboVec.
10. Retrieve top-k chunks for a student question.

Final config uses:

- Backend: `turbovec`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Hashing fallback: `1024` features
- TurboVec bits: `3`
- `top_k = 4`

## 17. Socratic Tutor Layer

The tutor receives:

- student question
- weak skill
- mastery probability
- retrieved chunks
- optional final answer for leakage checking

It builds a prompt that instructs the tutor to:

- use retrieved context when relevant
- avoid giving the final answer
- give a hint, not a full solution
- ask exactly one guiding question
- adapt to mastery level
- optionally recommend next practice

Generation modes:

- OpenAI API mode
- Local HuggingFace mode
- TurboQuant mode
- Template fallback mode

The actual tutor logs show template fallback examples, so do not overclaim that the final measured tutor quality depends on a fully deployed LLM. The final measured core is the AKT student model; the tutor layer is a prototype around it.

## 18. Answer Leakage And Faithfulness

The tutor has rule-based safety checks.

Leakage detection flags phrases such as:

- "the answer is"
- "final answer"
- "solution is"
- "so the answer"
- "therefore x ="

If leakage is detected, the system tries a stricter prompt. If that still leaks, it repairs the response by removing direct-answer sentences and replacing weak responses with a generic hint-only message.

Faithfulness is checked using token overlap between the tutor response and retrieved chunks. This is a simple grounding proxy, not a deep factuality evaluator.

## 19. Agentic RAG

The `AgenticRAGTutor` adds deterministic planning before generation.

It does three planning steps:

1. Analyze student state:
   - weak skill
   - mastery probability
   - recent mistakes
   - learning trend

2. Decide retrieval:
   - low mastery gets prerequisite, beginner, worked-example style queries
   - medium mastery gets conceptual explanation, scaffolded example, practice problem
   - high mastery gets challenge practice and misconception checks

3. Plan response:
   - low mastery or step-by-step learners get scaffolded mode
   - medium mastery gets Socratic mode
   - high mastery gets hint-only mode
   - negative trend can trigger motivational hint mode

This is "agentic" because retrieval and response planning are separate state-aware decisions, not just one retrieve-and-answer call.

## 20. Student Twin

The student twin is a persistent profile saved under `outputs/student_twins`.

It tracks:

- interactions
- confidence
- hint dependence
- difficulty tolerance
- speed of mastery
- whether the student needs step-by-step help
- preferred explanation style
- weak topics
- common mistakes
- mastery history
- response mode history

The twin update is heuristic, not learned. It uses exponential moving averages to update confidence, weakness, hint dependence, and style. This is appropriate to describe as a reproducible personalization baseline, not as a fully learned psychological model.

## 21. Explainability

The main explanation function runs AKT with `return_attention=True`.

For a batch, it:

1. Computes probabilities for all valid target positions.
2. Picks the weakest target position, meaning the lowest predicted mastery/correctness probability.
3. Converts question and skill IDs back to readable vocabulary keys.
4. Stores recent interactions.
5. Looks at the last attention layer.
6. Averages attention across heads.
7. Saves the top attended previous interactions.

The explanation JSON includes:

- student ID
- weak skill
- weak question
- mastery probability
- recent interactions
- top attended interactions
- a note that attention is not causal proof

Good defense phrasing:

> Attention gives an interpretable signal about which past interactions the model focused on, but it should not be treated as a causal explanation. It is an explanation hook, not proof of reasoning.

## 22. ONNX Deployment

The trained AKT model is exported to ONNX through `src/deployment/export_onnx.py`.

Export inputs:

- `question_ids`
- `skill_ids`
- `correct_ids`
- `target_question_ids`
- `target_skill_ids`
- `input_mask`

Output:

- `logits`

The export uses opset 17 and dynamic batch axis. After export, the code can compare ONNX Runtime output against PyTorch output using max absolute difference.

The final artifact manifest reports:

- GPU: `NVIDIA GeForce RTX 4060`
- ONNX model: `AI Tutor/outputs/onnx/akt_model.onnx`

Efficiency benchmark examples:

- PyTorch CUDA fp16/mixed: around `0.95 ms` sequence-level inference
- ONNX Runtime CPU fp32: around `1.24 ms`
- Research efficiency file also reports ONNX CPU around `1.32 ms`

These numbers are for synthetic sequence-level inference, not full tutor response latency.

## 23. Frontend Prototype

The frontend is a browser demo console:

- chat area
- model selector
- mastery bars
- student twin controls
- grounding panel
- prompt payload preview

Important caveat:

- The public labels now use AKT wording.
- `frontend/artifacts` appears empty.
- The JavaScript uses `fakeRetrieve`.
- Browser ONNX input names are DKT-style (`x`, `skill`, `mask`), while the AKT ONNX export expects six AKT inputs.

So the frontend should be presented as a prototype UI mock/demo, not as the final integrated AKT production frontend.

## 24. Strengths

- The core KT task is implemented end to end.
- Correctness labels are derived properly from EdNet metadata.
- The model uses causal attention, masking, validation, checkpointing, and test evaluation.
- Metrics go beyond accuracy: AUC, PR-AUC, precision, recall, F1, confusion matrix.
- The dataset scale is large.
- The system has deployment support through ONNX.
- The tutor layer adds retrieval, leakage checks, personalization state, and explanation artifacts.

## 25. Limitations To Admit

Be honest about these:

- No final DKT baseline is implemented or evaluated.
- The frontend is not fully wired to the AKT ONNX input format.
- KT3 and KT4 final sequence counts are identical, and the parser ignores KT4-specific extra fields such as `cursor_time`.
- The same EdNet user may appear across different KT levels. The current split is sequence-level across merged sources, not a strict global raw-user split across KT1/KT2/KT3/KT4.
- Source-prefixing prevents ID collisions but treats the same question in different sources as different items.
- Only the first skill tag is used.
- Rich features like elapsed time, platform, source, bundle, and cursor behavior are not used by AKT.
- RAG grounding uses simple embedding retrieval and overlap-based faithfulness checks.
- Student twin updates are heuristic.
- External paper comparisons are not directly comparable because datasets, splits, preprocessing, and model variants differ.

## 26. Likely Discussion Questions And Answers

Q: What is the main model?

A: An AKT-style attention-based knowledge tracing model. It predicts the probability that a learner will answer the next target question correctly, based on previous question, skill, and correctness interactions.

Q: How did you get labels from EdNet?

A: EdNet logs store the student's answer. I used `questions.csv` to map each question to the official `correct_answer`, then marked a response correct if the lowercased student answer matched the lowercased correct answer.

Q: Why not just use `user_answer` directly?

A: `user_answer` is the selected option, not a correctness label. Without comparing it against metadata, the model would not know if option `a`, `b`, `c`, or `d` was correct.

Q: What does the model see at each time step?

A: It sees the past question ID, past skill ID, and whether the student got that past item correct. It predicts correctness for the next target question and skill.

Q: Why use a mask?

A: Sequences have different lengths, so shorter sequences are padded to length 200. The mask prevents padding from affecting loss, metrics, and attention.

Q: Why use AUC?

A: Knowledge tracing is a binary prediction problem, and AUC measures how well the model ranks correct responses above incorrect responses across thresholds. Accuracy depends on one threshold, usually 0.5.

Q: What does AUC 0.787 mean?

A: It means that, across a random correct and incorrect response, the model gives the correct response a higher probability about 78.7 percent of the time. It is clearly above random 0.5, but not perfect.

Q: Why is recall higher than precision?

A: At threshold 0.5, the model is more willing to predict "correct", so it catches many true correct answers but also creates false positives.

Q: What is the loss?

A: Binary cross entropy with logits, averaged only over real unpadded sequence positions.

Q: Why AKT instead of DKT?

A: AKT uses attention, which can focus on relevant past interactions and supports explanation through attention maps. DKT is an RNN/LSTM-style baseline, but a final DKT implementation was not included in this codebase.

Q: What does RAG add?

A: RAG retrieves relevant educational chunks so the tutor response can be grounded in source material rather than generated from nothing.

Q: What does the student twin add?

A: It stores persistent learner state such as weak topics, confidence, hint dependence, and preferred explanation style, so tutoring can adapt over turns.

Q: Is attention a complete explanation?

A: No. Attention shows what the model attended to, but it is not causal proof. It is a practical explanation signal for inspection.

Q: Can this run in deployment?

A: The AKT model can be exported to ONNX and run with ONNX Runtime. The tutor/RAG layer is also modular, but the current frontend is still a prototype.

Q: What is the biggest improvement you would make next?

A: Add a true DKT baseline on the exact same split, enforce global raw-student splits across KT levels, use all skill tags and time/cursor features, and fully wire the frontend to the AKT ONNX inputs.

## 27. One-Minute Defense Summary

This project builds an AI tutoring pipeline around knowledge tracing. The core model is an AKT-style attention model trained on multi-source EdNet logs. Raw student answers are converted into correctness labels by comparing them with EdNet's official `correct_answer` metadata. The data is grouped into chronological student sequences, split into train/validation/test sets, windowed to length 200, padded, masked, and used to predict next-response correctness. The AKT model embeds questions, skills, correctness, and positions, applies causal multi-head self-attention, and predicts a probability for the next target question. The final EdNet run evaluated over 32.27 million test tokens and achieved AUC 0.7874, accuracy 0.7213, F1 0.7774, and PR-AUC 0.8381. Around the student model, the project adds RAG retrieval, Socratic hint generation, answer-leakage checks, student-twin personalization, attention-based explanation artifacts, and ONNX export for deployment. The main limitation is that the final implementation is AKT-only, while DKT remains an earlier concept/frontend label rather than a trained final baseline.
