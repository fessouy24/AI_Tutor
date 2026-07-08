const state = {
  config: null,
  session: null,
  skills: ["linear equations", "fractions", "geometry", "probability", "word problems"],
  mastery: {},
  history: [],
  ortSession: null,
};

const $ = (id) => document.getElementById(id);

function clamp(v, lo = 0, hi = 1) {
  return Math.min(hi, Math.max(lo, v));
}

function fmt(v) {
  return Number.isFinite(v) ? v.toFixed(3) : "--";
}

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.innerHTML = `<small>${role === "student" ? "Student" : "Tutor"}</small>${text}`;
  $("chatLog").appendChild(el);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

async function loadJSON(path, fallback) {
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(res.statusText);
    return await res.json();
  } catch {
    return fallback;
  }
}

async function init() {
  state.config = await loadJSON("artifacts/model_config.json", {
    models: { rag_conditioned: "artifacts/akt_model.onnx", baseline: "artifacts/baseline_akt.onnx" },
    metrics: { baseline: {}, rag_conditioned: {}, n_skills: 5 },
    sample_skills: state.skills,
    max_seq_len: 200,
  });
  state.skills = (state.config.sample_skills || state.skills).slice(0, 8).map(String);
  state.skills.forEach((s, i) => (state.mastery[s] = 0.35 + i * 0.05));
  renderMetrics();
  renderMastery();
  await loadModel();
  addMessage("tutor", "Ready. Ask about a topic and I will choose a question using the current student model, retrieval evidence, and the Twin profile.");
}

async function loadModel() {
  const key = $("modelSelect").value;
  const path = state.config.models?.[key];
  $("artifactPath").textContent = path || "No ONNX path configured";
  try {
    if (!window.ort || !path) throw new Error("ONNX Runtime or model path missing");
    state.ortSession = await ort.InferenceSession.create(path, { executionProviders: ["wasm"] });
    $("modelStatus").textContent = `Loaded ${key} ONNX model`;
  } catch (err) {
    state.ortSession = null;
    $("modelStatus").textContent = `Demo mode: ${err.message}`;
  }
}

function renderMetrics() {
  const key = $("modelSelect").value;
  const metrics = state.config?.metrics?.[key] || {};
  $("aucMetric").textContent = fmt(metrics.auc);
  $("accMetric").textContent = fmt(metrics.accuracy);
}

function renderMastery() {
  const box = $("masteryList");
  box.innerHTML = "";
  Object.entries(state.mastery)
    .sort((a, b) => a[1] - b[1])
    .forEach(([skill, value]) => {
      const row = document.createElement("div");
      row.className = "skill-row";
      row.innerHTML = `<span>${skill}</span><div class="bar"><i style="width:${Math.round(value * 100)}%"></i></div><b>${Math.round(value * 100)}%</b>`;
      box.appendChild(row);
    });
}

function fakeRetrieve(query) {
  const weak = Object.entries(state.mastery).sort((a, b) => a[1] - b[1])[0]?.[0] || "math";
  return [
    { source: "RAG corpus", score: 0.82, text: `Retrieved example related to ${weak}: isolate the unknown using inverse operations.` },
    { source: "MathDial", score: 0.74, text: "Socratic prompt pattern: ask the student to identify the first valid operation." },
    { source: "ScienceQA/ARC", score: 0.61, text: `General reasoning support for: ${query}` },
  ];
}

function renderEvidence(hits) {
  $("groundingList").innerHTML = hits
    .map((h) => `<div class="evidence"><strong>${h.source} &middot; ${h.score.toFixed(2)}</strong>${h.text}</div>`)
    .join("");
  const top = hits[0]?.score || 0;
  $("hallucinationStatus").textContent =
    top >= 0.55
      ? "Grounded response: enough retrieval evidence to ask a targeted question."
      : "Low evidence: tutor should ask a clarifying question instead of inventing content.";
  $("hallucinationStatus").style.color = top >= 0.55 ? "var(--good)" : "var(--bad)";
}

function buildPrompt(query, hits, targetSkill) {
  return {
    schema: "AI_TUTOR_AKT_RAG_TWIN_V2",
    student_state: {
      pace: $("paceSelect").value,
      frustration: Number($("frustrationInput").value),
      weak_skills: Object.entries(state.mastery)
        .sort((a, b) => a[1] - b[1])
        .slice(0, 3)
        .map(([skill, mastery]) => ({ skill, mastery: Number(mastery.toFixed(3)) })),
      recent_interactions: state.history.slice(-6),
    },
    retrieval_grounding: {
      query,
      evidence: hits,
      if_insufficient: "ask for clarification instead of fabricating",
    },
    pedagogical_control: {
      target_skill: targetSkill,
      output: "one Socratic tutoring question",
    },
  };
}

async function inferMastery() {
  if (!state.ortSession) return;
  const seqLen = state.config.max_seq_len || 200;
  const x = new BigInt64Array(seqLen);
  const skill = new BigInt64Array(seqLen);
  const mask = new Array(seqLen).fill(false);
  state.history.slice(-seqLen).forEach((h, i) => {
    const sid = BigInt((state.skills.indexOf(h.skill) + 1) || 1);
    skill[i] = sid;
    x[i] = sid * 2n + BigInt(h.correct ? 1 : 0);
    mask[i] = true;
  });
  const feeds = {
    x: new ort.Tensor("int64", x, [1, seqLen]),
    skill: new ort.Tensor("int64", skill, [1, seqLen]),
    mask: new ort.Tensor("bool", mask, [1, seqLen]),
  };
  try {
    await state.ortSession.run(feeds);
  } catch (err) {
    $("modelStatus").textContent = `ONNX loaded; browser inference skipped: ${err.message}`;
  }
}

async function tutorTurn(query) {
  addMessage("student", query);
  const targetSkill = Object.entries(state.mastery).sort((a, b) => a[1] - b[1])[0][0];
  const hits = fakeRetrieve(query);
  const prompt = buildPrompt(query, hits, targetSkill);
  $("promptPreview").textContent = JSON.stringify(prompt, null, 2);
  renderEvidence(hits);

  const correct = Math.random() < clamp(state.mastery[targetSkill] + 0.1);
  state.history.push({ skill: targetSkill, correct });
  state.mastery[targetSkill] = clamp(state.mastery[targetSkill] + (correct ? 0.05 : -0.03));
  await inferMastery();
  renderMastery();

  const question = `Let's focus on <b>${targetSkill}</b>. ${hits[0].score >= 0.55 ? "Using the retrieved example," : "Before we go further,"} what would be your first step, and why?`;
  addMessage("tutor", question);
}

$("queryForm").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = $("studentQuery").value.trim();
  if (q) tutorTurn(q);
});

$("runBtn").addEventListener("click", () => tutorTurn($("studentQuery").value.trim()));
$("modelSelect").addEventListener("change", async () => {
  renderMetrics();
  await loadModel();
});
$("resetBtn").addEventListener("click", () => {
  state.history = [];
  state.skills.forEach((s, i) => (state.mastery[s] = 0.35 + i * 0.05));
  $("chatLog").innerHTML = "";
  renderMastery();
  addMessage("tutor", "Session reset.");
});

init();
