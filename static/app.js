const EXAMPLE_QUESTIONS = [
  "What are the two main components of the Transformer architecture?",
  "How many layers does the base Transformer model use in both encoder and decoder?",
  "What is the formula for Scaled Dot-Product Attention?",
  "How many attention heads does the Transformer use, and what is the dimension of each head?",
];

const els = {
  chat: document.getElementById("chat"),
  askForm: document.getElementById("askForm"),
  questionInput: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  demoBtn: document.getElementById("demoBtn"),
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
  statusMessage: document.getElementById("statusMessage"),
  progressFill: document.getElementById("progressFill"),
  docName: document.getElementById("docName"),
  examples: document.getElementById("examples"),
};

let ready = false;
let ingesting = false;
let pollTimer = null;

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function appendMessage(role, content, sources = []) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    details.appendChild(summary);

    sources.forEach((source) => {
      const item = document.createElement("p");
      item.className = "source-item";
      const flags = [];
      if (source.has_tables) flags.push("tables");
      if (source.has_images) flags.push("images");
      const meta = flags.length ? ` · ${flags.join(", ")}` : "";
      item.innerHTML = `<span class="source-meta">Chunk ${escapeHtml(
        source.index
      )}${escapeHtml(meta)}</span>${escapeHtml(source.preview)}`;
      details.appendChild(item);
    });

    wrap.appendChild(details);
  }

  els.chat.appendChild(wrap);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function setComposerEnabled(enabled) {
  els.questionInput.disabled = !enabled;
  els.sendBtn.disabled = !enabled;
}

function updateStatusUI(data) {
  const state = data.state || "idle";
  ready = Boolean(data.ready);
  ingesting = ["parsing", "summarising", "embedding"].includes(state);

  els.statusDot.dataset.state = state;
  els.statusLabel.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  els.statusMessage.textContent = data.error
    ? data.error
    : data.message || "—";
  els.progressFill.style.width = `${Math.round((data.progress || 0) * 100)}%`;
  els.docName.textContent = data.document_name || "—";

  els.demoBtn.disabled = ingesting;
  els.dropzone.style.pointerEvents = ingesting ? "none" : "auto";
  els.dropzone.style.opacity = ingesting ? "0.6" : "1";
  setComposerEnabled(ready && !ingesting);
}

async function fetchStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("Could not load status");
  return res.json();
}

async function pollStatusUntilSettled() {
  if (pollTimer) clearInterval(pollTimer);

  const tick = async () => {
    try {
      const data = await fetchStatus();
      updateStatusUI(data);
      if (data.state === "ready" || data.state === "error" || data.state === "idle") {
        if (data.state === "ready" || data.state === "error") {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        if (data.state === "ready") {
          appendMessage(
            "system",
            `Document ready: ${data.document_name || "indexed PDF"}. Ask a question below.`
          );
        }
        if (data.state === "error") {
          appendMessage("system", `Ingest failed: ${data.error || "unknown error"}`);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  await tick();
  pollTimer = setInterval(tick, 2000);
}

async function startDemoIngest() {
  els.demoBtn.disabled = true;
  appendMessage("system", "Indexing demo PDF (Attention Is All You Need). This can take several minutes…");
  const res = await fetch("/api/ingest/demo", { method: "POST" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    appendMessage("system", data.detail || "Could not start demo ingest.");
    els.demoBtn.disabled = false;
    return;
  }
  await pollStatusUntilSettled();
}

async function startUploadIngest(file) {
  if (!file) return;
  appendMessage("system", `Uploading ${file.name}…`);
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/ingest", { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    appendMessage("system", data.detail || "Upload failed.");
    return;
  }
  appendMessage("system", "Ingest started. Progress updates on the left.");
  await pollStatusUntilSettled();
}

async function askQuestion(question) {
  if (!question.trim()) return;
  appendMessage("user", question.trim());
  els.questionInput.value = "";
  setComposerEnabled(false);
  appendMessage("system", "Thinking…");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim() }),
    });
    const data = await res.json().catch(() => ({}));

    // Remove trailing "Thinking…" system message
    const last = els.chat.lastElementChild;
    if (last && last.classList.contains("system")) {
      last.remove();
    }

    if (!res.ok) {
      appendMessage("system", data.detail || "Ask failed.");
      return;
    }
    appendMessage("assistant", data.answer || "No answer returned.", data.sources || []);
  } catch (err) {
    appendMessage("system", err.message || "Network error while asking.");
  } finally {
    const status = await fetchStatus().catch(() => null);
    if (status) updateStatusUI(status);
    else setComposerEnabled(ready && !ingesting);
  }
}

function renderExamples() {
  EXAMPLE_QUESTIONS.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      els.questionInput.value = q;
      els.questionInput.focus();
    });
    els.examples.appendChild(btn);
  });
}

function wireEvents() {
  els.demoBtn.addEventListener("click", () => {
    startDemoIngest().catch((err) => appendMessage("system", err.message));
  });

  els.askForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(els.questionInput.value).catch((err) =>
      appendMessage("system", err.message)
    );
  });

  els.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.askForm.requestSubmit();
    }
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    startUploadIngest(file).finally(() => {
      els.fileInput.value = "";
    });
  });

  ["dragenter", "dragover"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    startUploadIngest(file);
  });
}

async function init() {
  renderExamples();
  wireEvents();
  setComposerEnabled(false);

  try {
    const health = await fetch("/api/health").then((r) => r.json());
    if (!health.has_api_key) {
      appendMessage(
        "system",
        "Server is missing OLLAMA_API_KEY. Set it in .env locally or as a Hugging Face Space secret."
      );
    }
  } catch (_) {
    appendMessage("system", "Could not reach the API. Is the server running?");
  }

  try {
    const status = await fetchStatus();
    updateStatusUI(status);
    if (status.ready) {
      appendMessage(
        "system",
        `Ready with ${status.document_name || "an indexed PDF"}. Try an example question.`
      );
    } else if (["parsing", "summarising", "embedding"].includes(status.state)) {
      appendMessage("system", "Ingest already in progress…");
      await pollStatusUntilSettled();
    } else {
      appendMessage(
        "system",
        "Start with “Use demo PDF” or upload your own document, then ask a question."
      );
    }
  } catch (_) {
    appendMessage("system", "Status check failed.");
  }
}

init();
