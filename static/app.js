const transcriptEl = document.getElementById("transcript");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const micLabel = document.getElementById("micLabel");
const resetBtn = document.getElementById("resetBtn");
const statusLine = document.getElementById("statusLine");
const stateChips = document.getElementById("stateChips");
const voiceHint = document.getElementById("voiceHint");
const aiBanner = document.getElementById("aiBanner");

let sessionId = null;
let recognition = null;
let listening = false;

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

function renderAiBanner(aiStatus) {
  if (!aiBanner) return;
  if (!aiStatus || !aiStatus.enabled) {
    aiBanner.classList.add("hidden");
    aiBanner.textContent = "";
    return;
  }
  if (aiStatus.quota_exceeded) {
    aiBanner.classList.remove("hidden");
    aiBanner.textContent =
      `AI assist is temporarily unavailable (${aiStatus.provider || "model"} quota exceeded). ` +
      "Running in rule-based fallback mode.";
    return;
  }
  if (aiStatus.last_ok === false && aiStatus.last_error_code) {
    aiBanner.classList.remove("hidden");
    aiBanner.textContent =
      `AI assist had an error (${aiStatus.provider || "provider"} ${aiStatus.last_error_code}). ` +
      "Using fallback logic for now.";
    return;
  }
  aiBanner.classList.add("hidden");
  aiBanner.textContent = "";
}

function appendBubble(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = role === "user" ? "Caller" : "Assistant";
  const body = document.createElement("div");
  body.textContent = text;
  wrap.append(label, body);
  transcriptEl.appendChild(wrap);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderChips(state) {
  stateChips.innerHTML = "";
  const pairs = [
    ["Intent", state.intent],
    ["Phase", state.phase],
    ["Patient", state.patient_name],
    ["Date", state.appointment_date],
    ["Time", state.appointment_time],
    ["Prior date", state.old_appointment_date],
  ];
  for (const [k, v] of pairs) {
    if (!v) continue;
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = `${k}: ${v}`;
    stateChips.appendChild(c);
  }
}

async function ensureSession() {
  if (sessionId) return sessionId;
  const res = await fetch("/api/session", { method: "POST" });
  if (!res.ok) throw new Error("Could not start session");
  const data = await res.json();
  sessionId = data.session_id;
  return sessionId;
}

async function sendMessage(message) {
  const trimmed = message.trim();
  if (!trimmed) return;
  await ensureSession();
  appendBubble("user", trimmed);
  textInput.value = "";
  setStatus("Thinking…");
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: trimmed }),
  });
  if (!res.ok) {
    setStatus("Something went wrong. Try again.", true);
    return;
  }
  const data = await res.json();
  appendBubble("assistant", data.reply);
  renderChips(data);
  renderAiBanner(data.ai_status);
  setStatus("Ready");
  speak(data.reply);
}

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1;
  const voices = window.speechSynthesis.getVoices();
  const preferred =
    voices.find((v) => /English.*United States/i.test(v.name) && v.lang.startsWith("en")) ||
    voices.find((v) => v.lang.startsWith("en"));
  if (preferred) u.voice = preferred;
  window.speechSynthesis.speak(u);
}

function setupSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    voiceHint.textContent =
      "Voice input is not supported in this browser. Use Chrome, Edge, or Safari, or type your message.";
    micBtn.disabled = true;
    micBtn.classList.add("ghost");
    return;
  }
  recognition = new SR();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (ev) => {
    const text = ev.results[0][0].transcript;
    listening = false;
    micBtn.classList.remove("listening");
    micBtn.setAttribute("aria-pressed", "false");
    micLabel.textContent = "Hold to talk";
    sendMessage(text);
  };

  recognition.onerror = (ev) => {
    listening = false;
    micBtn.classList.remove("listening");
    micLabel.textContent = "Hold to talk";
    if (ev.error === "no-speech") {
      setStatus("No speech detected");
      return;
    }
    if (ev.error === "not-allowed") {
      setStatus("Microphone permission denied", true);
      return;
    }
    setStatus(`Voice error: ${ev.error}`, true);
  };

  recognition.onend = () => {
    if (listening) {
      // restarted by browser; ignore
    }
    listening = false;
    micBtn.classList.remove("listening");
    micLabel.textContent = "Hold to talk";
  };
}

function startListen() {
  if (!recognition) return;
  try {
    listening = true;
    micBtn.classList.add("listening");
    micBtn.setAttribute("aria-pressed", "true");
    micLabel.textContent = "Listening…";
    setStatus("Listening…");
    recognition.start();
  } catch {
    setStatus("Tap again to start the microphone", true);
  }
}

function stopListen() {
  if (!recognition) return;
  try {
    recognition.stop();
  } catch {
    /* ignore */
  }
}

micBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (micBtn.disabled) return;
  startListen();
});

micBtn.addEventListener("pointerup", () => {
  stopListen();
});

micBtn.addEventListener("pointerleave", () => {
  if (listening) stopListen();
});

sendBtn.addEventListener("click", () => sendMessage(textInput.value));

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage(textInput.value);
});

resetBtn.addEventListener("click", async () => {
  if (!sessionId) {
    transcriptEl.innerHTML = "";
    stateChips.innerHTML = "";
    return;
  }
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: "" }),
  });
  transcriptEl.innerHTML = "";
  stateChips.innerHTML = "";
  appendBubble("assistant", "Session cleared. How can I help you today?");
  setStatus("Ready");
});

window.speechSynthesis?.addEventListener("voiceschanged", () => {
  window.speechSynthesis.getVoices();
});

(async function init() {
  setupSpeechRecognition();
  let llmAssist = false;
  let llmLabel = "";
  let aiStatus = null;
  try {
    const fr = await fetch("/api/features");
    if (fr.ok) {
      const feat = await fr.json();
      llmAssist = !!(feat.llm_configured ?? feat.openai_configured);
      if (feat.llm_provider && feat.llm_model) {
        llmLabel = `${feat.llm_provider} (${feat.llm_model})`;
      } else if (feat.llm_model) {
        llmLabel = String(feat.llm_model);
      }
      aiStatus = feat.llm_runtime || null;
    }
  } catch {
    /* ignore */
  }
  try {
    await ensureSession();
    renderAiBanner(aiStatus);
    setStatus(llmAssist ? `Ready · AI assist: ${llmLabel || "on"}` : "Ready");
    appendBubble(
      "assistant",
      "Hi — I can book a new visit, reschedule an existing one, or answer quick questions about hours and insurance. How can I help?"
    );
  } catch (err) {
    setStatus("Cannot reach the server. Run the app locally and refresh.", true);
    console.error(err);
  }
})();
