// ===========================================================
// The Verification Desk — frontend logic
// ===========================================================

const API_BASE = ""; // same origin (Flask serves both API + frontend)

const els = {
  textarea: document.getElementById("articleInput"),
  charCount: document.getElementById("charCount"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  errorMsg: document.getElementById("errorMsg"),
  placeholder: document.getElementById("placeholder"),
  verdictContent: document.getElementById("verdictContent"),
  stampEl: document.getElementById("stampEl"),
  fillFake: document.getElementById("fillFake"),
  fillReal: document.getElementById("fillReal"),
  probFake: document.getElementById("probFake"),
  probReal: document.getElementById("probReal"),
  signalsList: document.getElementById("signalsList"),
  termsList: document.getElementById("termsList"),
  historyList: document.getElementById("historyList"),
  statGrid: document.getElementById("statGrid"),
  tickerTrack: document.getElementById("tickerTrack"),
  dateline: document.getElementById("dateline"),
};

const SAMPLES = {
  real: "The Ministry of Health confirmed on Wednesday that vaccination coverage reached 78% nationwide, according to figures released at a press briefing in Geneva. Officials said the data had been independently verified by the World Health Organization.",
  fake: "SHOCKING: Anonymous insiders reveal the government is secretly BANNING a miracle cure and mainstream media refuses to report it!!! Share this before it gets DELETED forever!!!",
};

const TICKER_ITEMS = [
  "WIRE: Newsroom AI flags sensational phrasing in real time",
  "DESK NOTE: Attribution and specificity remain the strongest signals of credible reporting",
  "REMINDER: A confident model is not a fact-checker — verify against primary sources",
  "STYLE GUIDE: Exclamation marks, ALL CAPS, and vague sourcing correlate with low-credibility content",
  "TIP: Paste a full paragraph, not just a headline, for a more reliable read",
];

function initTicker() {
  const items = TICKER_ITEMS.map(t => `<span>${t}</span>`).join("<span>&nbsp;&nbsp;·&nbsp;&nbsp;</span>");
  els.tickerTrack.innerHTML = items + items; // duplicate for seamless loop
}

function initDateline() {
  const now = new Date();
  const opts = { weekday: "long", year: "numeric", month: "long", day: "numeric" };
  els.dateline.textContent = `Desk Edition · ${now.toLocaleDateString(undefined, opts)}`;
}

function updateCharCount() {
  const n = els.textarea.value.length;
  els.charCount.textContent = `${n} character${n === 1 ? "" : "s"}`;
}

function showError(msg) {
  els.errorMsg.textContent = msg;
  els.errorMsg.hidden = false;
}
function hideError() {
  els.errorMsg.hidden = true;
}

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();
    const items = els.statGrid.querySelectorAll(".statplate__item .statplate__num");
    items[0].textContent = `${data.accuracy}%`;
    items[1].textContent = `${data.precision}%`;
    items[2].textContent = `${data.recall}%`;
    items[3].textContent = `${data.f1}%`;
  } catch (e) {
    console.error("Failed to load stats", e);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    const data = await res.json();
    renderHistory(data);
  } catch (e) {
    console.error("Failed to load history", e);
  }
}

function renderHistory(items) {
  if (!items || items.length === 0) {
    els.historyList.innerHTML = `<li class="history__empty">Nothing reviewed yet — run your first verification.</li>`;
    return;
  }
  els.historyList.innerHTML = items.map(item => `
    <li class="history__item">
      <span class="history__badge history__badge--${item.label.toLowerCase()}">${item.label}</span>
      <span class="history__text">${escapeHtml(item.preview)}</span>
    </li>
  `).join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderResult(result) {
  els.placeholder.hidden = true;
  els.verdictContent.hidden = false;

  // Stamp
  els.stampEl.textContent = result.label === "REAL" ? "VERIFIED" : "FLAGGED";
  els.stampEl.className = result.label === "REAL" ? "stamp is-real" : "stamp is-fake";
  // restart animation
  void els.stampEl.offsetWidth;
  els.stampEl.style.animation = "none";
  void els.stampEl.offsetWidth;
  els.stampEl.style.animation = "";

  // Confidence bars
  els.fillFake.style.width = `${result.probabilities.FAKE}%`;
  els.fillReal.style.width = `${result.probabilities.REAL}%`;
  els.probFake.textContent = `${result.probabilities.FAKE}%`;
  els.probReal.textContent = `${result.probabilities.REAL}%`;

  // Stylistic signals
  const maxima = { "Exclamation marks": 6, "Question marks": 4, "ALL-CAPS word ratio": 1, "Sensational phrase hits": 5, "Average word length": 8, "Text length (chars)": 400 };
  els.signalsList.innerHTML = result.stylistic_signals.map(sig => {
    const max = maxima[sig.name] || 10;
    const pct = Math.min(100, (Number(sig.value) / max) * 100);
    const displayVal = sig.name === "ALL-CAPS word ratio" ? `${(sig.value * 100).toFixed(0)}%` : sig.value;
    return `
      <li class="evidence__item">
        <span>${sig.name}</span>
        <span class="evidence__value">${displayVal}</span>
        <div class="evidence__meter"><div class="evidence__meter-fill" style="width:${pct}%"></div></div>
      </li>
    `;
  }).join("");

  // Top contributing terms
  if (result.top_terms && result.top_terms.length > 0) {
    els.termsList.innerHTML = result.top_terms.map(t => `
      <span class="term-tag term-tag--${t.direction.toLowerCase()}">${escapeHtml(t.term)}</span>
    `).join("");
  } else {
    els.termsList.innerHTML = `<span class="term-tag" style="color:var(--muted); border-color: rgba(236,230,216,0.2);">No strong keyword signals — decision driven mostly by writing style</span>`;
  }
}

async function analyze() {
  const text = els.textarea.value.trim();
  hideError();

  if (!text) {
    showError("Please paste some article text first.");
    return;
  }
  if (text.length < 15) {
    showError("Please provide at least a full sentence (15+ characters) for a reliable result.");
    return;
  }

  els.analyzeBtn.disabled = true;
  els.analyzeBtn.querySelector(".btn__label").textContent = "READING...";

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Something went wrong. Please try again.");
      return;
    }
    renderResult(data);
    fetchHistory();
  } catch (e) {
    console.error(e);
    showError("Could not reach the verification backend. Is the Flask server running?");
  } finally {
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.querySelector(".btn__label").textContent = "RUN VERIFICATION";
  }
}

function loadSample(kind) {
  els.textarea.value = SAMPLES[kind];
  updateCharCount();
  hideError();
  els.textarea.focus();
}

function clearSheet() {
  els.textarea.value = "";
  updateCharCount();
  hideError();
  els.placeholder.hidden = false;
  els.verdictContent.hidden = true;
}

// ---------------- Event wiring ----------------
els.textarea.addEventListener("input", updateCharCount);
els.analyzeBtn.addEventListener("click", analyze);
document.querySelectorAll(".tab[data-sample]").forEach(btn => {
  btn.addEventListener("click", () => loadSample(btn.dataset.sample));
});
document.getElementById("clearBtn").addEventListener("click", clearSheet);

els.textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    analyze();
  }
});

// ---------------- Init ----------------
initTicker();
initDateline();
updateCharCount();
fetchStats();
fetchHistory();
