/* ══════════════════════════════════════════════════
   RAG Pipeline — 前端交互
   ══════════════════════════════════════════════════ */

"use strict";

/* ─── Constants ─── */
const API = "/api/v1";
const HISTORY_KEY = "rag_history";
const DOCS_KEY = "rag_documents";

/* ─── DOM refs ─── */
const $ = (s, p) => (p || document).querySelector(s);
const $$ = (s, p) => Array.from((p || document).querySelectorAll(s));

/* ═══════════════════════════════════════════════════
   Storage
   ═══════════════════════════════════════════════════ */

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}

function saveHistory(h) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 50)));
}

function addHistory(item) {
  const h = getHistory();
  h.unshift({ ...item, time: Date.now() });
  saveHistory(h);
}

function getDocuments() {
  try { return JSON.parse(localStorage.getItem(DOCS_KEY) || "[]"); } catch { return []; }
}

function saveDocuments(docs) {
  localStorage.setItem(DOCS_KEY, JSON.stringify(docs.slice(-50)));
}

function addDocument(doc) {
  const docs = getDocuments();
  const idx = docs.findIndex((d) => d.task_id === doc.task_id);
  if (idx >= 0) docs[idx] = { ...docs[idx], ...doc };
  else docs.unshift(doc);
  saveDocuments(docs);
}

function updateDocument(taskId, updates) {
  const docs = getDocuments();
  const idx = docs.findIndex((d) => d.task_id === taskId);
  if (idx >= 0) {
    docs[idx] = { ...docs[idx], ...updates };
    saveDocuments(docs);
  }
}

async function deleteDocument(taskId, name) {
  if (!confirm(`确定要删除「${name}」吗？\n文档文件、索引和切片数据都会被清除。`)) return;

  // Always remove from localStorage FIRST — even if the backend already
  // lost track of this task (e.g. from a pre-persistence session), the
  // stale entry should still go away.
  const docs = getDocuments().filter((d) => d.task_id !== taskId);
  saveDocuments(docs);
  renderDocuments();

  // Best-effort backend cleanup
  try {
    const res = await fetch(`${API}/documents/${taskId}`, { method: "DELETE" });
    if (!res.ok) {
      const detail = ((await res.json().catch(() => ({}))).detail) || "";
      console.warn("Backend delete warning:", detail);
    }
  } catch (err) {
    console.warn("Backend delete error:", err.message);
  }

  showToast("success", `「${name}」已删除`);
}

// ---------------------------------------------------------------------------
// Background task watcher — polls task status until completion, then auto-
// refreshes the review list.
// ---------------------------------------------------------------------------
const _watchedTasks = new Set();

async function _watchTask(taskId, fileName) {
  if (_watchedTasks.has(taskId)) return;
  _watchedTasks.add(taskId);

  for (let i = 0; i < 120; i++) {
    await sleep(3000);
    try {
      const res = await fetch(`${API}/documents/${taskId}/status`);
      if (!res.ok) continue;
      const data = await res.json();
      const status = data.status;

      if (status === "failed") {
        updateDocument(taskId, { status: "failed" });
        _watchedTasks.delete(taskId);
        return;
      }

      if (["indexed", "review"].includes(status)) {
        const needsReview = status === "review";
        updateDocument(taskId, {
          status: needsReview ? "review" : "indexed",
          pages: data.total_pages || 0,
          chunks: data.chunk_count || 0,
          indexed: data.indexed_count || 0,
        });
        updateBadges();
        if ($("#tab-review")?.classList.contains("active")) {
          loadReviews();
        }
        _watchedTasks.delete(taskId);
        return;
      }
    } catch (_) {
      /* network glitch — retry */
    }
  }
  // Timed out after ~6 min — give up
  _watchedTasks.delete(taskId);
}

/* ═══════════════════════════════════════════════════
   Navigation
   ═══════════════════════════════════════════════════ */

$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    $$(".tab-content").forEach((t) => t.classList.toggle("active", t.id === "tab-" + tab));
    if (tab === "review") loadReviews();
    if (tab === "documents") renderDocuments();
  });
});

/* ─── Keyboard shortcuts ─── */
document.addEventListener("keydown", (e) => {
  if (e.altKey) {
    const map = { 1: "workspace", 2: "documents", 3: "review" };
    const tab = map[e.key];
    if (tab) {
      e.preventDefault();
      $(`[data-tab="${tab}"]`)?.click();
    }
  }
});

/* ═══════════════════════════════════════════════════
   Health Check
   ═══════════════════════════════════════════════════ */

async function checkHealth() {
  const dot = $("#health-dot");
  const txt = $("#health-text");
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    dot.className = "health-dot " + (d.status === "ok" ? "ok" : "err");
    txt.textContent = d.status === "ok" ? "服务正常" : "异常";
  } catch {
    dot.className = "health-dot err";
    txt.textContent = "无法连接";
  }
}
checkHealth();
setInterval(checkHealth, 30000);

/* ═══════════════════════════════════════════════════
   Upload
   ═══════════════════════════════════════════════════ */

let uploadAborted = false;

const uploadZone = $("#upload-zone");
const fileInput = $("#file-input");

// Click to select
$("#select-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener("click", () => fileInput.click());

// Drag & drop
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
  fileInput.value = "";
});

// Cancel button
$("#up-cancel")?.addEventListener("click", () => {
  uploadAborted = true;
  hideProgress();
});

function showProgress(filename) {
  uploadZone.style.display = "none";
  const prog = $("#upload-progress");
  prog.style.display = "block";
  $("#up-filename").textContent = filename;
  setProgressStep("upload");
  setProgressBar(10);
  $("#up-status").textContent = "上传中…";
  uploadAborted = false;
}

function hideProgress() {
  const prog = $("#upload-progress");
  prog.style.display = "none";
  uploadZone.style.display = "block";
}

function setProgressBar(pct) {
  $("#up-bar-fill").style.width = Math.min(pct, 100) + "%";
}

function setProgressStep(step) {
  const steps = ["upload", "process", "index", "done"];
  $$(".up-step").forEach((el) => {
    const s = el.dataset.step;
    const idx = steps.indexOf(s);
    const cur = steps.indexOf(step);
    el.classList.toggle("done", idx < cur);
    el.classList.toggle("active", idx === cur);
  });
}

function setProgressStatus(text) {
  $("#up-status").textContent = text;
}

async function handleFile(file) {
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (![".pdf", ".docx", ".doc"].includes(ext)) {
    return showToast("error", `不支持 ${ext} 格式`);
  }

  showProgress(file.name);

  try {
    const fd = new FormData();
    fd.append("file", file);

    setProgressBar(15);
    setProgressStatus("上传中…");

    const res = await fetch(`${API}/documents/upload`, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || "上传失败");
    const data = await res.json();

    if (uploadAborted) return;

    setProgressStep("process");
    setProgressBar(40);
    setProgressStatus("正在解析文档…");

    // Poll status
    const taskId = data.task_id;
    let status = "queued";
    let pages = 0,
      chunks = 0,
      indexed = 0;

    for (let i = 0; i < 180; i++) {
      await sleep(1000);

      if (uploadAborted) return;

      const sr = await fetch(`${API}/documents/${taskId}/status`);
      if (!sr.ok) continue;

      const sd = await sr.json();
      status = sd.status;
      pages = sd.total_pages || 0;
      chunks = sd.chunk_count || 0;
      indexed = sd.indexed_count || 0;

      if (status === "failed") break;
      if (["indexed", "review"].includes(status)) {
        setProgressStep("index");
        setProgressBar(85);
        await sleep(300);
        break;
      }

      // Update progress
      const pct = 40 + Math.min(i * 0.25, 30);
      setProgressBar(pct);
      setProgressStep(i < 20 ? "process" : "index");
      setProgressStatus(`处理中 (${Math.round(i + 1)}s)…`);
    }

    if (!uploadAborted) {
      setProgressStep("done");
      setProgressBar(100);

      if (status === "failed") {
        setProgressStatus("处理失败");
        showToast("error", `${file.name} 处理失败`);
        hideProgress();
        addHistory({ name: file.name, status: "failed" });
        addDocument({ task_id: taskId, name: file.name, status: "failed", time: Date.now() });
      } else if (["indexed", "review"].includes(status)) {
        setProgressStatus("处理完成 ✓");
        const needsReview = status === "review";
        showToast(
          "success",
          `${file.name} 处理完成 — ${pages} 页, ${chunks} 切片, ${indexed} 已索引`
        );
        setTimeout(hideProgress, 1200);
        addHistory({ name: file.name, status: needsReview ? "review" : "ok" });
        addDocument({
          task_id: taskId,
          name: file.name,
          status: needsReview ? "review" : "indexed",
          pages,
          chunks,
          indexed,
          time: Date.now(),
        });
        updateBadges();
        if (needsReview && $("#tab-review")?.classList.contains("active")) {
          loadReviews();
        }
      } else {
        // Background processing still running after polling — watch in bg
        setProgressStatus("后台处理中…");
        setTimeout(hideProgress, 800);
        addDocument({ task_id: taskId, name: file.name, status: "processing", time: Date.now() });
        _watchTask(taskId, file.name);
      }
      renderHistory();
    }
  } catch (err) {
    if (!uploadAborted) {
      setProgressStatus("上传失败");
      showToast("error", err.message);
      addHistory({ name: file.name, status: "failed" });
      renderHistory();
      setTimeout(hideProgress, 1500);
    }
  }
}

function showToast(type, message) {
  const el = $("#upload-toast");
  el.style.display = "block";
  el.className = "upload-toast " + type;
  el.textContent = message;
  setTimeout(() => {
    el.style.display = "none";
  }, 4000);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ─── Upload History ─── */

function renderHistory() {
  const list = $("#upload-history-list");
  if (!list) return;
  const h = getHistory();

  if (!h.length) {
    list.innerHTML = '<div class="history-empty">暂无上传记录</div>';
    return;
  }

  list.innerHTML = h
    .map((item) => {
      const cls =
        item.status === "failed" ? "fail" : item.status === "review" ? "review" : "ok";
      const label =
        item.status === "failed"
          ? "失败"
          : item.status === "review"
            ? "待审核"
            : "完成";
      const time =
        typeof item.time === "number"
          ? new Date(item.time).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
          : "";
      return `<div class="history-item">
        <span class="h-name">${escHtml(item.name)}</span>
        <span class="h-status ${cls}">${label}</span>
        <span class="h-time">${time}</span>
      </div>`;
    })
    .join("");
}

$("#history-clear")?.addEventListener("click", () => {
  saveHistory([]);
  renderHistory();
});

renderHistory();

/* ═══════════════════════════════════════════════════
   Chat
   ═══════════════════════════════════════════════════ */

const chatInput = $("#chat-input");
const chatSend = $("#chat-send");
const chatMessages = $("#chat-messages");

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  chatSend.disabled = !chatInput.value.trim();
});

// Send on Enter (Shift+Enter for newline)
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (chatInput.value.trim()) sendMessage();
  }
});

chatSend.addEventListener("click", sendMessage);

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMsg("user", text);
  chatInput.value = "";
  chatInput.style.height = "auto";
  chatSend.disabled = true;

  const botMsgId = "msg-" + Date.now();
  const botMsgDiv = document.createElement("div");
  botMsgDiv.className = "msg msg-bot";
  botMsgDiv.id = botMsgId;
  botMsgDiv.innerHTML = `<div class="msg-avatar"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg></div><div class="msg-body"><div class="msg-sender">AI 助手</div><div class="msg-text streaming"><span class="streaming-cursor">▊</span></div></div>`;
  chatMessages.appendChild(botMsgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const textEl = botMsgDiv.querySelector(".msg-text");
  let fullAnswer = "";
  let finalMeta = null;

  try {
    const res = await fetch(`${API}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, top_k: 10 }),
    });

    if (!res.ok) throw new Error("请求失败 (" + res.status + ")");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split("\n");
      buf = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.type === "token") {
            fullAnswer += evt.content || "";
            textEl.textContent = fullAnswer + "▊";
            chatMessages.scrollTop = chatMessages.scrollHeight;
          } else if (evt.type === "meta") {
            finalMeta = evt;
          }
        } catch (_) { /* skip malformed events */ }
      }
    }

    // Remove cursor
    textEl.textContent = fullAnswer || "抱歉，无法获取回答。请稍后重试。";

    // Append citations & confidence from meta
    let parts = [];
    if (finalMeta) {
      let confidenceHtml = "";
      if (finalMeta.confidence !== undefined) {
        const pct = (finalMeta.confidence * 100).toFixed(0);
        const cls = finalMeta.confidence >= 0.75 ? "high" : finalMeta.confidence >= 0.4 ? "med" : "low";
        const warn = finalMeta.needs_review ? " ⚠️ 需人工审核" : "";
        confidenceHtml = `<div class="msg-confidence"><span class="confidence-dot ${cls}"></span>置信度 ${pct}%${warn}</div>`;
      }

      let citationsHtml = "";
      if (finalMeta.citations && finalMeta.citations.length) {
        citationsHtml =
          `<div class="msg-citations"><details><summary>📎 ${finalMeta.citations.length} 个引用来源</summary><div class="cite-list">` +
          finalMeta.citations
            .map(
              (c) =>
                `<div class="cite-item"><span class="cite-source">📄 ${escHtml(c.source_file)}</span> · 第${c.page_num}页${c.section ? " · §" + escHtml(c.section) : ""}<br><span style="font-size:11px;color:var(--text-muted)">${truncate(escHtml(c.text), 120)}</span></div>`
            )
            .join("") +
          `</div></details></div>`;
      }

      if (citationsHtml) parts.push(citationsHtml);
      if (confidenceHtml) parts.push(confidenceHtml);
    }

    if (parts.length) {
      const extraWrap = document.createElement("div");
      extraWrap.innerHTML = parts.join("");
      textEl.parentNode.appendChild(extraWrap);
    }
  } catch (err) {
    textEl.textContent = "❌ " + err.message;
  } finally {
    chatInput.focus();
  }
}

function addMsg(role, html, id) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (id) div.id = id;

  const avatarContent =
    role === "user"
      ? "我"
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`;

  div.innerHTML = `
    <div class="msg-avatar">${avatarContent}</div>
    <div class="msg-body">
      <div class="msg-sender">${role === "user" ? "你" : "AI 助手"}</div>
      <div class="msg-text">${html}</div>
    </div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTyping(id) {
  const div = document.createElement("div");
  div.className = "msg bot msg-typing";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg></div>
    <div class="msg-body">
      <div class="msg-sender">AI 助手</div>
      <div class="msg-text"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>
    </div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeMsg(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ═══════════════════════════════════════════════════
   Document Library
   ═══════════════════════════════════════════════════ */

function renderDocuments() {
  const grid = $("#doc-grid");
  if (!grid) return;
  const docs = getDocuments();

  if (!docs.length) {
    grid.innerHTML = `<div class="doc-empty"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>暂无文档，请先上传</p></div>`;
    return;
  }

  // Wire up delete button clicks (event delegation)
  grid.onclick = (e) => {
    const btn = e.target.closest(".btn-delete-doc");
    if (btn) {
      const tid = btn.dataset.taskId;
      const doc = docs.find((d) => d.task_id === tid);
      if (doc) deleteDocument(tid, doc.name || "未知文档");
    }
  };

  grid.innerHTML = docs
    .map((doc) => {
      const ext = doc.name ? doc.name.split(".").pop().toLowerCase() : "file";
      const iconCls = ext === "pdf" ? "pdf" : "docx";
      const statusCls = doc.status || "processing";
      const statusLabels = {
        indexed: "已索引",
        review: "待审核",
        failed: "失败",
        processing: "处理中",
      };
      const statusLabel = statusLabels[statusCls] || statusCls;
      const time =
        doc.time
          ? new Date(doc.time).toLocaleString("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
          : "";
      return `<div class="doc-card">
        <div class="doc-card-header">
          <div class="doc-card-icon ${iconCls}">${ext.toUpperCase()}</div>
          <span class="doc-card-name">${escHtml(doc.name || "未知文档")}</span>
          <button class="btn-icon btn-delete-doc" data-task-id="${doc.task_id}" title="删除文档">🗑</button>
        </div>
        <div class="doc-card-stats">
          <span class="doc-card-stat">📄 <strong>${doc.pages || "-"}</strong> 页</span>
          <span class="doc-card-stat">✂️ <strong>${doc.chunks || "-"}</strong> 切片</span>
          <span class="doc-card-stat">📌 <strong>${doc.indexed || "-"}</strong> 已索引</span>
        </div>
        <div class="doc-card-status">
          <span class="status-badge ${statusCls}">${statusLabel}</span>
          <span style="margin-left:auto;font-size:11px;color:var(--text-muted)">${time}</span>
        </div>
        <div style="margin-top:8px;display:flex;gap:6px">
          <button class="btn btn-sm" style="background:var(--primary-light);border:none;color:var(--primary)" onclick="showDocDetail('${doc.task_id}')">🔍 查看切片</button>
        </div>
      </div>`;
    })
    .join("");
}

/* ═══════════════════════════════════════════════════
   Document Detail — 模态弹窗查看切片
   ═══════════════════════════════════════════════════ */

async function showDocDetail(taskId) {
  const overlay = $("#doc-detail-overlay");
  if (!overlay) return;

  overlay.classList.add("open");
  overlay.dataset.taskId = taskId;
  overlay.innerHTML = '<div class="doc-detail-modal"><div class="doc-detail-header"><span class="doc-detail-title">加载中…</span><button class="btn-icon" onclick="closeDocDetail()" style="font-size:18px">✕</button></div><div class="doc-detail-body"><div style="text-align:center;padding:40px;color:var(--text-muted)">加载中…</div></div></div>';

  try {
    const res = await fetch(`${API}/review/${taskId}`);
    if (!res.ok) throw new Error("获取失败");
    const detail = await res.json();
    renderDocDetailModal(overlay, detail);
  } catch (e) {
    const body = overlay.querySelector(".doc-detail-body");
    if (body) body.innerHTML = `<div style="text-align:center;padding:40px;color:var(--danger)">加载失败: ${escHtml(e.message)}</div>`;
  }
}

function closeDocDetail() {
  const overlay = $("#doc-detail-overlay");
  if (overlay) overlay.classList.remove("open");
}

function renderDocDetailModal(overlay, detail) {
  const header = overlay.querySelector(".doc-detail-header");
  const body = overlay.querySelector(".doc-detail-body");
  if (!header || !body) return;

  // Title
  const title = header.querySelector(".doc-detail-title");
  if (title) title.textContent = `📄 ${detail.filename || detail.task_id}`;

  let html = '';

  // Confidence gauge
  if (detail.confidence !== undefined) {
    html += renderConfidenceGauge(detail.confidence, detail.confidence_details);
  }

  // Summary stats
  const blockCount = detail.pages?.reduce((s, p) => s + (p.blocks?.length || 0), 0) || 0;
  html += `<div style="display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--border)">
    <span>📄 ${detail.total_pages || 0} 页</span>
    <span>📋 ${blockCount} 个内容块</span>
    <span>✂️ ${detail.total_chunks || 0} 个切片</span>
    <span>📌 ${detail.indexed_count || 0} 已索引</span>
  </div>`;

  // Tabs
  html += `<div class="detail-tabs">
    <button class="detail-tab active" data-pane="md-blocks-${detail.task_id}" onclick="switchDocDetailTab('${detail.task_id}','blocks')">📋 内容块 (${blockCount})</button>
    <button class="detail-tab" data-pane="md-chunks-${detail.task_id}" onclick="switchDocDetailTab('${detail.task_id}','chunks')">✂️ 切片 (${detail.total_chunks || 0})</button>
  </div>`;

  // Blocks pane
  html += `<div class="detail-pane active" id="md-pane-blocks-${detail.task_id}"><div class="block-list">`;
  if (detail.pages && detail.pages.length) {
    for (const page of detail.pages) {
      if (page.blocks && page.blocks.length) {
        html += `<div style="font-size:12px;font-weight:600;color:var(--text-secondary);padding:4px 0">第 ${page.page_num} 页 (${page.text_length} 字)</div>`;
        for (const block of page.blocks) {
          html += renderBlockHtml(block);
        }
      }
    }
  } else {
    html += '<div style="color:var(--text-muted);padding:8px">暂无内容块</div>';
  }
  html += `</div></div>`;

  // Chunks pane
  html += `<div class="detail-pane" id="md-pane-chunks-${detail.task_id}"><div class="chunk-list">`;
  if (detail.chunks && detail.chunks.length) {
    for (const chunk of detail.chunks) {
      html += renderChunkHtml(chunk);
    }
  } else {
    html += '<div style="color:var(--text-muted);padding:8px">暂无切片</div>';
  }
  html += `</div></div>`;

  body.innerHTML = html;
}

function switchDocDetailTab(taskId, pane) {
  const overlay = $("#doc-detail-overlay");
  if (!overlay) return;
  overlay.querySelectorAll(".detail-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.pane === `md-${pane}-${taskId}`);
  });
  const blocksPane = $(`#md-pane-blocks-${taskId}`);
  const chunksPane = $(`#md-pane-chunks-${taskId}`);
  if (blocksPane) blocksPane.classList.toggle("active", pane === "blocks");
  if (chunksPane) chunksPane.classList.toggle("active", pane === "chunks");
}

// Keyboard shortcut: Escape to close modal
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDocDetail();
});

// Click overlay background to close
document.addEventListener("click", (e) => {
  const overlay = $("#doc-detail-overlay");
  if (overlay && overlay.classList.contains("open") && e.target === overlay) {
    closeDocDetail();
  }
});

/* ═══════════════════════════════════════════════════
   Review
   ═══════════════════════════════════════════════════ */

/* ─── Helpers ─── */

function _confidenceClass(val) {
  if (val >= 0.75) return "high";
  if (val >= 0.4) return "med";
  return "low";
}

function _pct(val) {
  return Math.round((val || 0) * 100);
}

function renderConfidenceGauge(confidence, details) {
  const cls = _confidenceClass(confidence);
  const dimsHtml = Object.entries(details || {})
    .map(([k, v]) => {
      const dc = _confidenceClass(v);
      return `<div class="conf-dim">
        <div class="conf-dim-header">
          <span>${k}</span>
          <span>${_pct(v)}%</span>
        </div>
        <div class="conf-dim-track">
          <div class="conf-dim-fill ${dc}" style="width:${_pct(v)}%"></div>
        </div>
      </div>`;
    })
    .join("");

  return `<div class="confidence-section">
    <div class="confidence-main">
      <div class="confidence-score ${cls}">${_pct(confidence)}%</div>
      <div class="confidence-bar-track">
        <div class="confidence-bar-fill ${cls}" style="width:${_pct(confidence)}%"></div>
      </div>
    </div>
    <div class="confidence-dims">${dimsHtml}</div>
  </div>`;
}

function renderBlockHtml(block) {
  return `<div class="block-item" data-block-id="${block.block_id}">
    <div class="block-item-header">
      <span class="block-item-type">${escHtml(block.block_type)}</span>
      <span class="block-item-conf">${_pct(block.confidence)}%</span>
    </div>
    <div class="block-item-content">${escHtml(block.content) || "<em style='color:var(--text-muted)'>空</em>"}</div>
    <div class="block-item-actions" style="display:none">
      <button class="btn btn-sm btn-primary" onclick="saveEditBlock('${block.block_id}')">保存</button>
      <button class="btn btn-sm" style="background:var(--border-light);border:none" onclick="cancelEditBlock('${block.block_id}')">取消</button>
    </div>
  </div>`;
}

function renderChunkHtml(chunk) {
  return `<div class="chunk-item">
    <div class="chunk-item-header">
      <span class="chunk-item-type">${escHtml(chunk.chunk_type)}</span>
      <span>📄 ${escHtml(chunk.source_file)} · 第${chunk.page_num}页${chunk.section ? " · §" + escHtml(chunk.section) : ""}</span>
    </div>
    <div class="chunk-item-content">${escHtml(chunk.content) || "<em style='color:var(--text-muted)'>空</em>"}</div>
  </div>`;
}

/* ─── State ─── */

// Holds pending block edits in memory until approval
const pendingEdits = {};

/* ─── Main ─── */

async function loadReviews() {
  const badge = $("#review-badge");
  try {
    const res = await fetch(`${API}/review/pending`);
    if (!res.ok) throw new Error("获取失败");

    const items = await res.json();
    const count = items.length;

    if (count) {
      badge.textContent = count;
      badge.style.display = "inline";
    } else {
      badge.style.display = "none";
    }

    const list = $("#review-list");
    const empty = $("#review-empty");

    if (!count) {
      empty.style.display = "flex";
      list.innerHTML = `<div class="review-empty" style="display:flex"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg><p>暂无待审核项</p></div>`;
      return;
    }

    empty.style.display = "none";
    list.innerHTML = items
      .map(
        (item) => {
          const iconCls = item.status === "failed" ? "fail" : "warn";
          const icon = item.status === "failed" ? "❌" : "⚠️";
          const confHtml = item.confidence !== undefined
            ? `<span class="r-meta">置信度 ${_pct(item.confidence)}%</span>`
            : "";
          return `<div class="review-card" id="review-${item.task_id}" onclick="toggleReviewDetail('${item.task_id}')">
            <div class="review-card-top">
              <div class="review-card-icon ${iconCls}">${icon}</div>
              <div class="review-card-body">
                <div class="r-filename">${escHtml(item.filename || item.task_id)}</div>
                <div style="display:flex;gap:10px;font-size:12px;color:var(--text-muted)">
                  <span>📄 ${item.total_pages || 0} 页</span>
                  <span>✂️ ${item.total_chunks || 0} 切片</span>
                  ${confHtml}
                </div>
                ${item.error ? `<div class="r-error">${escHtml(item.error)}</div>` : ""}
              </div>
              <div class="review-card-actions" onclick="event.stopPropagation()">
                <button class="btn btn-sm btn-primary" onclick="approveReview('${item.task_id}','approve')">✅ 通过</button>
                <button class="btn btn-sm" style="background:var(--danger-bg);color:#991b1b;border:none" onclick="approveReview('${item.task_id}','reject')">❌ 拒绝</button>
              </div>
            </div>
            <div class="review-detail-panel" id="detail-${item.task_id}" onclick="event.stopPropagation()"></div>
          </div>`;
        }
      )
      .join("");
  } catch {
    badge.style.display = "none";
  }
}

async function toggleReviewDetail(taskId) {
  const panel = $(`#detail-${taskId}`);
  if (!panel) return;

  if (panel.classList.contains("open")) {
    panel.classList.remove("open");
    return;
  }

  // Close other panels
  $$(".review-detail-panel.open").forEach((p) => p.classList.remove("open"));

  panel.classList.add("open");

  // Lazy load if empty
  if (!panel.dataset.loaded) {
    panel.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-muted)">加载中…</div>';
    try {
      const res = await fetch(`${API}/review/${taskId}`);
      if (!res.ok) throw new Error("获取失败");
      const detail = await res.json();
      panel.dataset.loaded = "1";
      renderDetailPanel(panel, detail, taskId);
    } catch (e) {
      panel.innerHTML = `<div style="text-align:center;padding:20px;color:var(--danger)">加载失败: ${escHtml(e.message)}</div>`;
    }
  }
}

function renderDetailPanel(panel, detail, taskId) {
  // Confidence
  let html = renderConfidenceGauge(detail.confidence, detail.confidence_details);

  // Tabs: Blocks | Chunks
  html += `<div class="detail-tabs">
    <button class="detail-tab active" data-pane="blocks-${taskId}" onclick="switchDetailTab('${taskId}','blocks')">📋 内容块 (${detail.pages?.reduce((s, p) => s + (p.blocks?.length || 0), 0) || 0})</button>
    <button class="detail-tab" data-pane="chunks-${taskId}" onclick="switchDetailTab('${taskId}','chunks')">✂️ 切片 (${detail.total_chunks || 0})</button>
  </div>`;

  // Blocks pane
  html += `<div class="detail-pane active" id="pane-blocks-${taskId}"><div class="block-list">`;
  if (detail.pages && detail.pages.length) {
    for (const page of detail.pages) {
      if (page.blocks && page.blocks.length) {
        html += `<div style="font-size:12px;font-weight:600;color:var(--text-secondary);padding:4px 0">第 ${page.page_num} 页 (${page.text_length} 字)</div>`;
        for (const block of page.blocks) {
          html += renderBlockHtml(block);
        }
      }
    }
  } else {
    html += '<div style="color:var(--text-muted);padding:8px">暂无内容块</div>';
  }
  html += `</div></div>`;

  // Chunks pane
  html += `<div class="detail-pane" id="pane-chunks-${taskId}"><div class="chunk-list">`;
  if (detail.chunks && detail.chunks.length) {
    for (const chunk of detail.chunks) {
      html += renderChunkHtml(chunk);
    }
  } else {
    html += '<div style="color:var(--text-muted);padding:8px">暂无切片</div>';
  }
  html += `</div></div>`;

  // Actions: approve/reject with reason
  html += `<div class="review-actions-panel">
    <input class="review-reason-input" id="reason-${taskId}" placeholder="审批备注（拒绝时必填）…">
    <button class="btn btn-primary btn-sm" onclick="approveReview('${taskId}','approve')">✅ 通过</button>
    <button class="btn btn-sm btn-reprocess" onclick="reprocessDocument('${taskId}')">🔄 重新处理</button>
  </div>`;

  panel.innerHTML = html;
}

function switchDetailTab(taskId, pane) {
  // Update tab buttons
  const panel = $(`#detail-${taskId}`);
  if (!panel) return;
  panel.querySelectorAll(".detail-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.pane === `${pane}-${taskId}`);
  });
  // Update panes
  const blocksPane = $(`#pane-blocks-${taskId}`);
  const chunksPane = $(`#pane-chunks-${taskId}`);
  if (blocksPane) blocksPane.classList.toggle("active", pane === "blocks");
  if (chunksPane) chunksPane.classList.toggle("active", pane === "chunks");
}

/* ─── Block Editing ─── */

function editBlock(blockId) {
  const item = $(`[data-block-id="${blockId}"]`);
  if (!item) return;

  const contentEl = item.querySelector(".block-item-content");
  const actionsEl = item.querySelector(".block-item-actions");
  if (!contentEl || !actionsEl) return;

  const currentText = pendingEdits[blockId] !== undefined ? pendingEdits[blockId] : contentEl.textContent;
  contentEl.innerHTML = `<textarea id="edit-${blockId}">${escHtml(currentText)}</textarea>`;
  item.classList.add("editing");
  actionsEl.style.display = "flex";

  // Focus the textarea
  const ta = $(`#edit-${blockId}`);
  if (ta) { ta.focus(); ta.select(); }
}

// Double-click to edit
document.addEventListener("dblclick", (e) => {
  const item = e.target.closest("[data-block-id]");
  if (item && !item.classList.contains("editing")) {
    editBlock(item.dataset.blockId);
  }
});

function saveEditBlock(blockId) {
  const ta = $(`#edit-${blockId}`);
  if (!ta) return;
  pendingEdits[blockId] = ta.value;

  const item = $(`[data-block-id="${blockId}"]`);
  if (item) {
    const contentEl = item.querySelector(".block-item-content");
    const actionsEl = item.querySelector(".block-item-actions");
    if (contentEl) contentEl.textContent = ta.value || "(空)";
    contentEl.innerHTML = escHtml(ta.value) || "<em style='color:var(--text-muted)'>空</em>";
    item.classList.remove("editing");
    if (actionsEl) actionsEl.style.display = "none";
  }
}

function cancelEditBlock(blockId) {
  const item = $(`[data-block-id="${blockId}"]`);
  if (item) {
    const contentEl = item.querySelector(".block-item-content");
    const actionsEl = item.querySelector(".block-item-actions");
    const original = pendingEdits[blockId] !== undefined
      ? escHtml(pendingEdits[blockId])
      : (item.querySelector("textarea")?.defaultValue || "");
    if (contentEl) contentEl.textContent = original || "(空)";
    // Re-render with the stored text or original
    contentEl.innerHTML = original || "<em style='color:var(--text-muted)'>空</em>";
    item.classList.remove("editing");
    if (actionsEl) actionsEl.style.display = "none";
  }
}

/* ─── Approve / Reject ─── */

async function approveReview(taskId, action) {
  const reasonEl = $(`#reason-${taskId}`);
  const reason = reasonEl ? reasonEl.value.trim() : "";

  if (action === "reject" && !reason) {
    alert("拒绝时必须填写原因");
    if (reasonEl) reasonEl.focus();
    return;
  }

  // Collect pending edits as edited_blocks
  const editedBlocks = Object.entries(pendingEdits).map(([blockId, newContent]) => ({
    block_id: blockId,
    new_content: newContent,
  }));

  const card = $(`#review-${taskId}`);
  if (card) card.style.opacity = ".5";

  try {
    const r = await fetch(`${API}/review/${taskId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        reason,
        edited_blocks: editedBlocks,
      }),
    });

    if (r.ok) {
      // Clear edits for this task
      Object.keys(pendingEdits).forEach((k) => delete pendingEdits[k]);
      updateDocument(taskId, { status: action === "approve" ? "indexed" : "failed" });
      loadReviews();
      updateBadges();
    } else {
      const detail = ((await r.json().catch(() => ({}))).detail) || "操作失败";
      if (card) card.style.opacity = "1";
      alert(detail);
    }
  } catch (e) {
    if (card) card.style.opacity = "1";
    alert(e.message);
  }
}

/* ─── Reprocess ─── */

async function reprocessDocument(taskId) {
  if (!confirm("确定要重新处理这个文档吗？")) return;

  try {
    const r = await fetch(`${API}/review/${taskId}/reprocess`, { method: "POST" });
    if (r.ok) {
      const data = await r.json();
      alert(data.message || "已加入处理队列");
      loadReviews();
      updateBadges();
    } else {
      const detail = ((await r.json().catch(() => ({}))).detail) || "操作失败";
      alert(detail);
    }
  } catch (e) {
    alert(e.message);
  }
}

// Expose for inline onclick
window.approveReview = approveReview;
window.toggleReviewDetail = toggleReviewDetail;
window.switchDetailTab = switchDetailTab;
window.editBlock = editBlock;
window.saveEditBlock = saveEditBlock;
window.cancelEditBlock = cancelEditBlock;
window.reprocessDocument = reprocessDocument;
window.showDocDetail = showDocDetail;
window.closeDocDetail = closeDocDetail;
window.switchDocDetailTab = switchDocDetailTab;

/* ─── Badges ─── */

function updateBadges() {
  const docs = getDocuments();
  const reviewCount = docs.filter((d) => d.status === "review").length;
  const docBadge = $("#doc-badge");
  if (docs.length) {
    docBadge.textContent = docs.length;
    docBadge.style.display = "inline";
  } else {
    docBadge.style.display = "none";
  }
}

// Auto-refresh review when switching to review tab
new MutationObserver(() => {
  if ($("#tab-review")?.classList.contains("active")) loadReviews();
}).observe($("#tab-review"), { attributes: true, attributeFilter: ["class"] });

// Periodic poll: every 15 s while the review tab is visible, pick up
// documents whose background processing completed after the last check.
let _reviewPollId = null;
function _startReviewPoll() {
  if (_reviewPollId) clearInterval(_reviewPollId);
  _reviewPollId = setInterval(() => {
    if ($("#tab-review")?.classList.contains("active")) loadReviews();
  }, 15000);
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (_reviewPollId) { clearInterval(_reviewPollId); _reviewPollId = null; }
  } else {
    _startReviewPoll();
  }
});
_startReviewPoll();

// Initial loads
loadReviews();
updateBadges();
renderDocuments();

/* ═══════════════════════════════════════════════════
   Utilities
   ═══════════════════════════════════════════════════ */

function escHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, len) {
  if (str.length <= len) return str;
  return str.slice(0, len) + "…";
}

/* ═══════════════════════════════════════════════════
   Window resize — auto height
   ═══════════════════════════════════════════════════ */

// Ensure messages scroll to bottom on window resize
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 100);
});
