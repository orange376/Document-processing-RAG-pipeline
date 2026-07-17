/* ── State ── */
const API_BASE = '/api/v1';
let currentTab = 'upload';
let healthInterval = null;

/* ── DOM refs ── */
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

/* ── Navigation ── */
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  currentTab = tab;
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab-content').forEach(t => t.classList.toggle('active', t.id === 'tab-' + tab));
}

/* ── Health Check ── */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    const dot = $('#health-dot');
    const txt = $('#health-text');
    if (data.status === 'ok') {
      dot.className = 'health-dot ok';
      txt.textContent = '服务正常';
    } else {
      dot.className = 'health-dot err';
      txt.textContent = '异常';
    }
  } catch {
    $('#health-dot').className = 'health-dot err';
    $('#health-text').textContent = '无法连接';
  }
}
checkHealth();
healthInterval = setInterval(checkHealth, 30000);

/* ── Upload ── */
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  const okTypes = ['.pdf', '.docx', '.doc'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!okTypes.includes(ext)) {
    showUploadResult('error', `不支持的文件格式: ${ext}`, true);
    return;
  }

  uploadZone.style.display = 'none';
  const progress = $('#upload-progress');
  progress.style.display = 'block';
  $('#upload-file-info').textContent = file.name;
  $('#progress-fill').style.width = '10%';
  $('#progress-text').textContent = '上传中...';

  try {
    const form = new FormData();
    form.append('file', file);

    $('#progress-fill').style.width = '30%';
    $('#progress-text').textContent = '处理中...';

    const res = await fetch(`${API_BASE}/documents/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || '上传失败');
    }
    const data = await res.json();

    $('#progress-fill').style.width = '60%';
    $('#progress-text').textContent = '处理中，等待结果...';

    // Poll status
    const taskId = data.task_id;
    let status = 'queued';
    let attempts = 0;
    const maxAttempts = 120;

    while (status !== 'indexed' && status !== 'review' && status !== 'failed' && attempts < maxAttempts) {
      await sleep(1500);
      attempts++;
      const sr = await fetch(`${API_BASE}/documents/${taskId}/status`);
      const sd = await sr.json();
      status = sd.status;
      if (attempts % 4 === 0) {
        $('#progress-text').textContent = `处理中 (${Math.round(attempts * 1.5)}s)...`;
      }
    }

    $('#progress-fill').style.width = '100%';
    progress.style.display = 'none';

    if (status === 'failed') {
      showUploadResult('error', `处理失败: ${data.message || '未知错误'}`);
    } else {
      const detail = await fetch(`${API_BASE}/documents/${taskId}/status`).then(r => r.json());
      showUploadResult('success', `处理完成`, true);
      addHistory(file.name, status);
    }
  } catch (err) {
    $('#progress-fill').style.width = '0%';
    progress.style.display = 'none';
    uploadZone.style.display = 'block';
    showUploadResult('error', err.message || '上传失败');
  }
}

function showUploadResult(type, msg, showStats = false) {
  const el = $('#upload-result');
  el.style.display = 'block';
  el.className = `upload-result ${type}`;
  if (showStats) {
    el.innerHTML = `<p>${msg}</p><div class="result-grid">
      <div class="result-stat"><strong id="r-pages">-</strong><span>页数</span></div>
      <div class="result-stat"><strong id="r-chunks">-</strong><span>切片</span></div>
      <div class="result-stat"><strong id="r-indexed">-</strong><span>已索引</span></div>
    </div>`;
    // Fetch task detail for stats (re-fetch status to get full data)
    // The status endpoint doesn't return all stats; show what we have
  } else {
    el.innerHTML = `<p>${msg}</p>`;
  }
  $('#upload-history').style.display = 'block';
}

function addHistory(name, status) {
  const list = $('#history-list');
  const item = document.createElement('div');
  item.className = 'history-item';
  const cls = status === 'failed' ? 'fail' : status === 'review' ? 'processing' : 'ok';
  const label = status === 'failed' ? '失败' : status === 'review' ? '待审核' : '完成';
  item.innerHTML = `<span>${name}</span><span class="h-status ${cls}">${label}</span>`;
  list.prepend(item);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ── Chat ── */
const chatInput = $('#chat-input');
const chatSend = $('#chat-send');
const chatMessages = $('#chat-messages');

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage('user', text);
  chatInput.value = '';
  chatSend.disabled = true;
  chatSend.textContent = '...';

  // Add a temporary bot message
  const tempId = 'temp-msg';
  addMessage('bot', '思考中...', tempId);

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, top_k: 10 }),
    });

    const data = await res.json();

    // Remove temp message
    const tempEl = document.getElementById(tempId);
    if (tempEl) tempEl.remove();

    if (data.answer) {
      let citationsHtml = '';
      if (data.citations && data.citations.length > 0) {
        citationsHtml = `<div class="msg-citations">
          <details>
            <summary>📎 ${data.citations.length} 个引用来源</summary>
            ${data.citations.map(c => `<div class="cite-item">📄 ${c.source_file} | 第${c.page_num}页${c.section ? ' | §' + c.section : ''}</div>`).join('')}
          </details>
        </div>`;
      }
      const confidenceHtml = data.confidence !== undefined
        ? `<div class="msg-confidence">置信度: ${(data.confidence * 100).toFixed(0)}%${data.needs_review ? ' ⚠️ 需审核' : ''}</div>`
        : '';
      addMessage('bot', data.answer + citationsHtml + confidenceHtml);
    } else {
      addMessage('bot', '抱歉，无法获取回答。');
    }
  } catch (err) {
    const tempEl = document.getElementById(tempId);
    if (tempEl) tempEl.remove();
    addMessage('bot', '请求失败: ' + err.message);
  } finally {
    chatSend.disabled = false;
    chatSend.textContent = '发送';
    chatInput.focus();
  }
}

function addMessage(role, html, id = null) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  if (id) div.id = id;
  div.innerHTML = `<div class="msg-content">${html}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* ── Review ── */
async function loadReviews() {
  try {
    const res = await fetch(`${API_BASE}/review/pending`);
    if (!res.ok) return;
    const items = await res.json();

    const badge = $('#review-badge');
    if (items.length > 0) {
      badge.textContent = items.length;
      badge.style.display = 'inline';
    } else {
      badge.style.display = 'none';
    }

    const list = $('#review-list');
    const empty = $('#review-empty');

    if (items.length === 0) {
      list.style.display = 'none';
      empty.style.display = 'block';
      return;
    }

    empty.style.display = 'none';
    list.style.display = 'block';
    list.innerHTML = '';

    items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'review-card';
      card.innerHTML = `
        <div class="r-header">
          <div>
            <div class="r-filename">${item.filename || item.task_id}</div>
            <div class="r-meta">ID: ${item.task_id} | 切片: ${item.chunk_count || 0}</div>
            ${item.error ? `<div class="r-error">${item.error}</div>` : ''}
          </div>
        </div>
        <div class="r-actions">
          <button class="btn btn-primary btn-sm" onclick="approveReview('${item.task_id}', true)">✅ 通过</button>
          <button class="btn btn-danger btn-sm" onclick="approveReview('${item.task_id}', false)">❌ 拒绝</button>
        </div>
      `;
      list.appendChild(card);
    });
  } catch {
    // ignore
  }
}

async function approveReview(taskId, approve) {
  try {
    const res = await fetch(`${API_BASE}/review/${taskId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: approve ? 'approve' : 'reject' }),
    });
    if (res.ok) {
      loadReviews();
    } else {
      const err = await res.json();
      alert('操作失败: ' + (err.detail || ''));
    }
  } catch (err) {
    alert('请求失败: ' + err.message);
  }
}

// Auto-refresh reviews when tab is shown
const reviewTab = document.getElementById('tab-review');
const observer = new MutationObserver(() => {
  if (reviewTab.classList.contains('active')) {
    loadReviews();
  }
});
observer.observe(reviewTab, { attributes: true, attributeFilter: ['class'] });

// Initial review load
loadReviews();

/* ── Keyboard shortcut ── */
document.addEventListener('keydown', (e) => {
  if (e.altKey || e.metaKey) {
    const map = { '1': 'upload', '2': 'chat', '3': 'review' };
    if (map[e.key]) { e.preventDefault(); switchTab(map[e.key]); }
  }
});
