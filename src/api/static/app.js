/* ── State ── */
const API = '/api/v1';
const HISTORY_KEY = 'rag_history';

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}
function addHistory(item) {
  const h = getHistory();
  h.unshift({ ...item, time: Date.now() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 20)));
}
function renderHistory() {
  const list = $('#upload-history-list');
  if (!list) return;
  const h = getHistory();
  list.innerHTML = h.map(item => {
    const cls = item.status === 'failed' ? 'fail' : item.status === 'review' ? 'review' : 'ok';
    const label = item.status === 'failed' ? '失败' : item.status === 'review' ? '待审核' : '完成';
    return `<div class="history-item">
      <span>${item.name}</span>
      <span class="h-status ${cls}">${label}</span>
      <span class="h-time">${new Date(item.time).toLocaleTimeString()}</span>
    </div>`;
  }).join('');
}

/* ── Navigation ── */
$$('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    $$('.tab-content').forEach(t => t.classList.toggle('active', t.id === 'tab-' + tab));
    if (tab === 'review') loadReviews();
  });
});

/* ── Health ── */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    $('#health-dot').className = 'health-dot ' + (d.status === 'ok' ? 'ok' : 'err');
    $('#health-text').textContent = d.status === 'ok' ? '服务正常' : '异常';
  } catch {
    $('#health-dot').className = 'health-dot err';
    $('#health-text').textContent = '无法连接';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

/* ── Upload ── */
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!['.pdf', '.docx', '.doc'].includes(ext)) {
    return showResult('error', `不支持 ${ext}`);
  }

  uploadZone.style.display = 'none';
  const progress = $('#upload-progress');
  progress.style.display = 'block';
  $('#upload-file-info').textContent = file.name;
  setProgress(10, '上传中...');

  try {
    const fd = new FormData();
    fd.append('file', file);
    setProgress(30, '处理中...');

    const res = await fetch(`${API}/documents/upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || '上传失败');
    const data = await res.json();

    setProgress(60, '等待结果...');

    // Poll status
    const taskId = data.task_id;
    let status = 'queued';
    for (let i = 0; i < 120; i++) {
      await sleep(1500);
      const sr = await fetch(`${API}/documents/${taskId}/status`);
      const sd = await sr.json();
      status = sd.status;
      if (['indexed', 'review', 'failed'].includes(status)) {
        showResult(
          status === 'failed' ? 'error' : 'success',
          `${sd.filename || file.name}`,
          sd.total_pages || 0,
          sd.chunk_count || 0,
          sd.indexed_count || 0,
          status,
        );
        break;
      }
      if (i % 4 === 3) setProgress(60 + i * 0.3, `处理中 (${Math.round(i * 1.5)}s)...`);
    }
    setProgress(100, '完成');
  } catch (err) {
    showResult('error', err.message);
  }
  progress.style.display = 'none';
  uploadZone.style.display = 'block';
}

function setProgress(pct, text) {
  $('#progress-fill').style.width = pct + '%';
  $('#progress-text').textContent = text;
}

function showResult(type, filename, pages, chunks, indexed, status) {
  const el = $('#upload-result');
  el.style.display = 'block';
  el.className = 'upload-result ' + type;
  if (type === 'success') {
    const statusLabel = status === 'review' ? '<span class="r-status review">待审核</span>' : '<span class="r-status ok">完成</span>';
    el.innerHTML = `<div class="result-filename">${filename} ${statusLabel}</div>
      <div class="result-grid">
        <div class="result-stat"><strong>${pages}</strong><span>页数</span></div>
        <div class="result-stat"><strong>${chunks}</strong><span>切片</span></div>
        <div class="result-stat"><strong>${indexed}</strong><span>已索引</span></div>
      </div>`;
    addHistory({ name: filename, status });
  } else {
    el.innerHTML = `<div class="result-filename" style="color:var(--danger)">${filename || '上传失败'}</div>`;
    addHistory({ name: filename || 'unknown', status: 'failed' });
  }
  renderHistory();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ── Chat ── */
const chatInput = $('#chat-input');
const chatSend = $('#chat-send');
const chatMessages = $('#chat-messages');

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMsg('user', text);
  chatInput.value = '';
  chatSend.disabled = true;
  chatSend.textContent = '...';

  const tempId = 't' + Date.now();
  addMsg('bot', '思考中...', tempId);

  try {
    const res = await fetch(`${API}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, top_k: 10 }),
    });
    const data = await res.json();
    document.getElementById(tempId)?.remove();

    if (data.answer) {
      let html = data.answer;
      if (data.citations?.length) {
        html += `<div class="msg-citations"><details>
          <summary>📎 ${data.citations.length} 个引用来源</summary>
          ${data.citations.map(c => `<div class="cite-item">📄 ${c.source_file} | 第${c.page_num}页${c.section ? ' | §' + c.section : ''}</div>`).join('')}
        </details></div>`;
      }
      if (data.confidence !== undefined) {
        html += `<div class="msg-confidence">置信度 ${(data.confidence * 100).toFixed(0)}%${data.needs_review ? ' ⚠️' : ''}</div>`;
      }
      addMsg('bot', html);
    } else {
      addMsg('bot', '无法获取回答。');
    }
  } catch (err) {
    document.getElementById(tempId)?.remove();
    addMsg('bot', '请求失败: ' + err.message);
  } finally {
    chatSend.disabled = false;
    chatSend.textContent = '发送';
    chatInput.focus();
  }
}

function addMsg(role, html, id) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  if (id) div.id = id;
  div.innerHTML = '<div class="msg-content">' + html + '</div>';
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* ── Review ── */
async function loadReviews() {
  try {
    const res = await fetch(`${API}/review/pending`);
    if (!res.ok) return;
    const items = await res.json();
    const badge = $('#review-badge');
    if (items.length) { badge.textContent = items.length; badge.style.display = 'inline'; }
    else { badge.style.display = 'none'; }

    $('#review-empty').style.display = items.length ? 'none' : 'block';
    const list = $('#review-list');
    list.style.display = items.length ? 'block' : 'none';
    list.innerHTML = items.map(item => `
      <div class="review-card">
        <div class="r-filename">${item.filename || item.task_id}</div>
        <div class="r-meta">ID: ${item.task_id} | 切片: ${item.chunk_count || 0}${item.error ? ' | ' + item.error : ''}</div>
        <div class="r-actions">
          <button class="btn btn-primary btn-sm" onclick="approve('${item.task_id}',true)">✅ 通过</button>
          <button class="btn btn-danger btn-sm" onclick="approve('${item.task_id}',false)">❌ 拒绝</button>
        </div>
      </div>`).join('');
  } catch {}
}

async function approve(id, ok) {
  try {
    const r = await fetch(`${API}/review/${id}/approve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: ok ? 'approve' : 'reject' }),
    });
    if (r.ok) loadReviews();
    else alert('操作失败: ' + ((await r.json()).detail || ''));
  } catch (e) { alert(e.message); }
}

// Watch review tab
new MutationObserver(() => {
  if ($('#tab-review')?.classList.contains('active')) loadReviews();
}).observe($('#tab-review'), { attributes: true, attributeFilter: ['class'] });

loadReviews();
renderHistory();

/* ── Hotkeys ── */
document.addEventListener('keydown', e => {
  if (e.altKey) {
    const map = { '1': 'workspace', '2': 'review' };
    const tab = map[e.key];
    if (tab) { e.preventDefault(); document.querySelector(`[data-tab="${tab}"]`)?.click(); }
  }
});
