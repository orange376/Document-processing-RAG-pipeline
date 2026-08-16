<template>
  <div>
    <el-card shadow="never" class="toolbar">
      <div class="tb-row">
        <div class="tb-left">
          <el-button type="primary" :loading="uploading" @click="fileInput.click()">
            <el-icon><Upload /></el-icon>&nbsp;上传文档
          </el-button>
          <input ref="fileInput" type="file" accept=".pdf,.docx,.doc" style="display:none" @change="upload" />
          <span class="tb-hint">支持 PDF / Word，含公式、图表、表格</span>
        </div>
        <div class="tb-right">
          <el-tag v-if="queueActive" size="small" type="warning" effect="light" class="queue-badge">
            <el-icon class="spin"><Loading /></el-icon>
            处理队列 · 运行 {{ queueStats.running_count }}/{{ queueStats.max_concurrent }} · 排队 {{ queueStats.waiting_count }}
          </el-tag>
          <el-button @click="load" :loading="loading">刷新</el-button>
        </div>
      </div>
    </el-card>

    <el-card shadow="never">
      <el-empty
        v-if="!loading && !docs.length"
        description="还没有上传任何文档"
        class="doc-empty"
      >
        <el-button type="primary" @click="fileInput.click()">上传第一份文档</el-button>
      </el-empty>

      <el-table v-else :data="docs" v-loading="loading" row-key="task_id" @expand-change="onExpand">
        <el-table-column type="expand" width="36">
          <template #default="{ row }">
            <div v-if="detailLoading === row.task_id" class="detail-loading">加载结构…</div>
            <div v-else-if="detail[row.task_id]" class="doc-detail">
              <el-tabs>
                <el-tab-pane :label="`页面 (${detail[row.task_id].pages.length})`">
                  <el-collapse>
                    <el-collapse-item v-for="p in detail[row.task_id].pages" :key="p.page_num" :title="`第 ${p.page_num} 页 · ${p.text_length} 字`">
                      <div v-for="(b, bi) in p.blocks" :key="bi" class="block-row">
                        <el-tag size="small" :type="blockType(b.block_type)" class="block-tag">{{ b.block_type }}</el-tag>
                        <span class="block-content">{{ b.content }}</span>
                      </div>
                    </el-collapse-item>
                  </el-collapse>
                </el-tab-pane>
                <el-tab-pane :label="`切片 (${detail[row.task_id].chunks.length})`">
                  <div v-for="(c, ci) in detail[row.task_id].chunks" :key="ci" class="chunk-row">
                    <div class="chunk-head">
                      <el-tag size="small" type="info">{{ c.chunk_type }}</el-tag>
                      <span class="chunk-page">第{{ c.page_num }}页</span>
                      <span class="chunk-section">{{ c.section }}</span>
                    </div>
                    <div class="chunk-content">{{ c.content }}</div>
                  </div>
                </el-tab-pane>
              </el-tabs>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="文件名" min-width="240" show-overflow-tooltip>
          <template #default="{ row }">
            <div class="file-cell">
              <el-tag size="small" :type="fileType(row).type" effect="plain">{{ fileType(row).label }}</el-tag>
              <span>{{ cleanName(row.filename) }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="150">
          <template #default="{ row }">
            <div class="status-cell">
              <el-tag :type="statusType(row.status)" size="small">
                <el-icon v-if="row.status === 'processing'" class="spin"><Loading /></el-icon>
                {{ statusText(row.status) }}
              </el-tag>
              <span v-if="row.status === 'queued' && row.queue_position" class="queue-pos">第 {{ row.queue_position }} 位</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="total_pages" label="页数" width="70" />
        <el-table-column prop="chunk_count" label="切片" width="70" />
        <el-table-column label="置信度" width="120">
          <template #default="{ row }">
            <el-tooltip v-if="row.confidence" :content="`${(row.confidence * 100).toFixed(1)}%`" placement="top">
              <el-progress :percentage="Math.round((row.confidence || 0) * 100)" :stroke-width="8" :show-text="false"
                :color="row.confidence >= 0.75 ? '#67c23a' : row.confidence >= 0.4 ? '#e6a23c' : '#f56c6c'" />
            </el-tooltip>
            <span v-else class="no-conf">—</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="danger" text @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import client from '../api/client'

const docs = ref([])
const loading = ref(false)
const uploading = ref(false)
const detail = ref({})
const detailLoading = ref('')
const fileInput = ref(null)

// ---- 实时处理队列状态 ----
const queueStats = ref({ max_concurrent: 0, running_count: 0, waiting_count: 0 })
let queueTimer = null

const queueActive = computed(() => docs.value.some(d => d.status === 'queued' || d.status === 'processing'))

async function loadQueue() {
  try {
    const r = await client.get('/system/queue')
    queueStats.value = r.data
  } catch { /* server may be starting */ }
}

function cleanName(name) { return (name || '').split('_').pop() }
function statusText(s) { return ({ queued: '排队中', processing: '处理中', indexed: '已索引', review: '待审核', failed: '失败', accepted: '已通过', rejected: '已拒绝' })[s] || s }
function statusType(s) { return ({ queued: 'info', processing: 'warning', indexed: 'success', review: 'warning', failed: 'danger' })[s] || 'info' }
function blockType(t) { return ({ formula: 'warning', table: 'danger', figure: 'success' })[t] || 'info' }
function fileType(row) {
  const ext = (row.filename || row.file_type || '').split('.').pop().toUpperCase()
  const label = ({ PDF: 'PDF', DOCX: 'DOCX', DOC: 'DOC' })[ext] || (row.file_type || ext || 'FILE')
  const type = label === 'PDF' ? 'danger' : 'primary'
  return { label, type }
}

async function load() {
  loading.value = true
  try {
    const r = await client.get('/documents')
    docs.value = r.data
    loadQueue()
  } finally { loading.value = false }
}

async function upload(e) {
  const file = e.target.files[0]
  if (!file) return
  uploading.value = true
  try {
    const fd = new FormData()
    fd.append('file', file)
    await client.post('/documents/upload', fd, { timeout: 60000 })
    ElMessage.success('已加入处理队列')
    load()
    startQueuePolling()
  } finally { uploading.value = false; e.target.value = '' }
}

async function onExpand(row, rows) {
  if (rows.includes(row)) {
    detailLoading.value = row.task_id
    try { const r = await client.get(`/review/${row.task_id}`); detail.value[row.task_id] = r.data }
    catch { /* task may not have review detail */ }
    finally { detailLoading.value = '' }
  }
}

async function remove(row) {
  await ElMessageBox.confirm(`删除「${cleanName(row.filename)}」及其索引？`, '确认删除', { type: 'warning' })
  await client.delete(`/documents/${row.task_id}`)
  ElMessage.success('已删除')
  load()
}

// ---- 自动刷新：有排队/处理中的任务时每 3s 轮询状态 ----
function startQueuePolling() {
  if (queueTimer) return
  queueTimer = setInterval(() => { load(); }, 3000)
}
function stopQueuePolling() {
  if (queueTimer) { clearInterval(queueTimer); queueTimer = null }
}

onMounted(load)
onBeforeUnmount(stopQueuePolling)
</script>

<style scoped>
.toolbar { margin-bottom: 16px; }
.tb-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.tb-left, .tb-right { display: flex; align-items: center; gap: 10px; }
.tb-hint { color: #999; font-size: 13px; }
.queue-badge { display: inline-flex; align-items: center; gap: 4px; }
.spin { animation: rotating 1.2s linear infinite; }
@keyframes rotating { from { transform: rotate(0) } to { transform: rotate(360deg) } }
.doc-empty { padding: 40px 0; }
.file-cell { display: flex; align-items: center; gap: 8px; }
.status-cell { display: flex; align-items: center; gap: 6px; }
.queue-pos { font-size: 12px; color: #999; }
.no-conf { color: #c0c4cc; }
.detail-loading { padding: 20px; color: #999; text-align: center; }
.block-row { display: flex; gap: 8px; padding: 5px 0; border-bottom: 1px dashed #f0f0f0; }
.block-tag { flex-shrink: 0; }
.block-content { font-size: 13px; color: #333; line-height: 1.6; white-space: pre-wrap; }
.chunk-row { padding: 8px 0; border-bottom: 1px dashed #f0f0f0; }
.chunk-head { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
.chunk-page, .chunk-section { font-size: 12px; color: #999; }
.chunk-content { font-size: 13px; color: #333; line-height: 1.6; white-space: pre-wrap; }
</style>
