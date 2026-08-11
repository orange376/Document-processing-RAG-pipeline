<template>
  <div>
    <el-card shadow="never" class="toolbar">
      <div class="tb-row">
        <div>
          <el-button type="primary" :loading="uploading" @click="fileInput.click()">
            <el-icon><Upload /></el-icon>&nbsp;上传文档
          </el-button>
          <input ref="fileInput" type="file" accept=".pdf,.docx,.doc" style="display:none" @change="upload" />
          <span class="tb-hint">支持 PDF / Word，含公式、图表、表格</span>
        </div>
        <el-button @click="load" :loading="loading">刷新</el-button>
      </div>
    </el-card>

    <el-card shadow="never">
      <el-table :data="docs" v-loading="loading" row-key="task_id" @expand-change="onExpand">
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
        <el-table-column prop="filename" label="文件名" min-width="240" show-overflow-tooltip>
          <template #default="{ row }">{{ cleanName(row.filename) }}</template>
        </el-table-column>
        <el-table-column label="状态" width="130">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="total_pages" label="页数" width="70" />
        <el-table-column prop="chunk_count" label="切片" width="70" />
        <el-table-column label="置信度" width="120">
          <template #default="{ row }">
            <el-progress :percentage="Math.round((row.confidence || 0) * 100)" :stroke-width="8" :show-text="false" />
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
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import client from '../api/client'

const docs = ref([])
const loading = ref(false)
const uploading = ref(false)
const detail = ref({})
const detailLoading = ref('')
const fileInput = ref(null)

function cleanName(name) { return (name || '').split('_').pop() }
function statusText(s) { return ({ queued: '排队中', processing: '处理中', indexed: '已索引', review: '待审核', failed: '失败', accepted: '已通过', rejected: '已拒绝' })[s] || s }
function statusType(s) { return ({ queued: 'info', processing: 'warning', indexed: 'success', review: 'warning', failed: 'danger' })[s] || 'info' }
function blockType(t) { return ({ formula: 'warning', table: 'danger', figure: 'success' })[t] || 'info' }

async function load() {
  loading.value = true
  try { const r = await client.get('/documents'); docs.value = r.data } finally { loading.value = false }
}

async function upload(e) {
  const file = e.target.files[0]
  if (!file) return
  uploading.value = true
  try {
    const fd = new FormData()
    fd.append('file', file)
    const r = await client.post('/documents/upload', fd, { timeout: 60000 })
    ElMessage.success('已加入处理队列')
    setTimeout(load, 2000)
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

onMounted(load)
</script>

<style scoped>
.toolbar { margin-bottom: 16px; }
.tb-row { display: flex; justify-content: space-between; align-items: center; }
.tb-hint { margin-left: 12px; color: #999; font-size: 13px; }
.detail-loading { padding: 20px; color: #999; text-align: center; }
.block-row { display: flex; gap: 8px; padding: 5px 0; border-bottom: 1px dashed #f0f0f0; }
.block-tag { flex-shrink: 0; }
.block-content { font-size: 13px; color: #333; line-height: 1.6; white-space: pre-wrap; }
.chunk-row { padding: 8px 0; border-bottom: 1px dashed #f0f0f0; }
.chunk-head { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
.chunk-page, .chunk-section { font-size: 12px; color: #999; }
.chunk-content { font-size: 13px; color: #333; line-height: 1.6; white-space: pre-wrap; }
</style>
