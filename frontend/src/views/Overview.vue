<template>
  <div>
    <div class="stat-grid">
      <el-card shadow="hover" class="stat-card">
        <div class="stat-icon" style="background:#ecf5ff;color:#409eff">📄</div>
        <div><div class="stat-num">{{ totalDocs }}</div><div class="stat-label">文档总数</div></div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-icon" style="background:#f0f9eb;color:#67c23a">🧩</div>
        <div><div class="stat-num">{{ totalChunks }}</div><div class="stat-label">索引切片</div></div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-icon" style="background:#fdf6ec;color:#e6a23c">⏳</div>
        <div><div class="stat-num">{{ processing }}</div><div class="stat-label">处理中</div></div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-icon" style="background:#fef0f0;color:#f56c6c">✔️</div>
        <div><div class="stat-num">{{ pendingReview }}</div><div class="stat-label">待审核</div></div>
      </el-card>
    </div>

    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>
            <div class="card-head"><span>最近文档</span><el-button size="small" @click="$router.push('/documents')">全部</el-button></div>
          </template>
          <el-table :data="recent" size="small">
            <el-table-column label="文件名" min-width="200" show-overflow-tooltip>
              <template #default="{ row }">{{ cleanName(row.filename) }}</template>
            </el-table-column>
            <el-table-column label="状态" width="100">
              <template #default="{ row }"><el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag></template>
            </el-table-column>
            <el-table-column label="切片" width="70" prop="chunk_count" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span>快速开始</span></template>
          <div class="quick-list">
            <div class="quick-item" @click="$router.push('/capabilities')"><span class="qi-icon">✨</span><div><b>了解系统能力</b><p>这个平台能做什么，怎么看效果</p></div></div>
            <div class="quick-item" @click="$router.push('/workspace')"><span class="qi-icon">💬</span><div><b>开始问答</b><p>对已索引文档提问，带引用溯源</p></div></div>
            <div class="quick-item" @click="$router.push('/documents')"><span class="qi-icon">📤</span><div><b>上传新文档</b><p>PDF / Word，自动解析索引</p></div></div>
            <div class="quick-item" @click="$router.push('/api')"><span class="qi-icon">🔌</span><div><b>API 调用</b><p>把流水线作为工具集成到你的 Agent</p></div></div>
          </div>
        </el-card>
        <el-card shadow="never" class="health-card">
          <template #header><span>服务状态</span></template>
          <el-descriptions :column="1" size="small">
            <el-descriptions-item label="API 服务"><el-tag size="small" type="success">运行中</el-tag></el-descriptions-item>
            <el-descriptions-item label="端点">{{ apiBase }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import client from '../api/client'

const docs = ref([])
const pendingReview = ref(0)
const apiBase = location.origin

const recent = computed(() => docs.value.slice(0, 5))
const totalDocs = computed(() => docs.value.length)
const totalChunks = computed(() => docs.value.reduce((a, d) => a + (d.chunk_count || 0), 0))
const processing = computed(() => docs.value.filter(d => ['queued', 'processing'].includes(d.status)).length)

function cleanName(n) { return (n || '').split('_').pop() }
function statusText(s) { return ({ queued: '排队中', processing: '处理中', indexed: '已索引', review: '待审核', failed: '失败' })[s] || s }
function statusType(s) { return ({ queued: 'info', processing: 'warning', indexed: 'success', review: 'warning', failed: 'danger' })[s] || 'info' }

onMounted(async () => {
  try {
    const [d, r] = await Promise.all([client.get('/documents'), client.get('/review/pending')])
    docs.value = d.data
    pendingReview.value = r.data.length
  } catch { /* server may be starting */ }
})
</script>

<style scoped>
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px; }
.stat-card :deep(.el-card__body) { display: flex; align-items: center; gap: 14px; }
.stat-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; }
.stat-num { font-size: 24px; font-weight: 700; }
.stat-label { font-size: 13px; color: #999; }
.card-head { display: flex; justify-content: space-between; align-items: center; }
.quick-list { display: flex; flex-direction: column; gap: 4px; }
.quick-item { display: flex; gap: 12px; padding: 10px; border-radius: 8px; cursor: pointer; }
.quick-item:hover { background: #f5f7fa; }
.qi-icon { font-size: 22px; }
.quick-item b { font-size: 14px; }
.quick-item p { margin: 2px 0 0; font-size: 12px; color: #999; }
.health-card { margin-top: 16px; }
</style>
