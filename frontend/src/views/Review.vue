<template>
  <div class="review-page">
    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>
            <div class="card-head">
              <span>待审核任务（{{ pending.length }}）</span>
              <el-button size="small" @click="loadAll">刷新</el-button>
            </div>
          </template>
          <el-empty v-if="!pending.length" description="没有待审核任务" />
          <div v-for="t in pending" :key="t.task_id" class="review-item">
            <div class="ri-top" @click="toggleDetail(t)">
              <div class="ri-name">{{ cleanName(t.filename) }}</div>
              <div class="ri-meta">
                <span>{{ t.total_pages }}页 / {{ t.total_chunks }}切片</span>
                <el-tag size="small" :type="confType(t.confidence)">{{ (t.confidence * 100).toFixed(0) }}%</el-tag>
                <el-icon class="ri-arrow"><ArrowDown /></el-icon>
              </div>
            </div>
            <div v-if="expanded === t.task_id" class="ri-detail">
              <div class="conf-dims">
                <div v-for="(v, k) in t.confidence_details" :key="k" class="dim-row">
                  <span>{{ dimNames[k] || k }}</span>
                  <el-progress :percentage="Math.round(v * 100)" :stroke-width="6" :show-text="false" />
                  <span class="dim-val">{{ (v * 100).toFixed(0) }}%</span>
                </div>
              </div>
              <el-collapse>
                <el-collapse-item :title="`切片 (${t.chunks?.length || 0})`">
                  <div v-for="(c, ci) in t.chunks" :key="ci" class="chunk-item">
                    <div class="chunk-meta">第{{ c.page_num }}页 · {{ c.chunk_type }} · {{ c.section }}</div>
                    <div class="chunk-text">{{ c.content }}</div>
                  </div>
                </el-collapse-item>
              </el-collapse>
              <div class="approve-row">
                <el-input v-model="reason" placeholder="审批原因（拒绝时必填）" size="small" class="reason-input" />
                <el-button type="success" size="small" @click="decide(t, 'approve')">通过</el-button>
                <el-button type="danger" size="small" @click="decide(t, 'reject')">拒绝</el-button>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span>审核统计</span></template>
          <div v-if="stats.total_actions" class="stats">
            <div class="stat-grid">
              <div class="stat-box"><div class="stat-num">{{ stats.total_actions }}</div><div class="stat-label">审核动作</div></div>
              <div class="stat-box"><div class="stat-num">{{ (stats.approval_rate * 100).toFixed(0) }}%</div><div class="stat-label">通过率</div></div>
              <div class="stat-box"><div class="stat-num">{{ (stats.edit_rate * 100).toFixed(0) }}%</div><div class="stat-label">含编辑率</div></div>
            </div>
            <div ref="chartEl" class="chart"></div>
          </div>
          <el-empty v-else description="暂无审核数据，先处理文档积累反馈" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import client from '../api/client'

const pending = ref([])
const stats = ref({})
const expanded = ref('')
const reason = ref('')
const chartEl = ref(null)
let chart = null

const dimNames = { layout_quality: '版面', ocr_confidence: 'OCR', table_integrity: '表格', chunk_coherence: '切片', reranker_score: '重排', result_coverage: '覆盖' }

function cleanName(n) { return (n || '').split('_').pop() }
function confType(c) { return c >= 0.75 ? 'success' : c >= 0.4 ? 'warning' : 'danger' }

async function loadAll() {
  const [p, s] = await Promise.all([client.get('/review/pending'), client.get('/review/stats')])
  pending.value = p.data
  stats.value = s.data
  drawChart()
}

function toggleDetail(t) { expanded.value = expanded.value === t.task_id ? '' : t.task_id }

async function decide(t, action) {
  if (action === 'reject' && !reason.value.trim()) return ElMessage.warning('拒绝时必须填写原因')
  await client.post(`/review/${t.task_id}/approve`, { action, reason: reason.value.trim() })
  ElMessage.success(action === 'approve' ? '已通过' : '已拒绝')
  reason.value = ''
  loadAll()
}

function drawChart() {
  if (!chartEl.value || !stats.value.top_error_block_types?.length) return
  if (!chart) chart = echarts.init(chartEl.value)
  chart.setOption({
    title: { text: '问题类型 TOP', textStyle: { fontSize: 13 } },
    tooltip: {},
    grid: { left: 60, right: 20, top: 30, bottom: 20 },
    xAxis: { type: 'value', minInterval: 1 },
    yAxis: { type: 'category', data: stats.value.top_error_block_types.map(([k]) => k), inverse: true },
    series: [{ type: 'bar', data: stats.value.top_error_block_types.map(([, v]) => v), itemStyle: { color: '#e6a23c' }, barWidth: 14 }],
  })
}

onMounted(() => { loadAll(); window.addEventListener('resize', () => chart?.resize()) })
onBeforeUnmount(() => { window.removeEventListener('resize', () => chart?.resize()); chart?.dispose() })
</script>

<style scoped>
.card-head { display: flex; justify-content: space-between; align-items: center; }
.review-item { border: 1px solid #f0f0f0; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
.ri-top { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; cursor: pointer; }
.ri-top:hover { background: #f9fafb; }
.ri-name { font-weight: 600; font-size: 14px; }
.ri-meta { display: flex; gap: 10px; align-items: center; font-size: 12px; color: #999; }
.ri-arrow { transition: transform .2s; }
.ri-detail { padding: 0 14px 14px; border-top: 1px dashed #eee; }
.conf-dims { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px 0; }
.dim-row { display: grid; grid-template-columns: 40px 1fr 36px; align-items: center; gap: 6px; font-size: 12px; }
.dim-val { text-align: right; color: #333; }
.chunk-item { padding: 8px; border-bottom: 1px dashed #f0f0f0; }
.chunk-meta { font-size: 12px; color: #409eff; margin-bottom: 3px; }
.chunk-text { font-size: 13px; color: #333; line-height: 1.6; }
.approve-row { display: flex; gap: 8px; margin-top: 10px; }
.reason-input { flex: 1; }
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 10px; }
.stat-box { text-align: center; background: #f5f7fa; border-radius: 8px; padding: 14px 0; }
.stat-num { font-size: 22px; font-weight: 700; color: #303133; }
.stat-label { font-size: 12px; color: #999; }
.chart { height: 220px; }
</style>
