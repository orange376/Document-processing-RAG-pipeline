<template>
  <div class="workspace">
    <div class="chat-panel">
      <div class="chat-messages" ref="msgBox">
        <div v-if="!messages.length" class="chat-empty">
          <div class="empty-icon">💬</div>
          <p>对已索引的文档提问</p>
          <div class="suggestions">
            <el-tag v-for="s in suggestions" :key="s" class="sug" @click="send(s)">{{ s }}</el-tag>
          </div>
        </div>
        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div class="msg-avatar">{{ m.role === 'user' ? '🧑' : '🤖' }}</div>
          <div class="msg-body">
            <div class="msg-content">{{ m.content }}</div>
            <div v-if="m.role === 'assistant' && m.meta" class="msg-meta">
              <div class="meta-row">
                <el-tag size="small" :type="m.meta.needs_review ? 'warning' : 'success'" effect="light">
                  置信度 {{ (m.meta.confidence * 100).toFixed(0) }}%
                </el-tag>
                <span class="meta-note" v-if="m.meta.needs_review">⚠ 待人工审核</span>
              </div>
              <div v-if="m.meta.citations?.length" class="citations">
                <div class="cite-title">📎 引用来源（{{ m.meta.citations.length }}）</div>
                <div v-for="(c, ci) in m.meta.citations" :key="ci" class="cite-item">
                  <span class="cite-src">{{ c.source_file.split('_').pop() }}</span>
                  <span class="cite-page">第{{ c.page_num }}页</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div v-if="streaming" class="msg assistant">
          <div class="msg-avatar">🤖</div>
          <div class="msg-body"><div class="msg-content streaming"><span class="cursor">▋</span></div></div>
        </div>
      </div>
      <div class="chat-input">
        <el-input
          v-model="input" type="textarea" :rows="2" resize="none"
          placeholder="输入问题，Enter 发送（Shift+Enter 换行）"
          @keydown.enter.exact.prevent="send(input)"
        />
        <el-button type="primary" :loading="streaming" :disabled="!input.trim()" class="send-btn" @click="send(input)">
          发送
        </el-button>
      </div>
    </div>

    <div class="side-panel">
      <el-card shadow="never" v-if="lastMeta">
        <template #header><span>置信度评分</span></template>
        <div ref="gaugeEl" class="gauge"></div>
        <div class="conf-dims">
          <div v-for="(v, k) in lastMeta.confidence_details" :key="k" class="dim-row">
            <span class="dim-name">{{ dimNames[k] || k }}</span>
            <el-progress :percentage="Math.round(v * 100)" :stroke-width="8" :show-text="false" :color="dimColors[k]" />
            <span class="dim-val">{{ (v * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </el-card>
      <el-card shadow="never" v-else>
        <template #header><span>置信度评分</span></template>
        <div class="side-empty">提问后这里显示 5 维置信度分解</div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'

const messages = ref([])
const input = ref('')
const streaming = ref(false)
const msgBox = ref(null)
const gaugeEl = ref(null)
const lastMeta = ref(null)
let gaugeChart = null

const suggestions = ['系统总体架构分几层？', '有哪些功能模块？', '数据库如何设计？', '文档中提到了什么技术方案？']

const dimNames = { layout_quality: '版面质量', ocr_confidence: 'OCR 置信', table_integrity: '表格完整', chunk_coherence: '切片连贯', reranker_score: '重排得分', result_coverage: '结果覆盖' }
const dimColors = { layout_quality: '#409eff', ocr_confidence: '#67c23a', table_integrity: '#e6a23c', chunk_coherence: '#909399', reranker_score: '#f56c6c', result_coverage: '#7c6bd6' }

function scrollBottom() { nextTick(() => { msgBox.value?.scrollTo({ top: 99999 }) }) }

async function send(text) {
  if (!text.trim() || streaming.value) return
  input.value = ''
  messages.value.push({ role: 'user', content: text.trim() })
  const asst = { role: 'assistant', content: '' }
  messages.value.push(asst)
  streaming.value = true
  scrollBottom()
  try {
    const resp = await fetch('/api/v1/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text.trim(), top_k: 5 }),
    })
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop()
      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data: ')) continue
        try {
          const evt = JSON.parse(line.slice(6))
          if (evt.type === 'token') asst.content += evt.content
          else if (evt.type === 'meta') { lastMeta.value = evt; asst.meta = evt }
        } catch (e) { /* ignore partial */ }
      }
      scrollBottom()
    }
  } catch (err) {
    ElMessage.error('请求失败')
    asst.content = '请求失败：' + err.message
  } finally {
    streaming.value = false
    drawGauge()
    scrollBottom()
  }
}

function drawGauge() {
  if (!gaugeEl.value || !lastMeta.value) return
  if (!gaugeChart) gaugeChart = echarts.init(gaugeEl.value)
  gaugeChart.setOption({
    series: [{
      type: 'gauge', startAngle: 200, endAngle: -20, min: 0, max: 1,
      radius: '95%', progress: { show: true, width: 14 },
      axisLine: { lineStyle: { width: 14 } },
      axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { show: false }, pointer: { show: false },
      detail: {
        valueAnimation: true, formatter: v => (v * 100).toFixed(0) + '%',
        fontSize: 26, offsetCenter: [0, '10%'], color: '#303133', fontWeight: 700,
      },
      data: [{ value: lastMeta.value.confidence, itemStyle: { color: lastMeta.value.confidence >= 0.75 ? '#67c23a' : lastMeta.value.confidence >= 0.4 ? '#e6a23c' : '#f56c6c' } }],
    }],
  })
}

onMounted(() => { window.addEventListener('resize', () => gaugeChart?.resize()) })
onBeforeUnmount(() => { window.removeEventListener('resize', () => gaugeChart?.resize()); gaugeChart?.dispose() })
</script>

<style scoped>
.workspace { display: grid; grid-template-columns: 1fr 320px; gap: 16px; height: calc(100vh - 110px); }
.chat-panel { display: flex; flex-direction: column; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.chat-messages { flex: 1; overflow-y: auto; padding: 20px; }
.chat-empty { text-align: center; color: #999; padding-top: 60px; }
.empty-icon { font-size: 40px; }
.suggestions { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.sug { cursor: pointer; }
.msg { display: flex; gap: 10px; margin-bottom: 18px; }
.msg.user { flex-direction: row-reverse; }
.msg-avatar { width: 36px; height: 36px; border-radius: 50%; background: #f0f2f5; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.msg-body { max-width: 78%; }
.msg.user .msg-body { background: #409eff; color: #fff; padding: 10px 14px; border-radius: 12px 12px 2px 12px; }
.msg.assistant .msg-body { background: #f7f8fa; padding: 12px 14px; border-radius: 12px 12px 12px 2px; line-height: 1.7; }
.msg-meta { margin-top: 10px; border-top: 1px dashed #e4e7ed; padding-top: 10px; }
.meta-row { display: flex; align-items: center; gap: 8px; }
.meta-note { font-size: 12px; color: #e6a23c; }
.citations { margin-top: 8px; }
.cite-title { font-size: 12px; color: #666; margin-bottom: 6px; }
.cite-item { font-size: 12px; color: #409eff; padding: 3px 0; }
.cite-src { margin-right: 8px; }
.cite-page { color: #999; }
.streaming .cursor { animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0 } }
.chat-input { display: flex; gap: 10px; padding: 14px; border-top: 1px solid #eee; }
.send-btn { height: auto; }
.side-panel { display: flex; flex-direction: column; gap: 16px; }
.gauge { height: 150px; }
.conf-dims { margin-top: 6px; }
.dim-row { display: grid; grid-template-columns: 70px 1fr 36px; align-items: center; gap: 8px; margin: 8px 0; }
.dim-name { font-size: 12px; color: #666; }
.dim-val { font-size: 12px; color: #333; text-align: right; }
.side-empty { color: #999; font-size: 13px; padding: 20px 0; text-align: center; }
</style>
