<template>
  <div>
    <el-alert
      type="info" :closable="false" show-icon
      title="把这条流水线作为工具：所有能力都通过 REST API 暴露，其他 Agent 可用 HTTP 直接调用。"
      description="认证：若配置了 api_auth_token，请求需带 Authorization: Bearer <token>。"
    />

    <el-card shadow="never" class="try-card">
      <div class="try-header">
        <h3>在线试用：服务健康状态</h3>
        <el-button size="small" :loading="checking" @click="checkHealth">检测</el-button>
      </div>
      <pre v-if="health">{{ healthText }}</pre>
    </el-card>

    <div class="api-grid">
      <el-card v-for="ep in endpoints" :key="ep.path" shadow="hover" class="api-card">
        <div class="api-method" :class="ep.method.toLowerCase()">{{ ep.method }}</div>
        <div class="api-body">
          <div class="api-path">{{ ep.path }}</div>
          <div class="api-desc">{{ ep.desc }}</div>
          <pre class="api-example">{{ ep.example }}</pre>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import client from '../api/client'

const checking = ref(false)
const health = ref(null)
const healthText = ref('')

async function checkHealth() {
  checking.value = true
  try {
    const r = await client.get('/health')
    health.value = r.data
    healthText.value = JSON.stringify(r.data, null, 2)
  } finally {
    checking.value = false
  }
}

const endpoints = [
  {
    method: 'POST', path: '/documents/upload', desc: '上传文档并全流程处理（异步，返回 task_id）',
    example: `curl -X POST http://localhost:8001/api/v1/documents/upload \\
  -F "file=@报告.pdf"`,
  },
  {
    method: 'GET', path: '/documents/{task_id}/status', desc: '轮询处理状态（queued/processing/indexed/review/failed）',
    example: `curl http://localhost:8001/api/v1/documents/<task_id>/status`,
  },
  {
    method: 'POST', path: '/documents/parse', desc: '纯解析，不上索引 —— 供其他 Agent 消费结构化结果',
    example: `curl -X POST http://localhost:8001/api/v1/documents/parse \\
  -F "file=@报告.pdf"
# → { total_pages, chunks: [{ content, section, page_num }] }`,
  },
  {
    method: 'POST', path: '/query', desc: 'RAG 问答：答案 + 带页码引用 + 置信度',
    example: `curl -X POST http://localhost:8001/api/v1/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "系统架构分几层？", "top_k": 5}'
# → { answer, citations, confidence }`,
  },
  {
    method: 'GET', path: '/review/pending', desc: '待人工审核的任务列表（含置信度详情）',
    example: `curl http://localhost:8001/api/v1/review/pending`,
  },
  {
    method: 'GET', path: '/review/stats', desc: '审核反馈统计：错误率 / 问题类型 / 文件健康',
    example: `curl http://localhost:8001/api/v1/review/stats`,
  },
]
</script>

<style scoped>
.try-card { margin: 16px 0; }
.try-header { display: flex; align-items: center; justify-content: space-between; }
.try-header h3 { margin: 0; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto; line-height: 1.6; }
.api-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 14px; margin-top: 16px; }
.api-card { display: flex; gap: 12px; }
.api-method { font-weight: 700; font-size: 12px; padding: 3px 8px; border-radius: 4px; height: fit-content; }
.api-method.post { background: #ecf5ff; color: #409eff; }
.api-method.get { background: #f0f9eb; color: #67c23a; }
.api-path { font-family: monospace; font-weight: 600; font-size: 14px; }
.api-desc { color: #666; font-size: 12px; margin: 4px 0 8px; }
.api-example { font-size: 11px; }
</style>
