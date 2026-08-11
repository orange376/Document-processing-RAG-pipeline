<template>
  <div>
    <el-card shadow="never" class="hero">
      <h2>这个系统能帮你做什么？</h2>
      <p class="hero-sub">
        面向中文企业文档的 RAG 问答平台 —— 上传 PDF/Word，系统自动解析版面、识别公式图表、
        结构切片、建索引，然后你就能对着文档提问并获得带引用的回答。
      </p>
      <div class="hero-actions">
        <el-button type="primary" size="large" @click="$router.push('/workspace')">开始提问</el-button>
        <el-button size="large" @click="$router.push('/documents')">上传文档</el-button>
      </div>
    </el-card>

    <div class="cap-grid">
      <el-card v-for="cap in caps" :key="cap.title" shadow="hover" class="cap-card">
        <div class="cap-icon" :style="{ background: cap.color + '1a', color: cap.color }">{{ cap.icon }}</div>
        <h3>{{ cap.title }}</h3>
        <p class="cap-desc">{{ cap.desc }}</p>
        <div class="cap-detail">
          <div v-for="d in cap.details" :key="d" class="cap-detail-item">✓ {{ d }}</div>
        </div>
        <div class="cap-example" v-if="cap.example">
          <div class="cap-example-label">示例</div>
          <code>{{ cap.example }}</code>
        </div>
      </el-card>
    </div>

    <el-card shadow="never" class="howto">
      <h3>三步上手</h3>
      <el-steps :active="3" align-center finish-status="success">
        <el-step title="上传文档" description="PDF / Word，可含公式图片、图表、表格" />
        <el-step title="自动解析" description="版面分析 → 公式识别 → 图表描述 → 结构切片 → 建索引" />
        <el-step title="开始问答" description="提问即得带引用溯源的答案，低置信度自动转人工审核" />
      </el-steps>
    </el-card>
  </div>
</template>

<script setup>
const caps = [
  {
    icon: '📄', color: '#409eff', title: '文档解析',
    desc: 'PDF 与 Word 深度解析：版面分析、扫描件 OCR、公式、图表、表格全部转成可检索文本。',
    details: ['PP-DocLayoutV3 25 类版面识别', 'MathType/Equation 内嵌公式识别', '架构图/流程图 → 文字描述', '表格 → Markdown 结构化'],
    example: '上传「课程论文.pdf」，自动解析出 144 个版面元素、68 个切片',
  },
  {
    icon: '🔍', color: '#67c23a', title: '智能问答',
    desc: '对文档提问，混合检索 + 重排找到最相关内容，LLM 生成带引用溯源的答案。',
    details: ['向量 + 关键词混合检索（RRF 融合）', 'BGE 重排精筛', '每个答案带「来源: 文件 | 第N页」', '流式输出'],
    example: '「系统如何设计数据库表结构？」→ 答案 + 5 个引用切片',
  },
  {
    icon: '🧮', color: '#e6a23c', title: '公式识别',
    desc: 'MathType / Equation Editor / 图片里的数学公式，识别为 LaTeX 存入索引。',
    details: ['本地 pix2tex 推理，GPU 加速', 'OMML → LaTeX 递归转换', 'VML WMF 高分辨率渲染'],
    example: '问「求根公式是什么？」能直接搜到文档里的 \\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}',
  },
  {
    icon: '📊', color: '#909399', title: '图表理解',
    desc: '架构图、用例图、流程图、ER 图通过多模态模型转成文字描述，可被检索。',
    details: ['Qwen-VL 视觉描述', '节点/箭头/层次/数据流向全描述', '结果按裁剪图缓存'],
    example: '「系统的总体架构分几层？」命中架构图的文字描述',
  },
  {
    icon: '✔️', color: '#f56c6c', title: '置信度 + 审核',
    desc: '5 维置信度评分，低置信度自动进入人工审核，可编辑纠正后通过/拒绝。',
    details: ['布局/OCR/表格/切片/重排 5 维评分', '三级阈值：直接答 / 待审核 / 拒答', '审核界面可内联编辑块内容', '反馈统计：错误类型分析'],
    example: '文档解析质量低 → 出现在「审核中心」，展开看切片逐块审',
  },
  {
    icon: '🔌', color: '#7c6bd6', title: 'API 工具化',
    desc: '纯 REST API，其他 Agent 可直接把这条流水线当作工具调用。',
    details: ['/documents/parse 纯解析不上索引', '/query 带引用溯源问答', 'Bearer Token 认证', '限流保护'],
    example: 'POST /api/v1/documents/parse → 结构化 pages + chunks JSON',
  },
]
</script>

<style scoped>
.hero { background: linear-gradient(135deg, #1f2d3d, #2a4a6b); color: #fff; border: none; margin-bottom: 20px; }
.hero h2 { margin: 0 0 10px; font-size: 24px; }
.hero-sub { color: #c8d6e5; margin: 0 0 20px; font-size: 15px; line-height: 1.7; max-width: 760px; }
.cap-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; margin-bottom: 20px; }
.cap-icon { width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; margin-bottom: 10px; }
.cap-card h3 { margin: 0 0 6px; font-size: 16px; }
.cap-desc { color: #666; font-size: 13px; line-height: 1.6; margin: 0 0 10px; min-height: 40px; }
.cap-detail-item { font-size: 12px; color: #888; line-height: 1.9; }
.cap-example { margin-top: 10px; background: #f5f7fa; border-radius: 6px; padding: 8px 10px; }
.cap-example-label { font-size: 11px; color: #999; margin-bottom: 4px; }
.cap-example code { font-size: 12px; color: #409eff; word-break: break-all; }
.howto { margin-bottom: 20px; }
</style>
