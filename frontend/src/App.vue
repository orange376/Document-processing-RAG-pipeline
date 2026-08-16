<template>
  <el-container class="layout">
    <el-aside width="220px" class="sidebar">
      <div class="brand">
        <div class="brand-logo">📚</div>
        <div class="brand-text">
          <div class="brand-name">RAG Pipeline</div>
          <div class="brand-sub">文档问答平台</div>
        </div>
      </div>
      <el-menu :default-active="$route.path" router class="nav-menu">
        <el-menu-item index="/">
          <el-icon><Odometer /></el-icon><span>系统概览</span>
        </el-menu-item>
        <el-menu-item index="/capabilities">
          <el-icon><MagicStick /></el-icon><span>能力展示</span>
        </el-menu-item>
        <el-menu-item index="/workspace">
          <el-icon><ChatDotRound /></el-icon><span>问答工作台</span>
        </el-menu-item>
        <el-menu-item index="/documents">
          <el-icon><Folder /></el-icon><span>文档管理</span>
        </el-menu-item>
        <el-menu-item index="/review">
          <el-icon><Finished /></el-icon><span>审核中心</span>
        </el-menu-item>
        <el-menu-item index="/api">
          <el-icon><Connection /></el-icon><span>API 工具</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="header">
        <div class="header-title">{{ $route.meta.title }}</div>
        <div class="header-right">
          <el-tag v-if="!healthCheck" size="small" type="info" effect="plain">
            <el-icon class="spin"><Loading /></el-icon>&nbsp;检查服务…
          </el-tag>
          <el-tag v-else :type="healthCheck.ok ? 'success' : 'danger'" effect="plain" size="small">
            <span class="dot" :class="healthCheck.ok ? 'dot-ok' : 'dot-bad'"></span>
            {{ healthCheck.ok ? '服务运行中' : '服务异常' }}
          </el-tag>
        </div>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

const healthCheck = ref(null)
let healthTimer = null

async function checkHealth() {
  try {
    const r = await fetch('/api/v1/health')
    healthCheck.value = { ok: r.ok }
  } catch {
    healthCheck.value = { ok: false }
  }
}

onMounted(() => { checkHealth(); healthTimer = setInterval(checkHealth, 15000) })
onBeforeUnmount(() => { if (healthTimer) clearInterval(healthTimer) })
</script>

<style>
body { margin: 0; background: #f5f7fa; font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }
.layout { height: 100vh; }
.sidebar { background: #1f2d3d; }
.brand { display: flex; align-items: center; gap: 10px; padding: 18px 16px; color: #fff; }
.brand-logo { font-size: 26px; }
.brand-name { font-size: 16px; font-weight: 600; }
.brand-sub { font-size: 11px; color: #8a9bb0; }
.nav-menu { border-right: none; background: transparent; --el-menu-bg-color: #1f2d3d; --el-menu-text-color: #a0b0c3; --el-menu-active-color: #fff; --el-menu-hover-bg-color: #2a3b4e; }
.nav-menu .el-menu-item.is-active { background: #409eff; color: #fff; }
.header { display: flex; align-items: center; justify-content: space-between; background: #fff; border-bottom: 1px solid #e4e7ed; }
.header-title { font-size: 16px; font-weight: 600; }
.header-right { display: flex; align-items: center; }
.spin { animation: rotating 1.2s linear infinite; }
@keyframes rotating { from { transform: rotate(0) } to { transform: rotate(360deg) } }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.dot-ok { background: #67c23a; box-shadow: 0 0 4px #67c23a; }
.dot-bad { background: #f56c6c; box-shadow: 0 0 4px #f56c6c; }
.main { padding: 20px; }
</style>
