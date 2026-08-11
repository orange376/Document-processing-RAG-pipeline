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
        <el-tag size="small" type="success" effect="plain">服务运行中</el-tag>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
// Element Plus icons are globally registered in main.js
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
.main { padding: 20px; }
</style>
