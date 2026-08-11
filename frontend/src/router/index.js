import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'overview', component: () => import('../views/Overview.vue'), meta: { title: '系统概览' } },
  { path: '/capabilities', name: 'capabilities', component: () => import('../views/Capabilities.vue'), meta: { title: '能力展示' } },
  { path: '/api', name: 'api', component: () => import('../views/ApiDocs.vue'), meta: { title: 'API 工具' } },
  { path: '/workspace', name: 'workspace', component: () => import('../views/Workspace.vue'), meta: { title: '问答工作台' } },
  { path: '/documents', name: 'documents', component: () => import('../views/Documents.vue'), meta: { title: '文档管理' } },
  { path: '/review', name: 'review', component: () => import('../views/Review.vue'), meta: { title: '审核中心' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
