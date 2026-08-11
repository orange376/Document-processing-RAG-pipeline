import axios from 'axios'
import { ElMessage } from 'element-plus'

// Same-origin in production (FastAPI serves the SPA); dev uses the Vite proxy.
const client = axios.create({ baseURL: '/api/v1', timeout: 60000 })

client.interceptors.response.use(
  (resp) => resp,
  (err) => {
    const detail = err.response?.data?.detail
    if (detail) ElMessage.error(typeof detail === 'string' ? detail : '请求失败')
    return Promise.reject(err)
  }
)

export default client
