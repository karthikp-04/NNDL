import { useState, useEffect } from 'react'
import { useApp } from '../context/AppContext'

export default function Topbar({ page }) {
  const { selectedModel, setSelectedModel, addToast } = useApp()
  const [isBackendOnline, setIsBackendOnline] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:5000/health')
        const data = await res.json()
        setIsBackendOnline(data.status === 'online')
      } catch {
        setIsBackendOnline(false)
      }
    }
    checkHealth()
    const timer = setInterval(checkHealth, 4000)
    return () => clearInterval(timer)
  }, [])

  const handleModelChange = (e) => {
    const val = e.target.value
    setSelectedModel(val)
    const label = val === 'densenet121_gan'
      ? 'DenseNet-121 (cGAN Aug)'
      : val === 'densenet121'
      ? 'DenseNet-121 (Baseline)'
      : val === 'resnet50'
      ? 'ResNet-50'
      : val === 'efficientnet_b0'
      ? 'EfficientNet-B0'
      : 'Tri-Model Compare'
    addToast?.(`Selected model: ${label}`, 'info')
  }

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h2>{page.title}</h2>
        <p>{page.sub}</p>
      </div>

      <div className="topbar-right" style={{ display: 'flex', alignItems: 'center', gap: '.8rem' }}>
        {/* Model Selector Pill */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '.5rem',
          background: 'var(--bg2)', padding: '.35rem .75rem',
          borderRadius: '999px', border: '1px solid var(--border)',
          fontSize: '.78rem'
        }}>
          <span style={{ color: 'var(--dim)', fontWeight: 600 }}>MODEL:</span>
          <select
            value={selectedModel || 'densenet121_gan'}
            onChange={handleModelChange}
            id="topbar-model-select"
            style={{
              background: 'transparent',
              color: selectedModel === 'densenet121_gan' ? '#EC4899' : selectedModel === 'densenet121' ? '#60A5FA' : selectedModel === 'resnet50' ? '#A78BFA' : selectedModel === 'efficientnet_b0' ? '#10B981' : '#F59E0B',
              border: 'none',
              fontWeight: 700,
              cursor: 'pointer',
              outline: 'none',
              fontSize: '.78rem'
            }}
          >
            <option value="densenet121_gan" style={{ background: '#1E293B', color: '#fff' }}>
              DenseNet-121 (cGAN Aug) ⭐
            </option>
            <option value="densenet121" style={{ background: '#1E293B', color: '#fff' }}>
              DenseNet-121 (Baseline)
            </option>
            <option value="resnet50" style={{ background: '#1E293B', color: '#fff' }}>
              ResNet-50
            </option>
            <option value="efficientnet_b0" style={{ background: '#1E293B', color: '#fff' }}>
              EfficientNet-B0 (Edge)
            </option>
            <option value="compare" style={{ background: '#1E293B', color: '#fff' }}>
              Tri-Model Compare
            </option>
          </select>
        </div>

        <div
          className="topbar-badge"
          title={isBackendOnline ? "PyTorch Neural Engine Live: http://localhost:5000" : "Server Offline (Start api_server.py)"}
          style={{
            borderColor: isBackendOnline ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
            color: isBackendOnline ? '#4ADE80' : '#F87171'
          }}
        >
          <span
            className="status-dot"
            style={{
              width: 6, height: 6,
              background: isBackendOnline ? '#22C55E' : '#EF4444',
              boxShadow: isBackendOnline ? '0 0 6px #22C55E' : 'none'
            }}
          ></span>
          {isBackendOnline ? 'PyTorch Online' : 'API Offline'}
        </div>
        <div className="topbar-avatar" title="Madhav (CSE23661)">M</div>
      </div>
    </header>
  )
}
