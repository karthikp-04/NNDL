import { useState, useEffect } from 'react'
import { Sun, Moon, ChevronDown } from 'lucide-react'
import { useApp } from '../context/AppContext'

export default function Topbar({ page }) {
  const { selectedModel, setSelectedModel, addToast, theme, toggleTheme } = useApp()
  const [isBackendOnline, setIsBackendOnline] = useState(false)

  useEffect(() => {
    let isMounted = true
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:5000/health')
        const data = await res.json()
        if (isMounted) setIsBackendOnline(data.status === 'online')
      } catch {
        if (isMounted) setIsBackendOnline(false)
      }
    }
    checkHealth()
    const timer = setInterval(checkHealth, 4000)
    return () => {
      isMounted = false
      clearInterval(timer)
    }
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
    addToast(`Architecture switched to ${label}`, 'info')
  }

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h2>{page.title}</h2>
        <p>{page.sub}</p>
      </div>

      <div className="topbar-right">
        {/* Model Selector Dropdown */}
        <div className="model-select-wrapper" title="Select default neural network architecture">
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
            MODEL:
          </span>
          <select
            value={selectedModel || 'densenet121_gan'}
            onChange={handleModelChange}
            id="topbar-model-select"
            aria-label="Target Architecture Selector"
          >
            <option value="densenet121_gan" style={{ background: '#17191D', color: '#ECE9E2' }}>DenseNet-121 (cGAN Aug)</option>
            <option value="densenet121" style={{ background: '#17191D', color: '#ECE9E2' }}>DenseNet-121 (Baseline)</option>
            <option value="resnet50" style={{ background: '#17191D', color: '#ECE9E2' }}>ResNet-50</option>
            <option value="efficientnet_b0" style={{ background: '#17191D', color: '#ECE9E2' }}>EfficientNet-B0 (Edge)</option>
            <option value="compare" style={{ background: '#17191D', color: '#ECE9E2' }}>Tri-Model Compare</option>
          </select>
          <ChevronDown size={14} color="var(--text-muted)" style={{ pointerEvents: 'none' }} />
        </div>

        {/* Live Backend Status Badge */}
        <div
          className={`topbar-badge ${isBackendOnline ? 'online' : 'offline'}`}
          title={isBackendOnline ? "PyTorch Neural Inference Server Active on http://localhost:5000" : "Server offline. Start python api_server.py"}
        >
          <span className={`status-dot ${isBackendOnline ? 'online' : 'offline'}`}></span>
          <span>{isBackendOnline ? 'PyTorch Live' : 'API Offline'}</span>
        </div>

        {/* Dark/Light Theme Toggle */}
        <button
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={theme === 'dark' ? "Switch to light theme" : "Switch to dark theme"}
          aria-label="Toggle visual theme"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* User Monogram */}
        <div className="topbar-avatar" title="Madhav (CSE23661) · SolarGuard AOI Inspector">
          M
        </div>
      </div>
    </header>
  )
}
