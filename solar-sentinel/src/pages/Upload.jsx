import { useState, useRef, useCallback } from 'react'
import { useApp } from '../context/AppContext'
import { DEMO_SOLAR_PANEL } from '../utils/constants'

// Demo SVG images for instant testing without needing local files
const DEMO_PRESETS = [
  {
    id: 'crack',
    name: 'Micro Crack Defect',
    icon: '🔴',
    file: 'cell_micro_crack_EL_021.jpg',
    data: DEMO_SOLAR_PANEL,
    tag: 'Crack / Fracture'
  },
  {
    id: 'black_core',
    name: 'Black Core Inactive Area',
    icon: '⚡',
    file: 'black_core_inactive_045.jpg',
    data: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%233B82F6" stroke-width="2" stroke-opacity="0.3"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><rect x="275" y="135" width="250" height="230" fill="%23000000" stroke="%23EF4444" stroke-width="2" stroke-dasharray="4"/><text x="290" y="260" font-family="monospace" font-size="16" fill="%23EF4444">INACTIVE CORE (BLACK)</text></svg>`,
    tag: 'Severe Defect'
  },
  {
    id: 'finger',
    name: 'Finger Interruption',
    icon: '🟠',
    file: 'finger_interruption_018.jpg',
    data: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%233B82F6" stroke-width="2" stroke-opacity="0.3"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><line x1="140" y1="200" x2="400" y2="200" stroke="%23F59E0B" stroke-width="4" stroke-dasharray="8,6"/><text x="150" y="190" font-family="monospace" font-size="13" fill="%23F59E0B">BROKEN BUSBAR FINGER</text></svg>`,
    tag: 'Moderate Defect'
  },
  {
    id: 'dislocation',
    name: 'Crystal Dislocation',
    icon: '🟣',
    file: 'dislocation_crystal_0210.jpg',
    data: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%233B82F6" stroke-width="2" stroke-opacity="0.3"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><path d="M 420 150 Q 550 250 630 350 Q 480 320 420 150" fill="%238B5CF6" fill-opacity="0.3" stroke="%238B5CF6" stroke-width="2"/><text x="440" y="270" font-family="monospace" font-size="14" fill="%23C4B5FD">DISLOCATION BOUNDARY</text></svg>`,
    tag: 'DenseNet 99% F1'
  },
  {
    id: 'healthy',
    name: 'Normal Healthy Panel',
    icon: '🟢',
    file: 'healthy_normal_panel_025.jpg',
    data: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%2322C55E" stroke-width="2" stroke-opacity="0.4"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><text x="260" y="250" font-family="monospace" font-size="18" fill="%2322C55E">OPTIMAL EL LUMINESCENCE (HEALTHY)</text></svg>`,
    tag: 'Healthy / Clean'
  }
]

export default function Upload() {
  const { navigate, setAnalysis, addToast, selectedModel, setSelectedModel } = useApp()
  const [dragging, setDragging]   = useState(false)
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(DEMO_PRESETS[0].data)
  const [fileName, setFileName]   = useState(DEMO_PRESETS[0].file)
  const fileInputRef              = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) { addToast('Please upload an image file', 'error'); return }
    setFile(f)
    setFileName(f.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
    addToast('Image loaded successfully', 'success')
  }

  const selectPreset = (p) => {
    setFile(null)
    setPreview(p.data)
    setFileName(p.file)
    addToast(`Loaded preset: ${p.name}`, 'info')
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const handleAnalyze = () => {
    if (!preview) {
      addToast('Please upload or select an image first', 'error')
      return
    }
    setAnalysis({
      image: preview,
      fileName: fileName || 'demo_panel.jpg',
      model: selectedModel || 'densenet121'
    })
    navigate('analyzing')
  }

  return (
    <div>
      <div className="upload-hero" style={{ marginBottom: '1.2rem' }}>
        <h1 style={{ fontSize: '1.8rem', marginBottom: '.2rem' }}>
          <span className="gradient-text">Solar Defect Classification & Inference</span>
        </h1>
        <p style={{ color: 'var(--muted)', marginTop: '.2rem', fontSize: '.88rem' }}>
          Select a Deep Learning architecture below, upload or pick an EL image, and run real-time inference or model comparison.
        </p>
      </div>

      {/* Model Selection Header / Cards */}
      <div style={{ maxWidth: 840, margin: '0 auto 1.1rem auto' }}>
        <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: '.4rem' }}>
          Select Architecture for Inference & Evaluation:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '.6rem' }}>
          {/* DenseNet-121 */}
          <div
            onClick={() => setSelectedModel('densenet121')}
            style={{
              padding: '.75rem .9rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'densenet121' ? 'var(--blue)' : 'var(--border)'}`,
              background: selectedModel === 'densenet121' ? 'rgba(59,130,246,0.12)' : 'var(--bg2)',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.92rem' }}>DenseNet-121</span>
              <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: 999, background: 'var(--blue)', color: '#fff' }}>
                IN-LINE AOI
              </span>
            </div>
            <div style={{ fontSize: '.74rem', color: 'var(--muted)', marginBottom: '.3rem' }}>
              In-Line AOI Sorting · 6.97M Params
            </div>
            <div style={{ display: 'flex', gap: '.8rem', fontSize: '.72rem', color: 'var(--green)' }}>
              <span>⚡ 18.27 ms</span>
              <span>🎯 87.0% Acc</span>
            </div>
          </div>

          {/* ResNet-50 */}
          <div
            onClick={() => setSelectedModel('resnet50')}
            style={{
              padding: '.75rem .9rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'resnet50' ? 'var(--purple, #8B5CF6)' : 'var(--border)'}`,
              background: selectedModel === 'resnet50' ? 'rgba(139,92,246,0.12)' : 'var(--bg2)',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.92rem' }}>ResNet-50</span>
              <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: 999, background: '#8B5CF6', color: '#fff' }}>
                DIAGNOSTIC
              </span>
            </div>
            <div style={{ fontSize: '.74rem', color: 'var(--muted)', marginBottom: '.3rem' }}>
              Focal Loss Diagnostics · 25.56M Params
            </div>
            <div style={{ display: 'flex', gap: '.8rem', fontSize: '.72rem', color: '#C4B5FD' }}>
              <span>⏱️ 42.10 ms</span>
              <span>🎯 91.71% Acc</span>
            </div>
          </div>

          {/* EfficientNet-B0 (Ours) */}
          <div
            onClick={() => setSelectedModel('efficientnet_b0')}
            style={{
              padding: '.75rem .9rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'efficientnet_b0' ? '#10B981' : 'var(--border)'}`,
              background: selectedModel === 'efficientnet_b0' ? 'rgba(16,185,129,0.14)' : 'var(--bg2)',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.92rem' }}>EfficientNet-B0</span>
              <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: 999, background: '#10B981', color: '#fff' }}>
                EDGE DRONE
              </span>
            </div>
            <div style={{ fontSize: '.74rem', color: 'var(--muted)', marginBottom: '.3rem' }}>
              MBConv & SE Attention · 4.02M Params
            </div>
            <div style={{ display: 'flex', gap: '.8rem', fontSize: '.72rem', color: '#6EE7B7' }}>
              <span>⚡ 14.80 ms</span>
              <span>🎯 86.4% Acc</span>
            </div>
          </div>

          {/* Tri-Model Compare Mode */}
          <div
            onClick={() => setSelectedModel('compare')}
            style={{
              padding: '.75rem .9rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'compare' ? 'var(--orange, #F59E0B)' : 'var(--border)'}`,
              background: selectedModel === 'compare' ? 'rgba(245,158,11,0.12)' : 'var(--bg2)',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.92rem' }}>Multi-Model Compare</span>
              <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: 999, background: '#F59E0B', color: '#000' }}>
                TRI-MODEL
              </span>
            </div>
            <div style={{ fontSize: '.74rem', color: 'var(--muted)', marginBottom: '.3rem' }}>
              All 3 Networks Side-by-Side
            </div>
            <div style={{ display: 'flex', gap: '.8rem', fontSize: '.72rem', color: '#FCD34D' }}>
              <span>📊 3 Models</span>
              <span>⏱️ Speedup Stats</span>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Test Demo Presets */}
      <div style={{ maxWidth: 840, margin: '0 auto .9rem auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.4rem' }}>
          <span style={{ fontSize: '.74rem', color: 'var(--dim)', fontWeight: 600 }}>QUICK PRESETS (1-CLICK TEST SAMPLES):</span>
          <span style={{ fontSize: '.72rem', color: 'var(--blue)' }}>PVEL-AD Electroluminescence Test Set</span>
        </div>
        <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap' }}>
          {DEMO_PRESETS.map(p => (
            <button
              key={p.id}
              onClick={() => selectPreset(p)}
              className="btn"
              style={{
                background: fileName === p.file ? 'rgba(59,130,246,0.2)' : 'var(--bg2)',
                border: `1px solid ${fileName === p.file ? 'var(--blue)' : 'var(--border)'}`,
                color: fileName === p.file ? '#fff' : 'var(--muted)',
                fontSize: '.76rem', padding: '.3rem .7rem', borderRadius: 999,
                display: 'flex', alignItems: 'center', gap: '.35rem'
              }}
            >
              <span>{p.icon}</span>
              <span>{p.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Upload Zone & Preview Grid */}
      <div style={{ maxWidth: 840, margin: '0 auto' }}>
        <div
          className={`upload-zone ${dragging ? 'drag-over' : ''} ${preview ? 'has-file' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => !preview && fileInputRef.current?.click()}
          style={{ cursor: preview ? 'default' : 'pointer', padding: preview ? '1rem 1.25rem' : '2rem' }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
            id="file-input"
          />

          {!preview ? (
            <>
              <div className="upload-icon">{dragging ? '📂' : '☁️'}</div>
              <h3>{dragging ? 'Drop to upload' : 'Drag & drop solar EL image here'}</h3>
              <p>or click to browse from local computer · Supported: .jpg, .png, .tif</p>
            </>
          ) : (
            <div style={{ width: '100%' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem', alignItems: 'center' }}>
                <div>
                  <img
                    src={preview}
                    alt="Active panel"
                    style={{
                      maxHeight: 180, width: '100%', objectFit: 'contain',
                      borderRadius: 'var(--radius-sm)', background: '#0b1120', border: '1px solid var(--border)'
                    }}
                  />
                  <div style={{ marginTop: '.3rem', fontSize: '.74rem', color: 'var(--dim)', textAlign: 'center' }}>
                    📷 {fileName}
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
                  <div className="card" style={{ padding: '.75rem .9rem', background: 'var(--bg)' }}>
                    <div style={{ fontSize: '.78rem', fontWeight: 600, color: 'var(--text)', marginBottom: '.3rem' }}>
                      ⚙️ Selected Execution Configuration
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.25rem', fontSize: '.76rem', color: 'var(--muted)' }}>
                      <div><strong>Active Mode:</strong> <span style={{ color: '#60A5FA' }}>{selectedModel.toUpperCase()}</span></div>
                      <div><strong>Resolution:</strong> 224 × 224 Standardized RGB</div>
                      <div><strong>Backend API:</strong> http://localhost:5000</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '.6rem' }}>
                    <button
                      className="btn btn-primary"
                      onClick={handleAnalyze}
                      id="btn-run-analysis"
                      style={{ flex: 1, padding: '.8rem 1.2rem', fontSize: '.95rem', fontWeight: 700 }}
                    >
                      {selectedModel === 'compare'
                        ? 'Run Multi-Model Comparison (DenseNet vs ResNet vs EfficientNet)'
                        : `Run ${selectedModel === 'densenet121' ? 'DenseNet-121' : selectedModel === 'efficientnet_b0' ? 'EfficientNet-B0' : 'ResNet-50'} Inference`}
                    </button>
                    <button
                      className="btn btn-outline"
                      onClick={() => fileInputRef.current?.click()}
                      style={{ padding: '.8rem' }}
                      title="Upload custom image"
                    >
                      📁 Change
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
