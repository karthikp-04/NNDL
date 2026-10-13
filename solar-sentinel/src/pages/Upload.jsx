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
  },
  {
    id: 'gan_sample',
    name: 'cGAN Synthetic Defect',
    icon: '🧪',
    file: 'synthetic_cgan_star_crack_0001.png',
    data: '/gan_samples/synthetic_star_crack/synthetic_cgan_star_crack_0001.png',
    tag: 'cGAN 64x64'
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
    if (!f.type.startsWith('image/')) { addToast('Please upload an image file (.jpg, .png, .tif)', 'error'); return }
    setFile(f)
    setFileName(f.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
    addToast(`Loaded image "${f.name}"`, 'success')
  }

  const selectPreset = (p) => {
    setFile(null)
    setPreview(p.data)
    setFileName(p.file)
    addToast(`Selected sample: ${p.name}`, 'info')
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const handleAnalyze = () => {
    if (!preview) {
      addToast('Please upload or select an image first', 'error')
      return
    }
    const modelToRun = selectedModel || 'densenet121_gan'
    setAnalysis({
      image: preview,
      fileName: fileName || 'solar_panel_test.jpg',
      model: modelToRun
    })
    navigate('analyzing')
  }

  const getModelName = (m) => {
    if (m === 'densenet121_gan') return 'DenseNet-121 (cGAN Augmented) ⭐'
    if (m === 'densenet121') return 'DenseNet-121 (Baseline) ⚡'
    if (m === 'resnet50') return 'ResNet-50 🛡️'
    if (m === 'efficientnet_b0') return 'EfficientNet-B0 🌿'
    if (m === 'compare') return 'Multi-Model Comparative Suite ⚖️'
    return 'DenseNet-121 (cGAN Augmented) ⭐'
  }

  const getButtonLabel = () => {
    if (selectedModel === 'compare') {
      return '⚖️ Run Tri-Model Comparison (DenseNet vs ResNet vs EfficientNet)'
    }
    if (selectedModel === 'densenet121_gan') {
      return '⭐ Run DenseNet-121 (cGAN Augmented) Inference'
    }
    if (selectedModel === 'densenet121') {
      return '⚡ Run DenseNet-121 (Baseline) Inference'
    }
    if (selectedModel === 'efficientnet_b0') {
      return '🌿 Run EfficientNet-B0 Inference'
    }
    return '🛡️ Run ResNet-50 Inference'
  }

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', paddingBottom: '2.5rem' }}>
      {/* Hidden File Input for browser file dialog */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => handleFile(e.target.files[0])}
        id="file-input"
      />

      {/* Hero Header */}
      <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
        <h1 style={{ fontSize: '1.65rem', fontWeight: 800, marginBottom: '.25rem' }}>
          <span className="gradient-text">Solar Defect Classification & Inference</span>
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: '.85rem', margin: 0 }}>
          Select your target Deep Learning architecture, upload any EL solar image, and launch real-time defect classification.
        </p>
      </div>

      {/* Step 1: Model Selection Cards */}
      <div style={{ marginBottom: '1.1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.4rem' }}>
          <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', letterSpacing: '.06em' }}>
            STEP 1 · SELECT TARGET ARCHITECTURE:
          </span>
          <span style={{ fontSize: '.74rem', color: '#EC4899', fontWeight: 600 }}>
            Active: <strong>{getModelName(selectedModel)}</strong>
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))', gap: '.55rem' }}>
          {/* DenseNet-121 (cGAN Augmented) */}
          <div
            onClick={() => setSelectedModel('densenet121_gan')}
            style={{
              padding: '.65rem .8rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'densenet121_gan' ? '#EC4899' : 'var(--border)'}`,
              background: selectedModel === 'densenet121_gan' ? 'rgba(236,72,153,0.18)' : 'var(--bg2)',
              boxShadow: selectedModel === 'densenet121_gan' ? '0 0 16px rgba(236,72,153,0.25)' : 'none',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.88rem' }}>DenseNet + cGAN</span>
              <span style={{ fontSize: '.6rem', fontWeight: 700, padding: '2px 5px', borderRadius: 999, background: '#EC4899', color: '#fff' }}>
                ⭐ +15.7% ACC
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: '.25rem' }}>
              cGAN Data-Balanced · 6.97M
            </div>
            <div style={{ display: 'flex', gap: '.6rem', fontSize: '.7rem', color: '#F472B6', fontWeight: 600 }}>
              <span>⚡ 18.35 ms</span>
              <span>🎯 48.5% Bal Acc</span>
            </div>
          </div>

          {/* DenseNet-121 Baseline */}
          <div
            onClick={() => setSelectedModel('densenet121')}
            style={{
              padding: '.65rem .8rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'densenet121' ? 'var(--blue)' : 'var(--border)'}`,
              background: selectedModel === 'densenet121' ? 'rgba(59,130,246,0.15)' : 'var(--bg2)',
              boxShadow: selectedModel === 'densenet121' ? '0 0 14px rgba(59,130,246,0.25)' : 'none',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.88rem' }}>DenseNet-121</span>
              <span style={{ fontSize: '.6rem', fontWeight: 700, padding: '2px 5px', borderRadius: 999, background: 'var(--blue)', color: '#fff' }}>
                BASELINE
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: '.25rem' }}>
              In-Line AOI · 6.97M Params
            </div>
            <div style={{ display: 'flex', gap: '.6rem', fontSize: '.7rem', color: 'var(--green)', fontWeight: 600 }}>
              <span>⚡ 18.27 ms</span>
              <span>🎯 87.0% Acc</span>
            </div>
          </div>

          {/* ResNet-50 */}
          <div
            onClick={() => setSelectedModel('resnet50')}
            style={{
              padding: '.65rem .8rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'resnet50' ? '#8B5CF6' : 'var(--border)'}`,
              background: selectedModel === 'resnet50' ? 'rgba(139,92,246,0.15)' : 'var(--bg2)',
              boxShadow: selectedModel === 'resnet50' ? '0 0 14px rgba(139,92,246,0.25)' : 'none',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.88rem' }}>ResNet-50</span>
              <span style={{ fontSize: '.6rem', fontWeight: 700, padding: '2px 5px', borderRadius: 999, background: '#8B5CF6', color: '#fff' }}>
                DIAGNOSTIC
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: '.25rem' }}>
              Focal Loss · 25.56M Params
            </div>
            <div style={{ display: 'flex', gap: '.6rem', fontSize: '.7rem', color: '#C4B5FD', fontWeight: 600 }}>
              <span>⏱️ 42.10 ms</span>
              <span>🎯 91.7% Acc</span>
            </div>
          </div>

          {/* EfficientNet-B0 */}
          <div
            onClick={() => setSelectedModel('efficientnet_b0')}
            style={{
              padding: '.65rem .8rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'efficientnet_b0' ? '#10B981' : 'var(--border)'}`,
              background: selectedModel === 'efficientnet_b0' ? 'rgba(16,185,129,0.15)' : 'var(--bg2)',
              boxShadow: selectedModel === 'efficientnet_b0' ? '0 0 14px rgba(16,185,129,0.25)' : 'none',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.88rem' }}>EfficientNet-B0</span>
              <span style={{ fontSize: '.6rem', fontWeight: 700, padding: '2px 5px', borderRadius: 999, background: '#10B981', color: '#fff' }}>
                EDGE DRONE
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: '.25rem' }}>
              MBConv Attention · 4.02M
            </div>
            <div style={{ display: 'flex', gap: '.6rem', fontSize: '.7rem', color: '#6EE7B7', fontWeight: 600 }}>
              <span>⚡ 14.80 ms</span>
              <span>🎯 86.4% Acc</span>
            </div>
          </div>

          {/* Tri-Model Compare */}
          <div
            onClick={() => setSelectedModel('compare')}
            style={{
              padding: '.65rem .8rem', borderRadius: 'var(--radius)', cursor: 'pointer',
              border: `2px solid ${selectedModel === 'compare' ? '#F59E0B' : 'var(--border)'}`,
              background: selectedModel === 'compare' ? 'rgba(245,158,11,0.15)' : 'var(--bg2)',
              boxShadow: selectedModel === 'compare' ? '0 0 14px rgba(245,158,11,0.25)' : 'none',
              transition: 'all .2s ease'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.2rem' }}>
              <span style={{ fontWeight: 700, color: '#fff', fontSize: '.88rem' }}>Tri-Model Compare</span>
              <span style={{ fontSize: '.6rem', fontWeight: 700, padding: '2px 5px', borderRadius: 999, background: '#F59E0B', color: '#000' }}>
                3-WAY
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: '.25rem' }}>
              Side-by-Side Validation
            </div>
            <div style={{ display: 'flex', gap: '.6rem', fontSize: '.7rem', color: '#FCD34D', fontWeight: 600 }}>
              <span>📊 Multi-Model</span>
              <span>⚖️ Agreement</span>
            </div>
          </div>
        </div>
      </div>

      {/* Step 2: Main Image Upload & Classification Hub */}
      <div style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg, 12px)',
        padding: '1.2rem',
        boxShadow: '0 8px 30px rgba(0,0,0,0.35)'
      }}>
        {/* Top Action Bar: Upload File Button & Quick Presets */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '.8rem',
          paddingBottom: '1rem',
          borderBottom: '1px solid var(--border)',
          marginBottom: '1rem'
        }}>
          <div>
            <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase' }}>
              STEP 2 · PROVIDE SOLAR PANEL IMAGE
            </div>
            <div style={{ fontSize: '.85rem', color: 'var(--text)', marginTop: '.1rem' }}>
              Upload your own file or click any sample preset below:
            </div>
          </div>

          {/* PRIMARY UPLOAD BUTTON - Prominently Placed */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="btn"
            id="btn-upload-file"
            style={{
              background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
              color: '#fff',
              fontWeight: 700,
              fontSize: '.9rem',
              padding: '.6rem 1.25rem',
              borderRadius: 'var(--radius)',
              boxShadow: '0 4px 14px rgba(37,99,235,0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: '.5rem',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            <span style={{ fontSize: '1.15rem' }}>📁</span>
            <span>Upload Image from Computer</span>
          </button>
        </div>

        {/* Quick Presets Row */}
        <div style={{ marginBottom: '1.1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.35rem' }}>
            <span style={{ fontSize: '.72rem', color: 'var(--dim)', fontWeight: 600 }}>OR SELECT QUICK PRESET TEST SAMPLES:</span>
            <span style={{ fontSize: '.7rem', color: 'var(--blue)' }}>PVEL-AD Electroluminescence Benchmark Set</span>
          </div>
          <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap' }}>
            {DEMO_PRESETS.map(p => (
              <button
                key={p.id}
                onClick={() => selectPreset(p)}
                className="btn"
                style={{
                  background: fileName === p.file ? 'rgba(59,130,246,0.25)' : 'rgba(255,255,255,0.04)',
                  border: `1px solid ${fileName === p.file ? 'var(--blue)' : 'var(--border)'}`,
                  color: fileName === p.file ? '#fff' : 'var(--muted)',
                  fontSize: '.74rem', padding: '.3rem .65rem', borderRadius: 999,
                  display: 'flex', alignItems: 'center', gap: '.3rem',
                  cursor: 'pointer'
                }}
              >
                <span>{p.icon}</span>
                <span>{p.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Active Image Preview & Action CTA Card */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(240px, 320px) 1fr',
          gap: '1.25rem',
          background: 'var(--bg)',
          borderRadius: 'var(--radius)',
          border: `2px dashed ${dragging ? 'var(--blue)' : 'rgba(255,255,255,0.1)'}`,
          padding: '1rem',
          alignItems: 'center'
        }}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          {/* Left: Image Container */}
          <div style={{ textAlign: 'center', position: 'relative' }}>
            <div style={{
              background: '#070B13',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
              padding: '.4rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 180,
              maxHeight: 200,
              overflow: 'hidden'
            }}>
              {preview ? (
                <img
                  src={preview}
                  alt="Active solar panel"
                  style={{ maxHeight: 180, maxWidth: '100%', objectFit: 'contain', borderRadius: 4 }}
                />
              ) : (
                <div style={{ padding: '2rem 1rem', color: 'var(--muted)' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '.5rem' }}>📷</div>
                  <div style={{ fontSize: '.8rem' }}>No image loaded</div>
                </div>
              )}
            </div>

            <div style={{
              marginTop: '.4rem',
              fontSize: '.74rem',
              color: 'var(--dim)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              📷 {fileName || 'No file selected'}
            </div>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn btn-outline"
              style={{
                marginTop: '.4rem',
                fontSize: '.72rem',
                padding: '.25rem .6rem',
                borderRadius: 999,
                width: '100%'
              }}
            >
              🔄 Change / Replace Image
            </button>
          </div>

          {/* Right: Inference Execution Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.8rem', justifyContent: 'center' }}>
            <div style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '.75rem .9rem'
            }}>
              <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'var(--dim)', textTransform: 'uppercase', marginBottom: '.4rem' }}>
                EXECUTION PIPELINE READY:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '.4rem', fontSize: '.75rem' }}>
                <div>
                  <span style={{ color: 'var(--muted)' }}>Architecture: </span>
                  <strong style={{ color: selectedModel === 'densenet121_gan' ? '#EC4899' : '#60A5FA' }}>
                    {getModelName(selectedModel)}
                  </strong>
                </div>
                <div>
                  <span style={{ color: 'var(--muted)' }}>Input Size: </span>
                  <strong style={{ color: '#fff' }}>224 × 224 RGB</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--muted)' }}>Latency Target: </span>
                  <strong style={{ color: 'var(--green)' }}>
                    {selectedModel === 'densenet121_gan' ? '~18.35 ms' : selectedModel === 'efficientnet_b0' ? '~14.80 ms' : selectedModel === 'resnet50' ? '~42.10 ms' : '~18.27 ms'}
                  </strong>
                </div>
              </div>
            </div>

            {/* BIG VIBRANT ACTION BUTTON TO RUN CLASSIFICATION */}
            <button
              onClick={handleAnalyze}
              id="btn-run-analysis"
              className="btn"
              style={{
                background: selectedModel === 'densenet121_gan'
                  ? 'linear-gradient(135deg, #EC4899 0%, #BE185D 100%)'
                  : selectedModel === 'densenet121'
                  ? 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)'
                  : selectedModel === 'efficientnet_b0'
                  ? 'linear-gradient(135deg, #10B981 0%, #047857 100%)'
                  : selectedModel === 'compare'
                  ? 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)'
                  : 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                color: '#fff',
                fontWeight: 800,
                fontSize: '1rem',
                padding: '.95rem 1.4rem',
                borderRadius: 'var(--radius)',
                border: 'none',
                cursor: 'pointer',
                boxShadow: selectedModel === 'densenet121_gan'
                  ? '0 6px 20px rgba(236,72,153,0.45)'
                  : '0 6px 20px rgba(59,130,246,0.45)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '.6rem',
                transition: 'transform .15s ease, box-shadow .15s ease'
              }}
            >
              <span>{getButtonLabel()}</span>
            </button>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '.72rem', color: 'var(--dim)' }}>
              <span>💡 You can also drag & drop files directly onto the preview</span>
              <span>Local API: http://localhost:5000</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
