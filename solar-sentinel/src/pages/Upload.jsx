import { useState, useRef, useCallback } from 'react'
import {
  UploadCloud,
  FileImage,
  Play,
  RefreshCw,
  ArrowRight
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import { DEMO_SOLAR_PANEL } from '../utils/constants'
import ModelSelector, { MODEL_CONFIGS } from '../components/ui/ModelSelector'
import PresetChips, { PRESET_SAMPLES } from '../components/ui/PresetChips'
import Badge from '../components/ui/Badge'

// Demo data mapping with SVGs
const PRESET_DATA = {
  crack: DEMO_SOLAR_PANEL,
  black_core: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F1012"/><rect x="10" y="10" width="780" height="480" fill="%2317191D" rx="8" stroke="%232A2D33" stroke-width="2"/><g stroke="%232A2D33" stroke-width="1.5"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><rect x="275" y="135" width="250" height="230" fill="%23050505" stroke="%23E5484D" stroke-width="2" stroke-dasharray="4"/><text x="290" y="260" font-family="monospace" font-size="16" fill="%23E5484D">INACTIVE CORE (BLACK)</text></svg>`,
  finger: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F1012"/><rect x="10" y="10" width="780" height="480" fill="%2317191D" rx="8" stroke="%232A2D33" stroke-width="2"/><g stroke="%232A2D33" stroke-width="1.5"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><line x1="140" y1="200" x2="400" y2="200" stroke="%23E8A33D" stroke-width="4" stroke-dasharray="8,6"/><text x="150" y="190" font-family="monospace" font-size="13" fill="%23E8A33D">BROKEN BUSBAR FINGER</text></svg>`,
  dislocation: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F1012"/><rect x="10" y="10" width="780" height="480" fill="%2317191D" rx="8" stroke="%232A2D33" stroke-width="2"/><g stroke="%232A2D33" stroke-width="1.5"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><path d="M 420 150 Q 550 250 630 350 Q 480 320 420 150" fill="%238E9097" fill-opacity="0.2" stroke="%23ECE9E2" stroke-width="1.5"/><text x="440" y="270" font-family="monospace" font-size="14" fill="%23ECE9E2">DISLOCATION BOUNDARY</text></svg>`,
  healthy: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F1012"/><rect x="10" y="10" width="780" height="480" fill="%2317191D" rx="8" stroke="%232A2D33" stroke-width="2"/><g stroke="%232A2D33" stroke-width="1.5"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><text x="260" y="250" font-family="monospace" font-size="16" fill="%233FB68B">OPTIMAL EL LUMINESCENCE (HEALTHY)</text></svg>`,
  gan_sample: '/gan_samples/synthetic_star_crack/synthetic_cgan_star_crack_0001.png'
}

export default function Upload() {
  const { navigate, setAnalysis, addToast, selectedModel, setSelectedModel } = useApp()
  const [dragging, setDragging]   = useState(false)
  const [preview, setPreview]     = useState(DEMO_SOLAR_PANEL)
  const [fileName, setFileName]   = useState(PRESET_SAMPLES[0].file)
  const fileInputRef              = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      addToast('Please upload an image file (.jpg, .png, .tif)', 'error')
      return
    }
    setFileName(f.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
    addToast(`Loaded ${f.name}`, 'info')
  }

  const selectPreset = (p) => {
    const data = PRESET_DATA[p.id] || p.file
    setPreview(data)
    setFileName(p.file)
    addToast(`Selected sample preset: ${p.name}`, 'info')
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

  const activeModelMeta = MODEL_CONFIGS.find(m => m.id === selectedModel) || MODEL_CONFIGS[0]

  return (
    <div style={{ maxWidth: 1260, margin: '0 auto' }}>
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => handleFile(e.target.files[0])}
        id="file-input"
      />

      {/* Hero Header */}
      <div style={{ marginBottom: '1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              INSPECTION WORKBENCH
            </span>
            <Badge variant="outline">PVEL-AD Standard</Badge>
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700 }}>
            Solar Panel Defect Classification
          </h1>
          <p style={{ fontSize: '0.88rem', marginTop: '0.2rem' }}>
            Select an active neural architecture, load high-resolution Electroluminescence (EL) imagery, and execute inference.
          </p>
        </div>

        <button
          className="btn btn-outline btn-sm"
          onClick={() => fileInputRef.current?.click()}
          id="btn-upload-file"
        >
          <UploadCloud size={15} />
          <span>Browse Image File</span>
        </button>
      </div>

      {/* Step 1: Model Selection Component */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Step 1 · Active Architecture
          </span>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            Selected: <strong style={{ color: 'var(--accent)' }}>{activeModelMeta.name}</strong>
          </span>
        </div>
        <ModelSelector
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
        />
      </div>

      {/* Step 2: Upload Dropzone & Inspection Pipeline */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Step 2 · Target Panel Imagery
          </span>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)' }}>
            Drag and drop or pick quick sample
          </span>
        </div>

        <div
          className={`upload-preview-container ${dragging ? 'drag-over' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          {/* Left: Image Preview Box */}
          <div>
            <div className="preview-media-box">
              {preview ? (
                <img src={preview} alt="Active solar cell test panel" />
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
                  <FileImage size={36} style={{ margin: '0 auto 0.5rem', opacity: 0.5 }} />
                  <div style={{ fontSize: '0.82rem' }}>No image loaded</div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', fontSize: '0.72rem' }}>
              <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 220 }}>
                {fileName || 'test_panel.jpg'}
              </span>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => fileInputRef.current?.click()}
                style={{ fontSize: '0.72rem', padding: '0.2rem 0.5rem' }}
              >
                <RefreshCw size={12} />
                <span>Replace</span>
              </button>
            </div>
          </div>

          {/* Right: Pipeline Summary & Action CTA */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center' }}>
            <div className="card-raised card" style={{ padding: '1rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                Execution Pipeline Specifications
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem 1rem', fontSize: '0.78rem' }}>
                <div>
                  <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>Model Architecture</div>
                  <div style={{ fontWeight: 600, color: 'var(--text)', marginTop: '2px' }}>
                    {activeModelMeta.name}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>Expected Latency</div>
                  <div className="font-mono" style={{ color: 'var(--accent)', fontWeight: 600, marginTop: '2px' }}>
                    {activeModelMeta.latency}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>Resolution</div>
                  <div className="font-mono" style={{ color: 'var(--text)', marginTop: '2px' }}>
                    224 × 224 RGB
                  </div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>Inference Engine</div>
                  <div style={{ color: 'var(--text)', marginTop: '2px' }}>
                    PyTorch (Local:5000)
                  </div>
                </div>
              </div>
            </div>

            {/* PRIMARY AMBER CTA BUTTON */}
            <button
              id="btn-run-analysis"
              className="btn btn-primary btn-lg"
              onClick={handleAnalyze}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              <Play size={16} fill="currentColor" />
              <span>Run {activeModelMeta.name} Inference</span>
              <ArrowRight size={16} />
            </button>

            <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', textAlign: 'center' }}>
              Drag & drop any solar EL, RGB, or infrared panel file (.jpg, .png, .tif) onto the panel
            </div>
          </div>
        </div>
      </div>

      {/* Step 3: Quick Presets */}
      <div className="card" style={{ padding: '0.85rem 1.15rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            QUICK TEST BENCHMARK PRESETS
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            Click any sample to instantly load
          </span>
        </div>
        <PresetChips activeFile={fileName} onSelectPreset={selectPreset} />
      </div>
    </div>
  )
}
