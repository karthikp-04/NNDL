import { useState, useRef, useCallback } from 'react'
import { useApp } from '../context/AppContext'

export default function Upload() {
  const { navigate, setAnalysis, addToast } = useApp()
  const [dragging, setDragging]   = useState(false)
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [mode, setMode]           = useState('single') // single | batch
  const fileInputRef              = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) { addToast('Please upload an image file', 'error'); return }
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
    addToast('Image loaded successfully', 'success')
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const handleAnalyze = () => {
    if (!file && !preview) {
      addToast('Please upload or select an image first', 'error')
      return
    }
    setAnalysis({ image: preview, fileName: file?.name || 'demo_panel.jpg' })
    navigate('analyzing')
  }

  return (
    <div>
      <div className="upload-hero">
        <h1>
          <span className="gradient-text">Detect Solar Defects</span><br />
          Before They Cost You Power
        </h1>
        <p style={{ color: 'var(--muted)', marginTop: '.6rem' }}>
          Upload an electroluminescence, RGB, or infrared image of a solar panel.
          Our CNN analyzes it instantly for cracks, hotspots, and other anomalies.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex items-center gap-3 mb-4" style={{ justifyContent:'center', marginBottom:'1.5rem' }}>
        <button
          onClick={() => setMode('single')}
          className="btn"
          style={{
            background: mode === 'single' ? 'var(--blue)' : 'transparent',
            color: mode === 'single' ? '#fff' : 'var(--muted)',
            border: `1px solid ${mode === 'single' ? 'var(--blue)' : 'var(--border)'}`,
            padding: '.5rem 1.4rem', borderRadius: '999px'
          }}
        >📄 Single Image</button>
        <button
          onClick={() => setMode('batch')}
          className="btn"
          style={{
            background: mode === 'batch' ? 'var(--blue)' : 'transparent',
            color: mode === 'batch' ? '#fff' : 'var(--muted)',
            border: `1px solid ${mode === 'batch' ? 'var(--blue)' : 'var(--border)'}`,
            padding: '.5rem 1.4rem', borderRadius: '999px'
          }}
        >📁 Batch Upload</button>
      </div>

      {/* Upload zone */}
      <div
        className={`upload-zone ${dragging ? 'drag-over' : ''} ${preview ? 'has-file' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !preview && fileInputRef.current?.click()}
        style={{ maxWidth: 700, margin: '0 auto', cursor: preview ? 'default' : 'pointer' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
          multiple={mode === 'batch'}
          id="file-input"
        />

        {!preview ? (
          <>
            <div className="upload-icon">
              {dragging ? '📂' : '☁️'}
            </div>
            <h3>{dragging ? 'Drop to upload' : 'Drag & drop your image here'}</h3>
            <p>or click to browse files · Max 50MB per image</p>
            <div className="upload-badges">
              <div className="upload-badge el">⚡ EL Images</div>
              <div className="upload-badge rgb">🎨 RGB Images</div>
              <div className="upload-badge ir">🌡️ Infrared</div>
            </div>
          </>
        ) : (
          <div style={{ width: '100%' }}>
            <div className="preview-grid">
              <div className="preview-item">
                <img src={preview} alt="Uploaded panel" style={{ maxHeight: 280, width:'100%', objectFit:'cover', borderRadius:'var(--radius)' }} />
                <div className="preview-label">📷 {file?.name || 'uploaded_image.jpg'}</div>
              </div>
              <div style={{ display:'flex', flexDirection:'column', gap:'1rem', justifyContent:'center', padding:'1rem' }}>
                <div className="card" style={{ padding:'1.2rem' }}>
                  <div className="card-title" style={{ marginBottom:'.8rem' }}>📋 Image Info</div>
                  <div style={{ display:'flex', flexDirection:'column', gap:'.5rem' }}>
                    <InfoRow label="File" value={file?.name || 'demo_panel.jpg'} />
                    <InfoRow label="Size" value={file?.size ? `${(file.size/1024).toFixed(1)} KB` : '200 KB'} />
                    <InfoRow label="Type" value="EL Image (Grayscale)" />
                    <InfoRow label="Status" value="✅ Ready to analyze" color="var(--green)" />
                  </div>
                </div>
                <div style={{ display:'flex', gap:'.6rem' }}>
                  <button className="btn btn-outline" style={{ flex:1 }}
                    onClick={() => { setFile(null); setPreview(null) }}>
                    ✕ Remove
                  </button>
                  <button className="btn btn-outline" style={{ flex:1 }}
                    onClick={() => fileInputRef.current?.click()}>
                    🔄 Change
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Analyze button */}
      <div style={{ textAlign:'center', marginTop:'2rem', display:'flex', flexDirection:'column', alignItems:'center', gap:'.75rem' }}>
        {!preview && (
          <div style={{
            display:'flex', alignItems:'center', gap:'.5rem',
            fontSize:'.82rem', color:'var(--orange)',
            background:'rgba(245,158,11,0.08)', border:'1px solid rgba(245,158,11,0.25)',
            padding:'.5rem 1.2rem', borderRadius:'999px'
          }}>
            ⚠️ Upload an image to enable analysis
          </div>
        )}
        <div style={{ display:'flex', gap:'1rem', justifyContent:'center', flexWrap:'wrap' }}>
          <button
            className={`btn btn-lg ${preview ? 'btn-green' : 'btn-outline'}`}
            onClick={handleAnalyze}
            id="analyze-btn"
            disabled={!preview}
            title={!preview ? 'Upload an image first' : 'Run CNN analysis'}
            style={!preview ? { opacity:0.45, cursor:'not-allowed', transform:'none' } : {}}
          >
            🔍 Analyze Panel
          </button>
        </div>
      </div>

      {/* Feature strip */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:'1rem', marginTop:'3rem', maxWidth:800, margin:'3rem auto 0' }}>
        {[
          { icon:'🧠', title:'DCNN Model', sub:'ResNet-50 backbone' },
          { icon:'🎯', title:'10 Defect Types', sub:'PVEL-AD dataset' },
          { icon:'🔥', title:'Grad-CAM', sub:'Visual explainability' },
          { icon:'⚡', title:'< 2s Inference', sub:'Real-time detection' },
        ].map(f => (
          <div key={f.title} className="card" style={{ textAlign:'center', padding:'1.2rem .8rem' }}>
            <div style={{ fontSize:'1.8rem', marginBottom:'.4rem' }}>{f.icon}</div>
            <div style={{ fontSize:'.82rem', fontWeight:700, color:'var(--text)', marginBottom:'.2rem' }}>{f.title}</div>
            <div style={{ fontSize:'.72rem', color:'var(--dim)' }}>{f.sub}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function InfoRow({ label, value, color }) {
  return (
    <div style={{ display:'flex', justifyContent:'space-between', fontSize:'.82rem' }}>
      <span style={{ color:'var(--dim)' }}>{label}</span>
      <span style={{ color: color || 'var(--text)', fontWeight:500 }}>{value}</span>
    </div>
  )
}
