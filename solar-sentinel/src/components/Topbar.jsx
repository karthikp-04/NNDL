import { useApp } from '../context/AppContext'

export default function Topbar({ page }) {
  const { selectedModel, setSelectedModel, addToast } = useApp()

  const handleModelChange = (e) => {
    const val = e.target.value
    setSelectedModel(val)
    const label = val === 'densenet121' ? 'DenseNet-121' : val === 'resnet50' ? 'ResNet-50' : 'Dual Model Compare'
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
            value={selectedModel || 'densenet121'}
            onChange={handleModelChange}
            id="topbar-model-select"
            style={{
              background: 'transparent',
              color: selectedModel === 'densenet121' ? '#60A5FA' : selectedModel === 'resnet50' ? '#A78BFA' : '#F59E0B',
              border: 'none',
              fontWeight: 700,
              cursor: 'pointer',
              outline: 'none',
              fontSize: '.78rem'
            }}
          >
            <option value="densenet121" style={{ background: '#1E293B', color: '#fff' }}>
              DenseNet-121
            </option>
            <option value="resnet50" style={{ background: '#1E293B', color: '#fff' }}>
              ResNet-50
            </option>
            <option value="compare" style={{ background: '#1E293B', color: '#fff' }}>
              Dual Model Compare
            </option>
          </select>
        </div>

        <div className="topbar-badge" title="Backend: http://localhost:5000">
          <span className="status-dot" style={{ width: 6, height: 6 }}></span>
          Model Online
        </div>
        <div className="topbar-avatar" title="Madhav (CSE23661)">M</div>
      </div>
    </header>
  )
}
