import { useApp } from '../context/AppContext'

const navItems = [
  { id: 'upload',    icon: '📤', label: 'Upload & Test',   section: 'ANALYSIS',   requiresResult: false },
  { id: 'results',   icon: '📊', label: 'Results',         section: null,         requiresResult: true  },
  { id: 'heatmap',   icon: '🔥', label: 'Grad-CAM',        section: null,         requiresResult: true  },
  { id: 'compare',   icon: '⚖️', label: 'Model Benchmarks', section: 'BENCHMARKS', requiresResult: false },
  { id: 'dashboard', icon: '📈', label: 'Dashboard',       section: 'MONITORING', requiresResult: false },
  { id: 'history',   icon: '📜', label: 'History',         section: null,         requiresResult: false },
  { id: 'about',     icon: 'ℹ️', label: 'About',            section: 'INFO',       requiresResult: false },
]

export default function Sidebar({ activePage, navigate, historyCount, hasResult }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="brand">
          <div className="sun-icon">☀️</div>
          <div>
            <div>SolarGuard <em style={{ fontStyle:'italic', color:'#F59E0B' }}>AI</em></div>
            <div className="tagline">See decay before it costs you power</div>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item, i) => {
          const prevItem = navItems[i - 1]
          const showSection = item.section && (!prevItem || prevItem.section !== item.section)
          const locked = item.requiresResult && !hasResult
          return (
            <div key={item.id}>
              {showSection && (
                <div className="nav-section-title">{item.section}</div>
              )}
              <button
                className={`nav-item ${activePage === item.id ? 'active' : ''} ${locked ? 'nav-locked' : ''}`}
                onClick={() => {
                  if (locked) return
                  navigate(item.id)
                }}
                id={`nav-${item.id}`}
                title={locked ? 'Upload and analyze an image first' : item.label}
                style={{ opacity: locked ? 0.4 : 1, cursor: locked ? 'not-allowed' : 'pointer' }}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
                {item.id === 'history' && historyCount > 0 && (
                  <span className="nav-badge">{historyCount}</span>
                )}
                {locked && (
                  <span style={{
                    marginLeft: 'auto', fontSize: '.62rem', color: 'var(--dim)',
                    fontWeight: 600, letterSpacing: '.04em'
                  }}>🔒</span>
                )}
              </button>
            </div>
          )
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status">
          <div className="status-dot"></div>
          <span>3 CNN Models Ready · PVEL-AD v2</span>
        </div>
      </div>
    </aside>
  )
}
