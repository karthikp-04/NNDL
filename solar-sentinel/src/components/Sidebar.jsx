import {
  UploadCloud,
  FileCheck2,
  Flame,
  Scale,
  FlaskConical,
  LayoutDashboard,
  History as HistoryIcon,
  Info,
  Lock,
  ChevronLeft,
  ChevronRight,
  SunMedium
} from 'lucide-react'

const navItems = [
  { id: 'upload',     icon: UploadCloud,   label: 'Upload & Test',    section: 'ANALYSIS',       requiresResult: false },
  { id: 'results',    icon: FileCheck2,    label: 'Results',          section: null,             requiresResult: true  },
  { id: 'heatmap',    icon: Flame,         label: 'Grad-CAM',         section: null,             requiresResult: true  },
  { id: 'compare',    icon: Scale,         label: 'Model Benchmarks', section: 'BENCHMARKS',     requiresResult: false },
  { id: 'gan-studio', icon: FlaskConical,  label: 'cGAN Studio',      section: 'GENERATIVE AI',  requiresResult: false },
  { id: 'dashboard',  icon: LayoutDashboard, label: 'Dashboard',      section: 'MONITORING',     requiresResult: false },
  { id: 'history',    icon: HistoryIcon,   label: 'History',          section: null,             requiresResult: false },
  { id: 'about',      icon: Info,          label: 'About',            section: 'INFO',           requiresResult: false },
]

export default function Sidebar({ activePage, navigate, historyCount, hasResult, isCollapsed, onToggleCollapse }) {
  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`} aria-label="Main Navigation">
      <div className="sidebar-logo">
        <div className="brand">
          <div className="logo-mark" title="SolarGuard AI">
            <SunMedium size={18} strokeWidth={2.2} />
          </div>
          {!isCollapsed && (
            <div className="brand-text">
              <div className="brand-title">SolarGuard <span>AI</span></div>
              <div className="brand-sub">PV Defect Diagnostics</div>
            </div>
          )}
        </div>
        <button
          className="collapse-btn"
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item, i) => {
          const prevItem = navItems[i - 1]
          const showSection = item.section && (!prevItem || prevItem.section !== item.section)
          const locked = item.requiresResult && !hasResult
          const Icon = item.icon

          return (
            <div key={item.id}>
              {showSection && !isCollapsed && (
                <div className="nav-section-title">{item.section}</div>
              )}
              <button
                className={`nav-item ${activePage === item.id ? 'active' : ''} ${locked ? 'nav-locked' : ''}`}
                onClick={() => {
                  if (locked) return
                  navigate(item.id)
                }}
                id={`nav-${item.id}`}
                title={locked ? 'Run inference on an image first' : item.label}
                aria-disabled={locked}
              >
                <span className="nav-icon">
                  <Icon size={17} strokeWidth={1.9} />
                </span>
                {!isCollapsed && (
                  <>
                    <span className="nav-label">{item.label}</span>
                    {item.id === 'history' && historyCount > 0 && (
                      <span className="nav-badge">{historyCount}</span>
                    )}
                    {locked && (
                      <span title="Locked until analysis completes" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}>
                        <Lock size={12} color="var(--text-dim)" />
                      </span>
                    )}
                  </>
                )}
              </button>
            </div>
          )
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status" title="4 CNN & cGAN Architectures · PVEL-AD Dataset">
          <span className="status-dot online"></span>
          {!isCollapsed && <span>4 DL Models · PVEL-AD</span>}
        </div>
      </div>
    </aside>
  )
}
