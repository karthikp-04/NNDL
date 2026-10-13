import { MODEL_CONFIGS } from '../../utils/constants'
export { MODEL_CONFIGS }

export default function ModelSelector({ selectedModel, onSelectModel }) {
  return (
    <div
      className="model-grid"
      role="radiogroup"
      aria-label="Deep Learning Model Architecture Selection"
    >
      {MODEL_CONFIGS.map((m) => {
        const isSelected = selectedModel === m.id
        const IconComponent = m.icon
        return (
          <div
            key={m.id}
            role="radio"
            aria-checked={isSelected}
            tabIndex={0}
            className={`model-card ${isSelected ? 'selected' : ''}`}
            onClick={() => onSelectModel(m.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelectModel(m.id)
              }
            }}
          >
            <div>
              <div className="model-card-topbar">
                <div className="model-card-icon-pill">
                  <IconComponent
                    size={16}
                    style={{
                      color: isSelected ? 'var(--accent)' : 'var(--text-muted)',
                      flexShrink: 0
                    }}
                  />
                </div>
                <span className="badge badge-outline" style={{ fontSize: '0.64rem', padding: '0.15rem 0.5rem', whiteSpace: 'nowrap' }}>
                  {m.badge}
                </span>
              </div>
              <div className="model-card-title">{m.name}</div>
              <div className="model-card-desc">{m.desc}</div>
            </div>

            <div className="model-card-stats">
              <div className="model-stat-row">
                <span className="model-stat-label">Latency</span>
                <span className="model-stat-value">{m.latency}</span>
              </div>
              <div className="model-stat-row">
                <span className="model-stat-label">{m.metricLabel}</span>
                <span className="model-stat-value">{m.metricValue}</span>
              </div>
              <div className="model-stat-row">
                <span className="model-stat-label">Parameters</span>
                <span className="model-stat-value">{m.params}</span>
              </div>
              <div className="model-stat-row">
                <span className="model-stat-label">Input Size</span>
                <span className="model-stat-value">224 × 224</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
