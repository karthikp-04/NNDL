import { PRESET_SAMPLES } from '../../utils/constants'
export { PRESET_SAMPLES }

export default function PresetChips({ activeFile, onSelectPreset, presets = PRESET_SAMPLES }) {
  return (
    <div className="preset-chips-row" role="tablist" aria-label="Quick Test Presets">
      {presets.map((p) => {
        const isActive = activeFile === p.file
        const Icon = p.icon
        return (
          <button
            key={p.id}
            role="tab"
            aria-selected={isActive}
            className={`preset-chip ${isActive ? 'active' : ''}`}
            onClick={() => onSelectPreset(p)}
          >
            <Icon size={14} style={{ flexShrink: 0 }} />
            <span>{p.name}</span>
          </button>
        )
      })}
    </div>
  )
}
