import { useState } from 'react'
import { useApp } from '../context/AppContext'

function PanelView({ label, icon, children, badge }) {
  return (
    <div className="heatmap-card">
      <div className="heatmap-card-header" style={{ justifyContent:'space-between' }}>
        <span>{icon} {label}</span>
        {badge && (
          <span style={{
            fontSize:'.65rem', fontWeight:700, padding:'2px 8px',
            borderRadius:'999px', background:'var(--blue-dim)',
            color:'var(--blue)', border:'1px solid rgba(59,130,246,.3)'
          }}>{badge}</span>
        )}
      </div>
      {children}
    </div>
  )
}

export default function HeatmapPage() {
  const { analysisState, navigate } = useApp()
  const image = analysisState?.image
  const [opacity, setOpacity] = useState(0.65)
  const [activeLayer, setActiveLayer] = useState('overlay')

  // Heatmap color stops
  const heatmapGrad = `linear-gradient(
    135deg,
    rgba(59,130,246,0.1) 0%,
    rgba(34,197,94,0.3) 20%,
    rgba(245,158,11,0.7) 45%,
    rgba(239,68,68,0.9) 70%,
    rgba(139,92,246,0.5) 100%
  )`

  return (
    <div>
      {/* Explanation banner */}
      <div style={{
        background:'linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.08) 100%)',
        border:'1px solid var(--border)',
        borderRadius:'var(--radius)',
        padding:'1.25rem 1.5rem',
        marginBottom:'1.5rem',
        display:'flex', alignItems:'center', gap:'1.2rem'
      }}>
        <div style={{
          width:48, height:48, borderRadius:12, flexShrink:0,
          background:'linear-gradient(135deg, var(--orange), #fbbf24)',
          display:'flex', alignItems:'center', justifyContent:'center',
          fontSize:'1.5rem', boxShadow:'0 0 20px rgba(245,158,11,0.3)'
        }}>🔥</div>
        <div>
          <div style={{ fontWeight:700, marginBottom:'.2rem' }}>Grad-CAM Visual Explanation</div>
          <p style={{ margin:0, fontSize:'.85rem' }}>
            Gradient-weighted Class Activation Mapping highlights the regions the CNN focused on to make
            its prediction. <span style={{ color:'var(--red)' }}>Red/Orange</span> = high attention,{' '}
            <span style={{ color:'var(--blue)' }}>Blue</span> = low attention.
          </p>
        </div>
      </div>

      {/* Three-panel heatmap grid */}
      <div className="heatmap-grid" style={{ marginBottom:'1.5rem' }}>

        {/* Original */}
        <PanelView label="Original Image" icon="📷" badge="Input">
          <div className="heatmap-img-wrap" style={{ height:260, background:'var(--bg3)' }}>
            {image ? (
              <img src={image} alt="Original solar panel"
                style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }} />
            ) : (
              <div style={{ height:'100%', display:'flex', alignItems:'center',
                justifyContent:'center', fontSize:'4rem', color:'var(--dim)' }}>☀️</div>
            )}
          </div>
          <div className="heatmap-legend" style={{ borderTop:'1px solid var(--border)' }}>
            <span style={{ fontSize:'.72rem', color:'var(--dim)' }}>Raw input to CNN</span>
          </div>
          <div className="cam-panel-footer" />
        </PanelView>

        {/* Grad-CAM heatmap */}
        <PanelView label="Grad-CAM Heatmap" icon="🌡️" badge="Attention Map">
          <div className="heatmap-img-wrap" style={{ height:260, background:'#0a0a0a' }}>
            {image ? (
              <>
                <img src={image} alt="Heatmap base"
                  style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', filter:'grayscale(1) brightness(0.3)' }} />
                <div style={{
                  position:'absolute', inset:0,
                  background: heatmapGrad,
                  mixBlendMode:'screen'
                }} />
                {/* Annotation boxes */}
                <div className="heat-annotation ha1">🔴 Crack</div>
                <div className="heat-annotation ha2">🟠 Hotspot</div>
              </>
            ) : (
              <div style={{
                height:'100%', display:'flex', alignItems:'center',
                justifyContent:'center', flexDirection:'column', gap:'.5rem'
              }}>
                <span style={{ fontSize:'2.5rem' }}>🌡️</span>
                <span style={{ fontSize:'.8rem', color:'var(--dim)' }}>No image loaded</span>
              </div>
            )}
          </div>
          <div className="heatmap-legend" style={{ borderTop:'1px solid var(--border)' }}>
            <div className="legend-bar" />
            <div className="legend-labels" style={{ minWidth:120 }}>
              <span>Low</span><span>High</span>
            </div>
          </div>
          <div className="cam-panel-footer" />
        </PanelView>

        {/* Overlay */}
        <PanelView label="Overlay" icon="🔀" badge="Combined">
          <div className="heatmap-img-wrap" style={{ height:260, background:'var(--bg3)' }}>
            {image ? (
              <>
                <img src={image} alt="Panel overlay"
                  style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }} />
                <div style={{
                  position:'absolute', inset:0,
                  background: heatmapGrad,
                  opacity,
                  mixBlendMode:'hard-light'
                }} />
                <div className="heat-annotation ha1">🔴 Crack</div>
              </>
            ) : (
              <div style={{ height:'100%', display:'flex', alignItems:'center',
                justifyContent:'center', fontSize:'4rem', color:'var(--dim)' }}>🔀</div>
            )}
          </div>
          <div className="heatmap-legend" style={{ borderTop:'1px solid var(--border)' }}>
            <div style={{ fontSize:'.72rem', color:'var(--dim)' }}>Opacity</div>
            <input
              type="range" min="0.2" max="1" step="0.05"
              value={opacity}
              onChange={e => setOpacity(Number(e.target.value))}
              style={{ flex:1, accentColor:'var(--blue)' }}
            />
            <span style={{ fontSize:'.72rem', color:'var(--muted)', fontFamily:'JetBrains Mono, monospace' }}>
              {Math.round(opacity * 100)}%
            </span>
          </div>
          <div className="cam-panel-footer" />
        </PanelView>
      </div>

      {/* Layer selector + CNN info */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem' }}>

        {/* Layer activation selector */}
        <div className="card">
          <div className="card-title" style={{ marginBottom:'1rem' }}>🧠 CNN Layer Activations</div>
          <p style={{ fontSize:'.85rem', marginBottom:'1rem' }}>
            Select which convolutional layer's gradient to visualize. Deeper layers capture high-level
            semantic features; shallow layers capture texture.
          </p>
          <div style={{ display:'flex', flexDirection:'column', gap:'.5rem' }}>
            {[
              { id:'conv1', label:'Conv Layer 1', desc:'Edge & texture features', depth:'Shallow' },
              { id:'layer2', label:'ResNet Block 2', desc:'Shape & structural patterns', depth:'Mid' },
              { id:'layer3', label:'ResNet Block 3', desc:'Defect-specific activations', depth:'Deep' },
              { id:'overlay', label:'Final Conv Layer', desc:'High-level semantic regions', depth:'Deepest', recommended:true },
            ].map(l => (
              <button
                key={l.id}
                onClick={() => setActiveLayer(l.id)}
                style={{
                  display:'flex', alignItems:'center', gap:'.8rem',
                  padding:'.75rem 1rem', borderRadius:'var(--radius-sm)',
                  border:`1px solid ${activeLayer === l.id ? 'var(--blue)' : 'var(--border)'}`,
                  background: activeLayer === l.id ? 'var(--blue-dim)' : 'rgba(255,255,255,0.02)',
                  cursor:'pointer', textAlign:'left', width:'100%',
                  transition:'all .2s'
                }}
              >
                <div style={{
                  width:10, height:10, borderRadius:'50%',
                  background: activeLayer === l.id ? 'var(--blue)' : 'var(--dim)',
                  flexShrink:0
                }} />
                <div style={{ flex:1 }}>
                  <div style={{ fontSize:'.85rem', fontWeight:600, color: activeLayer === l.id ? 'var(--blue)' : 'var(--text)' }}>
                    {l.label}
                    {l.recommended && (
                      <span style={{
                        fontSize:'.65rem', marginLeft:'.5rem', padding:'1px 6px',
                        borderRadius:'999px', background:'var(--green-dim)', color:'var(--green)'
                      }}>Recommended</span>
                    )}
                  </div>
                  <div style={{ fontSize:'.72rem', color:'var(--dim)' }}>{l.desc}</div>
                </div>
                <span style={{
                  fontSize:'.65rem', fontWeight:700, padding:'2px 7px',
                  borderRadius:'999px',
                  background: l.depth === 'Deepest' ? 'var(--orange-dim)' : l.depth === 'Deep' ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.05)',
                  color: l.depth === 'Deepest' ? 'var(--orange)' : l.depth === 'Deep' ? 'var(--purple)' : 'var(--dim)',
                }}>{l.depth}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Model focus summary */}
        <div className="card">
          <div className="card-title" style={{ marginBottom:'1rem' }}>📍 Focus Region Summary</div>
          <div style={{ display:'flex', flexDirection:'column', gap:'.75rem' }}>
            {[
              { region:'Top-left quadrant',     intensity:92, color:'var(--red)',    note:'Primary crack zone' },
              { region:'Center area',            intensity:71, color:'var(--orange)', note:'Thermal hotspot' },
              { region:'Bottom-right quadrant',  intensity:48, color:'var(--blue)',   note:'Minor discoloration' },
              { region:'Edges',                  intensity:22, color:'var(--green)',  note:'Normal — low attention' },
            ].map(r => (
              <div key={r.region}>
                <div style={{ display:'flex', justifyContent:'space-between', fontSize:'.8rem', marginBottom:'.25rem' }}>
                  <span style={{ color:'var(--muted)', fontWeight:500 }}>{r.region}</span>
                  <span style={{ color: r.color, fontWeight:700, fontFamily:'JetBrains Mono, monospace' }}>{r.intensity}%</span>
                </div>
                <div className="progress-track" style={{ height:6, marginBottom:'.2rem' }}>
                  <div style={{
                    height:'100%', width:`${r.intensity}%`,
                    borderRadius:'999px',
                    background: r.color,
                    transition:'width 1s ease'
                  }} />
                </div>
                <div style={{ fontSize:'.7rem', color:'var(--dim)' }}>{r.note}</div>
              </div>
            ))}
          </div>

          <hr className="divider" />

          <div style={{
            background:'rgba(239,68,68,0.05)', border:'1px solid rgba(239,68,68,0.2)',
            borderRadius:'var(--radius-sm)', padding:'1rem'
          }}>
            <div style={{ fontSize:'.82rem', fontWeight:700, color:'var(--red)', marginBottom:'.4rem' }}>
              ⚠️ CNN Attention Summary
            </div>
            <p style={{ fontSize:'.78rem', margin:0, lineHeight:1.6 }}>
              The model detected high activation in the top-left quadrant consistent with
              micro-crack patterns. This region contributed most to the defect classification decision.
            </p>
          </div>

          <div style={{ display:'flex', gap:'.6rem', marginTop:'1rem' }}>
            <button className="btn btn-primary" style={{ flex:1, justifyContent:'center', fontSize:'.82rem' }}
              onClick={() => navigate('results')}>
              ← Back to Results
            </button>
            <button className="btn btn-outline" style={{ flex:1, justifyContent:'center', fontSize:'.82rem' }}
              onClick={() => navigate('upload')}>
              🔄 New Upload
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
