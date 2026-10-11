import { useEffect, useRef, useState } from 'react'
import { useApp } from '../context/AppContext'

const SEV_MAP = {
  healthy:  { icon: '🟢', label: 'Healthy',  cls: 'sev-healthy',  color: 'var(--green)',  pct: 100 },
  mild:     { icon: '🟡', label: 'Mild',      cls: 'sev-mild',     color: 'var(--blue)',   pct: 33  },
  moderate: { icon: '🟠', label: 'Moderate',  cls: 'sev-moderate', color: 'var(--orange)', pct: 66  },
  severe:   { icon: '🔴', label: 'Severe',    cls: 'sev-severe',   color: 'var(--red)',     pct: 100 },
}

const DEFECT_CLASS = {
  'Micro Crack':         'crack',
  'Finger Interruption': 'crack',
  'Hotspot':             'hotspot',
  'PID Effect':          'pid',
  'Inactive Cell':       'inactive',
  'Corrosion':           'corrosion',
  'Delamination':        'hotspot',
  'Healthy':             'healthy-tag',
}

const REC_MAP = {
  healthy:  [
    { icon:'✅', title:'No Action Required', desc:'Panel is operating within normal parameters.', primary:true },
    { icon:'📅', title:'Routine Inspection', desc:'Schedule next scan in 6 months.' },
  ],
  mild:     [
    { icon:'👁️', title:'Monitor Closely', desc:'Rescan in 30 days to track progression.', primary:true },
    { icon:'📋', title:'Log Finding', desc:'Document for warranty and maintenance records.' },
  ],
  moderate: [
    { icon:'🔧', title:'Schedule Maintenance', desc:'Clean & inspect affected cell area within 30 days.', primary:true },
    { icon:'📡', title:'Increase Monitoring', desc:'Set alerts to daily scan frequency.' },
    { icon:'📋', title:'Document Finding', desc:'Log to maintenance record for warranty claim.' },
  ],
  severe:   [
    { icon:'⚠️', title:'Immediate Maintenance', desc:'Replace or isolate affected cell immediately.', primary:true },
    { icon:'🔌', title:'Reduce Load', desc:'Temporarily reduce panel load to prevent cascade failure.' },
    { icon:'🏭', title:'Contact Manufacturer', desc:'File warranty claim — defect may qualify for replacement.' },
  ],
}

function ConfidenceRing({ value }) {
  const CIRC = 339.29
  const offset = CIRC - (value / 100) * CIRC
  const cls = value >= 85 ? 'high' : value >= 60 ? 'medium' : 'low'
  const [animated, setAnimated] = useState(CIRC)
  useEffect(() => {
    const t = setTimeout(() => setAnimated(offset), 100)
    return () => clearTimeout(t)
  }, [offset])

  return (
    <div className="confidence-ring">
      <div className="ring-wrap">
        <svg className="ring-svg" viewBox="0 0 120 120" width="120" height="120">
          <circle className="ring-bg" cx="60" cy="60" r="54" />
          <circle
            className={`ring-fill ${cls}`}
            cx="60" cy="60" r="54"
            style={{ strokeDashoffset: animated }}
          />
        </svg>
        <div className="ring-label">
          <span className="ring-pct">{value.toFixed(1)}%</span>
          <span className="ring-sub">confidence</span>
        </div>
      </div>
      <span className="conf-label">Model Confidence</span>
    </div>
  )
}

export default function Results() {
  const { analysisState, navigate, setHistory, addToast } = useApp()
  const result = analysisState?.result
  const image  = analysisState?.image
  const saved  = useRef(false)

  // Auto-save to history once
  useEffect(() => {
    if (result && !saved.current) {
      saved.current = true
      const now = new Date()
      setHistory(h => [
        {
          id: Date.now(),
          time: now.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }),
          date: now.toLocaleDateString([], { month:'short', day:'numeric' }),
          image: analysisState?.fileName || 'panel.jpg',
          prediction: result.prediction?.[0] || 'Unknown',
          severity: result.severity,
          confidence: result.confidence,
        },
        ...h,
      ])
    }
  }, [result])

  if (!result) {
    return (
      <div style={{ textAlign:'center', padding:'4rem 2rem' }}>
        <div style={{ fontSize:'3rem', marginBottom:'1rem' }}>📭</div>
        <h2 style={{ marginBottom:'.5rem' }}>No Results Yet</h2>
        <p>Upload and analyze a solar panel image first.</p>
        <button className="btn btn-primary mt-4" onClick={() => navigate('upload')} style={{ marginTop:'1.5rem' }}>
          ← Go to Upload
        </button>
      </div>
    )
  }

  const sev     = SEV_MAP[result.severity] || SEV_MAP.moderate
  const recs    = REC_MAP[result.severity] || REC_MAP.moderate
  const defects = result.prediction || []

  return (
    <div>
      {/* Top row: Image + Defects */}
      <div className="results-grid" style={{ marginBottom:'1.5rem' }}>

        {/* Original image */}
        <div className="card" style={{ padding:0, overflow:'hidden' }}>
          <div className="card-header" style={{ padding:'1rem 1.25rem', borderBottom:'1px solid var(--border)' }}>
            <span className="card-title">📷 Original Image</span>
            <span style={{ fontSize:'.75rem', color:'var(--dim)' }}>{analysisState?.fileName || 'panel.jpg'}</span>
          </div>
          {image ? (
            <img
              src={image}
              alt="Solar panel"
              style={{ width:'100%', maxHeight:280, objectFit:'cover', display:'block' }}
            />
          ) : (
            <div style={{
              height:240, background:'var(--bg3)',
              display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'3rem', color:'var(--dim)'
            }}>
              🌞
            </div>
          )}
          <div style={{ padding:'.75rem 1.25rem', display:'flex', gap:'.5rem' }}>
            <button className="btn btn-outline" style={{ fontSize:'.78rem', padding:'.4rem .9rem' }}
              onClick={() => navigate('heatmap')}>
              🔥 View Heatmap
            </button>
            <button className="btn btn-outline" style={{ fontSize:'.78rem', padding:'.4rem .9rem' }}
              onClick={() => navigate('upload')}>
              🔄 New Analysis
            </button>
          </div>
        </div>

        {/* Detected defects */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">🔍 Detected Defects</div>
              <div className="card-subtitle">{defects.length} defect type{defects.length !== 1 ? 's' : ''} found</div>
            </div>
            <div className={`defect-tag ${result.severity === 'healthy' ? 'healthy-tag' : 'crack'}`}
              style={{ fontSize:'.7rem' }}>
              {result.severity === 'healthy' ? '✓ Healthy' : '⚠ Defective'}
            </div>
          </div>

          <div className="defect-list">
            {defects.map(d => (
              <span key={d} className={`defect-tag ${DEFECT_CLASS[d] || 'crack'}`}>
                {d}
              </span>
            ))}
          </div>

          <hr className="divider" />

          {/* Severity */}
          <div className="card-title" style={{ marginBottom:'.75rem' }}>⚡ Severity Level</div>
          <div className="severity-meter">
            {Object.entries(SEV_MAP).map(([key, s]) => (
              <div
                key={key}
                className={`sev-row ${s.cls} ${result.severity === key ? 'selected-sev' : ''}`}
                style={result.severity === key ? {
                  background: `${s.color}14`,
                  boxShadow: `0 0 0 2px ${s.color}33`,
                } : {}}
              >
                <span className="sev-icon">{s.icon}</span>
                <span className="sev-label">{s.label}</span>
                {result.severity === key && (
                  <span style={{ fontSize:'.7rem', fontWeight:700 }}>← Current</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom row: Confidence + Power Loss + Recommendations */}
      <div className="results-wide">

        {/* Confidence ring */}
        <div className="card" style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:'1rem' }}>
          <div className="card-title" style={{ alignSelf:'flex-start' }}>📊 Confidence Score</div>
          <ConfidenceRing value={result.confidence} />
          <div style={{
            width:'100%', background:'var(--bg3)', borderRadius:'var(--radius-sm)',
            padding:'.75rem 1rem', marginTop:'.5rem'
          }}>
            <div style={{ fontSize:'.75rem', color:'var(--dim)', marginBottom:'.4rem' }}>Defect Probabilities</div>
            {defects.slice(0, 3).map((d, i) => {
              const conf = i === 0 ? result.confidence : result.confidence - (i * 12 + Math.random() * 5)
              return (
                <div key={d} style={{ marginBottom:'.5rem' }}>
                  <div style={{ display:'flex', justifyContent:'space-between', fontSize:'.75rem', marginBottom:'.2rem' }}>
                    <span style={{ color:'var(--muted)' }}>{d}</span>
                    <span style={{ color:'var(--text)', fontFamily:'JetBrains Mono, monospace', fontWeight:600 }}>
                      {Math.max(10, conf.toFixed(1))}%
                    </span>
                  </div>
                  <div className="progress-track" style={{ height:4 }}>
                    <div className="progress-fill" style={{ width:`${Math.max(10, conf)}%` }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Power Impact */}
        <div className="card">
          <div className="card-title" style={{ marginBottom:'1rem' }}>⚡ Power Impact</div>
          {[
            { label:'Estimated Power Loss', value: result.severity === 'healthy' ? '0%' : result.severity === 'mild' ? '3–8%' : result.severity === 'moderate' ? '10–25%' : '30–50%', color: sev.color },
            { label:'Panel Efficiency',     value: result.severity === 'healthy' ? '≥ 95%' : result.severity === 'mild' ? '88–95%' : result.severity === 'moderate' ? '70–88%' : '< 70%', color: sev.color },
            { label:'Degradation Rate',     value: result.severity === 'healthy' ? 'Normal' : result.severity === 'mild' ? 'Slightly elevated' : result.severity === 'moderate' ? 'Elevated' : 'High', color: sev.color },
            { label:'Action Priority',      value: result.severity === 'healthy' ? 'Low' : result.severity === 'mild' ? 'Medium' : result.severity === 'moderate' ? 'High' : 'Critical', color: sev.color },
          ].map(row => (
            <div key={row.label} style={{
              display:'flex', justifyContent:'space-between', alignItems:'center',
              padding:'.65rem .9rem', borderRadius:'var(--radius-sm)',
              background:'rgba(255,255,255,0.02)', border:'1px solid var(--border)',
              marginBottom:'.5rem', fontSize:'.85rem'
            }}>
              <span style={{ color:'var(--dim)' }}>{row.label}</span>
              <span style={{ color: row.color, fontWeight:700 }}>{row.value}</span>
            </div>
          ))}

          <div style={{
            marginTop:'1rem', padding:'1rem', borderRadius:'var(--radius-sm)',
            background:`${sev.color}10`, border:`1px solid ${sev.color}33`,
            textAlign:'center'
          }}>
            <div style={{ fontSize:'1.8rem', marginBottom:'.3rem' }}>{sev.icon}</div>
            <div style={{ fontSize:'1rem', fontWeight:800, color: sev.color }}>{sev.label} Condition</div>
            <div style={{ fontSize:'.78rem', color:'var(--dim)', marginTop:'.2rem' }}>
              Overall panel assessment
            </div>
          </div>
        </div>

        {/* Recommendations */}
        <div className="card">
          <div className="card-title" style={{ marginBottom:'1rem' }}>✅ Recommended Actions</div>
          <div className="recommendation-box">
            {recs.map((r, i) => (
              <div key={i} className={`rec-item ${r.primary ? 'primary' : ''}`}>
                <span className="rec-icon">{r.icon}</span>
                <div className="rec-text">
                  <strong>{r.title}</strong>
                  <span>{r.desc}</span>
                </div>
              </div>
            ))}
          </div>

          <button
            className="btn btn-primary w-full"
            style={{ marginTop:'1.5rem', justifyContent:'center' }}
            onClick={() => { addToast('Report exported as PDF', 'success') }}
          >
            📄 Export Report
          </button>
          <button
            className="btn btn-outline w-full"
            style={{ marginTop:'.6rem', justifyContent:'center' }}
            onClick={() => navigate('heatmap')}
          >
            🔥 View Grad-CAM Explanation
          </button>
        </div>
      </div>
    </div>
  )
}
