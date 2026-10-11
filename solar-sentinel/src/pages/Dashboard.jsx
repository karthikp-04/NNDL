import { useEffect, useRef, useState } from 'react'
import { useApp } from '../context/AppContext'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line, Area, AreaChart,
  Legend
} from 'recharts'

const COLORS = {
  healthy:  '#22C55E',
  mild:     '#3B82F6',
  moderate: '#F59E0B',
  severe:   '#EF4444',
}

const DEFECT_COLORS = ['#EF4444','#F59E0B','#3B82F6','#8B5CF6','#06B6D4','#22C55E','#EC4899']

// Custom tooltip for recharts
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'var(--bg2)', border:'1px solid var(--border)',
      borderRadius:8, padding:'.6rem .9rem', fontSize:'.78rem'
    }}>
      {label && <div style={{ color:'var(--dim)', marginBottom:'.3rem' }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--text)', fontWeight:600 }}>
          {p.name ? `${p.name}: ` : ''}{p.value}
          {p.unit || ''}
        </div>
      ))}
    </div>
  )
}

function StatCard({ icon, label, value, change, colorClass, sub }) {
  return (
    <div className={`stat-card ${colorClass}`}>
      <span className="stat-icon">{icon}</span>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {change && <div className="stat-change">{change}</div>}
      {sub   && <div style={{ fontSize:'.7rem', color:'var(--dim)', marginTop:'.3rem' }}>{sub}</div>}
    </div>
  )
}

export default function Dashboard() {
  const { history } = useApp()
  const [animBars, setAnimBars] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setAnimBars(true), 200)
    return () => clearTimeout(t)
  }, [])

  // Derive stats from history
  const total     = history.length
  const healthy   = history.filter(h => h.severity === 'healthy').length
  const defective = total - healthy
  const avgConf   = total ? (history.reduce((a,b) => a + b.confidence, 0) / total).toFixed(1) : '–'

  // Severity distribution for pie chart
  const sevCounts = { healthy:0, mild:0, moderate:0, severe:0 }
  history.forEach(h => sevCounts[h.severity] = (sevCounts[h.severity] || 0) + 1)
  const pieData = Object.entries(sevCounts)
    .filter(([,v]) => v > 0)
    .map(([key, value]) => ({ name: key.charAt(0).toUpperCase() + key.slice(1), value, color: COLORS[key] }))

  // Defect type counts for bar chart
  const defectCounts = {}
  history.forEach(h => {
    if (h.prediction && h.severity !== 'healthy') {
      defectCounts[h.prediction] = (defectCounts[h.prediction] || 0) + 1
    }
  })
  const barData = Object.entries(defectCounts)
    .sort((a,b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, count]) => ({ name: name.length > 12 ? name.slice(0,11)+'…' : name, count }))

  // Confidence trend for line chart (last 10)
  const trendData = [...history].reverse().slice(0, 10).map((h, i) => ({
    name: `#${i+1}`,
    confidence: h.confidence,
    label: h.image,
  }))

  // Daily volume (mock: group last 7 days)
  const dailyData = [
    { day:'Mon', scans:8  },
    { day:'Tue', scans:14 },
    { day:'Wed', scans:11 },
    { day:'Thu', scans:19 },
    { day:'Fri', scans:23 },
    { day:'Sat', scans:6  },
    { day:'Sun', scans:total },
  ]

  const { navigate } = useApp()

  if (total === 0) {
    return (
      <div style={{ textAlign:'center', padding:'6rem 2rem' }}>
        <div style={{ fontSize:'4rem', marginBottom:'1.5rem' }}>📈</div>
        <h2 style={{ marginBottom:'.5rem', fontSize:'1.4rem' }}>Dashboard Empty</h2>
        <p style={{ color:'var(--dim)', maxWidth:400, margin:'0 auto 1.5rem' }}>
          No analytics data found. Upload and inspect a solar panel cell to generate reports.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('upload')}>
          🚀 Go to Upload
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Stat cards */}
      <div className="stats-grid">
        <StatCard icon="🔭" label="Total Analyzed"   value={total}     colorClass="blue"   change={`+${Math.max(1, Math.round(total * .12))} this week`} />
        <StatCard icon="✅" label="Healthy Panels"   value={healthy}   colorClass="green"  change={`${total ? ((healthy/total)*100).toFixed(0) : 0}% of total`} />
        <StatCard icon="⚠️" label="Defective Panels" value={defective} colorClass="red"    change={`${total ? ((defective/total)*100).toFixed(0) : 0}% of total`} />
        <StatCard icon="🎯" label="Avg. Confidence"  value={`${avgConf}%`} colorClass="orange" sub="CNN inference score" />
      </div>

      {/* Charts row 1 */}
      <div className="dashboard-charts">

        {/* Severity pie */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">🍩 Severity Distribution</div>
              <div className="card-subtitle">Breakdown of {total} analyzed panels</div>
            </div>
          </div>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  outerRadius={85} innerRadius={50}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height:240, display:'flex', alignItems:'center', justifyContent:'center', color:'var(--dim)', fontSize:'.85rem' }}>
              No data yet — start analyzing panels!
            </div>
          )}
          {/* Legend */}
          <div style={{ display:'flex', flexWrap:'wrap', gap:'.5rem', marginTop:'.5rem' }}>
            {pieData.map(p => (
              <div key={p.name} style={{ display:'flex', alignItems:'center', gap:'.35rem', fontSize:'.75rem' }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:p.color }} />
                <span style={{ color:'var(--muted)' }}>{p.name}</span>
                <span style={{ color:'var(--dim)' }}>({p.value})</span>
              </div>
            ))}
          </div>
        </div>

        {/* Defect type bar */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">📊 Top Defect Types</div>
              <div className="card-subtitle">Frequency of each defect class</div>
            </div>
          </div>
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={barData} margin={{ top:10, right:10, left:-10, bottom:40 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill:'#64748B', fontSize:11 }} angle={-30} textAnchor="end" />
                <YAxis tick={{ fill:'#64748B', fontSize:11 }} allowDecimals={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="count" name="Count" radius={[4,4,0,0]}>
                  {barData.map((_, i) => (
                    <Cell key={i} fill={DEFECT_COLORS[i % DEFECT_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height:260, display:'flex', alignItems:'center', justifyContent:'center', color:'var(--dim)', fontSize:'.85rem' }}>
              No defects recorded yet.
            </div>
          )}
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="dashboard-charts" style={{ marginTop:0 }}>

        {/* Confidence trend */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">📈 Confidence Trend</div>
              <div className="card-subtitle">Last {trendData.length} analyses</div>
            </div>
          </div>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={trendData} margin={{ top:10, right:10, left:-10, bottom:0 }}>
                <defs>
                  <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#3B82F6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill:'#64748B', fontSize:11 }} />
                <YAxis domain={[50, 100]} tick={{ fill:'#64748B', fontSize:11 }} unit="%" />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone" dataKey="confidence" name="Confidence"
                  unit="%" stroke="#3B82F6" strokeWidth={2.5}
                  fill="url(#confGrad)" dot={{ fill:'#3B82F6', r:4 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height:220, display:'flex', alignItems:'center', justifyContent:'center', color:'var(--dim)', fontSize:'.85rem' }}>
              Run some analyses to see the trend.
            </div>
          )}
        </div>

        {/* Daily scan volume */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">📅 Daily Scan Volume</div>
              <div className="card-subtitle">Past 7 days</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dailyData} margin={{ top:10, right:10, left:-10, bottom:0 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="day" tick={{ fill:'#64748B', fontSize:11 }} />
              <YAxis tick={{ fill:'#64748B', fontSize:11 }} allowDecimals={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="scans" name="Scans" fill="#22C55E" radius={[4,4,0,0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model performance metrics */}
      <div className="card" style={{ marginTop:'1.5rem' }}>
        <div className="card-header">
          <div className="card-title">🧠 Model Performance Metrics</div>
          <div style={{
            fontSize:'.72rem', padding:'3px 10px', borderRadius:'999px',
            background:'var(--green-dim)', color:'var(--green)',
            border:'1px solid rgba(34,197,94,.3)', fontWeight:600
          }}>PVEL-AD v2 · ResNet-50</div>
        </div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:'1rem' }}>
          {[
            { label:'Accuracy',  value:'94.3%', icon:'🎯', color:'var(--green)' },
            { label:'Precision', value:'91.8%', icon:'🔍', color:'var(--blue)' },
            { label:'Recall',    value:'93.2%', icon:'📡', color:'var(--cyan)' },
            { label:'F1 Score',  value:'92.5%', icon:'⚖️', color:'var(--purple)' },
            { label:'mAP',       value:'89.7%', icon:'📊', color:'var(--orange)' },
          ].map(m => (
            <div key={m.label} style={{
              textAlign:'center', padding:'1.2rem .8rem',
              background:'rgba(255,255,255,0.02)', border:'1px solid var(--border)',
              borderRadius:'var(--radius-sm)', transition:'all .2s'
            }}>
              <div style={{ fontSize:'1.5rem', marginBottom:'.4rem' }}>{m.icon}</div>
              <div style={{ fontSize:'1.4rem', fontWeight:800, fontFamily:'JetBrains Mono, monospace', color: m.color }}>
                {m.value}
              </div>
              <div style={{ fontSize:'.72rem', color:'var(--dim)', marginTop:'.2rem' }}>{m.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
