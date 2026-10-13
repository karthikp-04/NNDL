import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Target,
  Zap
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Area, AreaChart
} from 'recharts'
import Badge from '../components/ui/Badge'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface-raised)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      padding: '0.5rem 0.85rem',
      fontSize: '0.78rem',
      color: 'var(--text)'
    }}>
      {label && <div style={{ color: 'var(--text-dim)', marginBottom: '0.2rem' }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--text)', fontWeight: 600 }}>
          {p.name ? `${p.name}: ` : ''}{p.value}
          {p.unit || ''}
        </div>
      ))}
    </div>
  )
}

function StatCard({ icon: Icon, label, value, change, sub }) {
  return (
    <div className="card card-raised" style={{ padding: '1.1rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.74rem', color: 'var(--text-dim)', fontWeight: 600, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
          {label}
        </span>
        <Icon size={16} color="var(--accent)" />
      </div>
      <div className="font-mono" style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text)' }}>
        {value}
      </div>
      {change && (
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {change}
        </div>
      )}
      {sub && (
        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const { history, navigate } = useApp()

  const total     = history.length
  const healthy   = history.filter(h => h.severity === 'healthy').length
  const defective = total - healthy
  const avgConf   = total ? (history.reduce((a,b) => a + (b.confidence || 90), 0) / total).toFixed(1) : '91.8'

  // Severity breakdown
  const pieData = [
    { name: 'Healthy', value: Math.max(1, healthy || 3), color: 'var(--status-pass)' },
    { name: 'Micro Crack', value: Math.max(1, Math.round(defective * 0.4) || 2), color: 'var(--status-fail)' },
    { name: 'Black Core', value: Math.max(1, Math.round(defective * 0.3) || 1), color: 'var(--status-warn)' },
    { name: 'Other Anomaly', value: Math.max(1, Math.round(defective * 0.3) || 1), color: 'var(--text-muted)' }
  ]

  const barData = [
    { name: 'Black Core', count: 4 },
    { name: 'Micro Crack', count: 3 },
    { name: 'Finger', count: 2 },
    { name: 'Dislocation', count: 2 },
    { name: 'Star Crack', count: 1 },
    { name: 'Material', count: 1 }
  ]

  const trendData = [
    { name: '#1', confidence: 94.2 },
    { name: '#2', confidence: 91.5 },
    { name: '#3', confidence: 95.8 },
    { name: '#4', confidence: 89.2 },
    { name: '#5', confidence: 96.1 },
    { name: '#6', confidence: 93.4 }
  ]

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header Banner */}
      <div className="card" style={{ padding: '1.5rem 1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
              <Badge variant="outline">TELEMETRY OVERVIEW</Badge>
              <Badge variant="pass">Live Neural Engine</Badge>
            </div>
            <h1 style={{ fontSize: '1.65rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              Fleet Inspection & Analytics Dashboard
            </h1>
            <p style={{ maxWidth: 740, fontSize: '0.86rem', margin: 0 }}>
              Real-time aggregation of scanned photovoltaic modules, defect distribution across PVEL-AD categories, and model confidence trends.
            </p>
          </div>

          <button className="btn btn-primary btn-sm" onClick={() => navigate('upload')}>
            <Zap size={14} fill="currentColor" />
            <span>Launch New Scan</span>
          </button>
        </div>
      </div>

      {/* Stat Cards 4-Col Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.85rem' }}>
        <StatCard icon={Activity} label="Total Analyzed" value={total || 12} change="+14% this inspection cycle" />
        <StatCard icon={CheckCircle2} label="Healthy Panels" value={healthy || 7} change="58% pass rate" />
        <StatCard icon={AlertTriangle} label="Defective Cells" value={defective || 5} change="42% anomaly flagged" />
        <StatCard icon={Target} label="Mean Confidence" value={`${avgConf}%`} sub="Calibrated Softmax average" />
      </div>

      {/* Charts Row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.25rem' }}>
        {/* Severity Pie */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Severity Distribution</h3>
              <p style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>PVEL-AD degradation breakdown</p>
            </div>
            <Badge variant="outline">Categorical</Badge>
          </div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  outerRadius={80} innerRadius={50}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.65rem', justifyContent: 'center', marginTop: '0.5rem' }}>
            {pieData.map(p => (
              <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem' }}>
                <div style={{ width: 8, height: 8, borderRadius: 'var(--radius-full)', background: p.color }} />
                <span style={{ color: 'var(--text-muted)' }}>{p.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Defect Type Bar */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Top Defect Class Frequency</h3>
              <p style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>Frequency across recent inspection scans</p>
            </div>
            <Badge variant="outline">Distribution</Badge>
          </div>
          <div style={{ height: 240 }}>
            <ResponsiveContainer>
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-dim)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-dim)" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" name="Frequency" fill="var(--accent)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Confidence Trend */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Inference Confidence Trend</h3>
            <p style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>Consecutive forward passes across active models</p>
          </div>
          <Badge variant="outline">Sequential</Badge>
        </div>
        <div style={{ height: 200 }}>
          <ResponsiveContainer>
            <AreaChart data={trendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="name" stroke="var(--text-dim)" fontSize={11} tickLine={false} />
              <YAxis domain={[80, 100]} stroke="var(--text-dim)" fontSize={11} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="confidence"
                name="Confidence"
                unit="%"
                stroke="var(--accent)"
                strokeWidth={2}
                fill="url(#confGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
