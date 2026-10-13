import { useState, useMemo } from 'react'
import { useApp } from '../context/AppContext'
import Badge from '../components/ui/Badge'
import { 
  History as HistoryIcon, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  Search, 
  SlidersHorizontal, 
  ArrowUpRight, 
  Download, 
  FileText, 
  Inbox, 
  Eye,
  X
} from 'lucide-react'

const SEV_LABELS = { 
  healthy: 'Healthy', 
  mild: 'Mild', 
  moderate: 'Moderate', 
  severe: 'Severe' 
}

export default function History() {
  const { history, navigate, addToast } = useApp()
  const [search, setSearch]   = useState('')
  const [sevFilter, setSev]   = useState('all')
  const [sortBy, setSort]     = useState('newest')

  const filtered = useMemo(() => {
    let arr = [...history]
    if (search) {
      arr = arr.filter(h =>
        h.image.toLowerCase().includes(search.toLowerCase()) ||
        h.prediction.toLowerCase().includes(search.toLowerCase())
      )
    }
    if (sevFilter !== 'all') arr = arr.filter(h => h.severity === sevFilter)
    if (sortBy === 'newest') arr.sort((a,b) => b.id - a.id)
    if (sortBy === 'oldest') arr.sort((a,b) => a.id - b.id)
    if (sortBy === 'confidence') arr.sort((a,b) => b.confidence - a.confidence)
    return arr
  }, [history, search, sevFilter, sortBy])

  const totalSevere   = history.filter(h => h.severity === 'severe').length
  const totalDefective = history.filter(h => h.severity !== 'healthy').length
  const totalHealthy   = history.length - totalDefective

  if (history.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '5rem 2rem', margin: '2rem auto', maxWidth: 540 }}>
        <div style={{
          width: 56, height: 56, borderRadius: '50%',
          background: 'var(--surface-raised)', border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 1.5rem', color: 'var(--accent)'
        }}>
          <Inbox size={28} />
        </div>
        <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', fontWeight: 700, marginBottom: '.5rem' }}>
          No Scan History Recorded
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '.9rem', margin: '0 auto 1.75rem', lineHeight: 1.6 }}>
          Your solar panel inspection logs will appear here once you process electroluminescence images.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('upload')}>
          <ArrowUpRight size={16} /> Go to Analysis Studio
        </button>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Summary Metrics Grid */}
      <div style={{
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', 
        gap: '1rem'
      }}>
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{
            width: 44, height: 44, borderRadius: 'var(--radius-sm)',
            background: 'var(--surface-raised)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--accent)'
          }}>
            <HistoryIcon size={20} />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>
              {history.length}
            </div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
              Total Logs
            </div>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{
            width: 44, height: 44, borderRadius: 'var(--radius-sm)',
            background: 'var(--surface-raised)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--status-warn)'
          }}>
            <AlertTriangle size={20} />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-warn)' }}>
              {totalDefective}
            </div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
              Defective Panels
            </div>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{
            width: 44, height: 44, borderRadius: 'var(--radius-sm)',
            background: 'var(--surface-raised)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--status-fail)'
          }}>
            <XCircle size={20} />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-fail)' }}>
              {totalSevere}
            </div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
              Severe Defects
            </div>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1.25rem' }}>
          <div style={{
            width: 44, height: 44, borderRadius: 'var(--radius-sm)',
            background: 'var(--surface-raised)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--status-pass)'
          }}>
            <CheckCircle2 size={20} />
          </div>
          <div>
            <div style={{ fontSize: '1.6rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-pass)' }}>
              {totalHealthy}
            </div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>
              Healthy Modules
            </div>
          </div>
        </div>
      </div>

      {/* Filter and Control Bar */}
      <div className="card" style={{ padding: '1rem 1.25rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.85rem', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: '1 1 240px' }}>
            <Search size={15} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              className="search-input"
              style={{ width: '100%', paddingLeft: '2.2rem' }}
              placeholder="Search by panel ID or classification..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              id="history-search"
            />
          </div>

          <div style={{ display: 'flex', gap: '.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', color: 'var(--text-muted)', fontSize: '.8rem' }}>
              <SlidersHorizontal size={14} />
              <span>Filter:</span>
            </div>
            <select 
              className="filter-select" 
              value={sevFilter} 
              onChange={e => setSev(e.target.value)} 
              id="sev-filter"
            >
              <option value="all">All Severities</option>
              <option value="healthy">Healthy Only</option>
              <option value="mild">Mild</option>
              <option value="moderate">Moderate</option>
              <option value="severe">Severe</option>
            </select>

            <select 
              className="filter-select" 
              value={sortBy} 
              onChange={e => setSort(e.target.value)} 
              id="sort-select"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="confidence">Confidence (High to Low)</option>
            </select>

            {(search || sevFilter !== 'all') && (
              <button 
                className="btn btn-outline" 
                style={{ padding: '.45rem .8rem', fontSize: '.78rem', display: 'inline-flex', alignItems: 'center', gap: '.35rem' }}
                onClick={() => { setSearch(''); setSev('all') }}
              >
                <X size={13} /> Reset
              </button>
            )}
          </div>

          <span style={{ marginLeft: 'auto', fontSize: '.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Showing {filtered.length} of {history.length}
          </span>
        </div>
      </div>

      {/* Records Table */}
      <div className="history-table-wrap card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th style={{ width: 50 }}>#</th>
              <th>Timestamp</th>
              <th>Panel Image</th>
              <th>Diagnostic Class</th>
              <th>Severity Grade</th>
              <th>Confidence</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-muted)' }}>
                  <Inbox size={32} style={{ margin: '0 auto .75rem', opacity: 0.5, display: 'block' }} />
                  <div>No audit logs match current filters.</div>
                </td>
              </tr>
            ) : (
              filtered.map((item, idx) => {
                const badgeVariant = 
                  item.severity === 'healthy' ? 'pass' :
                  item.severity === 'severe' ? 'fail' : 'warn'

                return (
                  <tr key={item.id}>
                    <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '.78rem' }}>
                      {String(idx + 1).padStart(2, '0')}
                    </td>
                    <td>
                      <div style={{ fontSize: '.82rem', color: 'var(--text)', fontFamily: 'var(--font-mono)' }}>
                        {item.time}
                      </div>
                      <div style={{ fontSize: '.72rem', color: 'var(--text-muted)' }}>{item.date}</div>
                    </td>
                    <td>
                      <span style={{ 
                        fontSize: '.82rem', 
                        color: 'var(--text)', 
                        fontFamily: 'var(--font-mono)', 
                        fontWeight: 500,
                        maxWidth: 180, 
                        display: 'inline-block',
                        overflow: 'hidden', 
                        textOverflow: 'ellipsis', 
                        whiteSpace: 'nowrap' 
                      }}>
                        {item.image}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        fontSize: '.82rem', 
                        fontWeight: 600,
                        color: item.severity === 'healthy' ? 'var(--status-pass)' :
                               item.severity === 'severe' ? 'var(--status-fail)' : 'var(--text)'
                      }}>
                        {item.prediction}
                      </span>
                    </td>
                    <td>
                      <Badge variant={badgeVariant}>
                        {SEV_LABELS[item.severity] || item.severity}
                      </Badge>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem' }}>
                        <div style={{
                          width: 56, height: 5, borderRadius: 999,
                          background: 'var(--border)', overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${item.confidence}%`, height: '100%',
                            background: item.confidence >= 80 ? 'var(--status-pass)' :
                                        item.confidence >= 60 ? 'var(--status-warn)' : 'var(--status-fail)',
                            borderRadius: 999
                          }} />
                        </div>
                        <span style={{
                          fontSize: '.8rem', fontFamily: 'var(--font-mono)',
                          color: 'var(--text)', fontWeight: 600
                        }}>
                          {item.confidence.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '.4rem', justifyContent: 'flex-end' }}>
                        <button 
                          className="btn btn-outline" 
                          style={{ padding: '.35rem .65rem', fontSize: '.75rem', display: 'inline-flex', alignItems: 'center', gap: '.3rem' }}
                          onClick={() => addToast(`Viewing audit report for ${item.image}`, 'info')}
                          title="View diagnostic inspection"
                        >
                          <Eye size={13} /> View
                        </button>
                        <button 
                          className="btn btn-outline" 
                          style={{ padding: '.35rem .6rem', fontSize: '.75rem', display: 'inline-flex', alignItems: 'center' }}
                          onClick={() => addToast(`Exported panel ${item.image} telemetry`, 'success')}
                          title="Export certificate"
                        >
                          <FileText size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Export Footer */}
      {history.length > 0 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '.85rem' }}>
          <button 
            className="btn btn-outline" 
            style={{ fontSize: '.82rem', display: 'inline-flex', alignItems: 'center', gap: '.45rem' }}
            onClick={() => addToast('Exporting audit log as CSV...', 'info')}
          >
            <Download size={14} /> Export CSV
          </button>
          <button 
            className="btn btn-outline" 
            style={{ fontSize: '.82rem', display: 'inline-flex', alignItems: 'center', gap: '.45rem' }}
            onClick={() => addToast('Compiling comprehensive audit PDF...', 'info')}
          >
            <FileText size={14} /> Export Report PDF
          </button>
        </div>
      )}
    </div>
  )
}
