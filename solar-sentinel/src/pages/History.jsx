import { useState, useMemo } from 'react'
import { useApp } from '../context/AppContext'

const SEV_LABELS = { healthy:'Healthy', mild:'Mild', moderate:'Moderate', severe:'Severe' }

export default function History() {
  const { history, navigate, addToast } = useApp()
  const [search, setSearch]   = useState('')
  const [sevFilter, setSev]   = useState('all')
  const [sortBy, setSort]     = useState('newest')

  const filtered = useMemo(() => {
    let arr = [...history]
    if (search)        arr = arr.filter(h =>
      h.image.toLowerCase().includes(search.toLowerCase()) ||
      h.prediction.toLowerCase().includes(search.toLowerCase())
    )
    if (sevFilter !== 'all') arr = arr.filter(h => h.severity === sevFilter)
    if (sortBy === 'newest')    arr.sort((a,b) => b.id - a.id)
    if (sortBy === 'oldest')    arr.sort((a,b) => a.id - b.id)
    if (sortBy === 'confidence') arr.sort((a,b) => b.confidence - a.confidence)
    return arr
  }, [history, search, sevFilter, sortBy])

  const totalSevere   = history.filter(h => h.severity === 'severe').length
  const totalDefective = history.filter(h => h.severity !== 'healthy').length

  if (history.length === 0) {
    return (
      <div style={{ textAlign:'center', padding:'6rem 2rem' }}>
        <div style={{ fontSize:'4rem', marginBottom:'1.5rem' }}>📜</div>
        <h2 style={{ marginBottom:'.5rem', fontSize:'1.4rem' }}>No History Recorded</h2>
        <p style={{ color:'var(--dim)', maxWidth:400, margin:'0 auto 1.5rem' }}>
          Your scan history list is currently empty. Run a solar panel inspection to save results.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('upload')}>
          🚀 Go to Upload
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Summary strip */}
      <div style={{
        display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:'1rem', marginBottom:'1.5rem'
      }}>
        {[
          { label:'Total Records',  value: history.length, color:'var(--blue)',   icon:'📜' },
          { label:'Defective',      value: totalDefective,  color:'var(--orange)', icon:'⚠️' },
          { label:'Critical',       value: totalSevere,     color:'var(--red)',    icon:'🔴' },
          { label:'Healthy',        value: history.length - totalDefective, color:'var(--green)', icon:'✅' },
        ].map(s => (
          <div key={s.label} style={{
            background:'var(--bg2)', border:'1px solid var(--border)',
            borderRadius:'var(--radius)', padding:'1rem 1.2rem',
            display:'flex', alignItems:'center', gap:'1rem'
          }}>
            <span style={{ fontSize:'1.5rem' }}>{s.icon}</span>
            <div>
              <div style={{ fontSize:'1.5rem', fontWeight:800, fontFamily:'JetBrains Mono, monospace', color: s.color }}>
                {s.value}
              </div>
              <div style={{ fontSize:'.72rem', color:'var(--dim)', fontWeight:600 }}>{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="history-filters">
        <input
          className="search-input"
          placeholder="🔍  Search by image name or defect..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          id="history-search"
        />
        <select className="filter-select" value={sevFilter} onChange={e => setSev(e.target.value)} id="sev-filter">
          <option value="all">All Severities</option>
          <option value="healthy">Healthy</option>
          <option value="mild">Mild</option>
          <option value="moderate">Moderate</option>
          <option value="severe">Severe</option>
        </select>
        <select className="filter-select" value={sortBy} onChange={e => setSort(e.target.value)} id="sort-select">
          <option value="newest">Newest First</option>
          <option value="oldest">Oldest First</option>
          <option value="confidence">Highest Confidence</option>
        </select>
        <span style={{ marginLeft:'auto', fontSize:'.8rem', color:'var(--dim)' }}>
          {filtered.length} of {history.length} records
        </span>
        {(search || sevFilter !== 'all') && (
          <button className="btn btn-outline" style={{ padding:'.4rem .9rem', fontSize:'.78rem' }}
            onClick={() => { setSearch(''); setSev('all') }}>
            ✕ Clear
          </button>
        )}
      </div>

      {/* Table */}
      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Image</th>
              <th>Prediction</th>
              <th>Severity</th>
              <th>Confidence</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ textAlign:'center', padding:'3rem', color:'var(--dim)' }}>
                  <div style={{ fontSize:'2rem', marginBottom:'.5rem' }}>📭</div>
                  No records match your filter.
                </td>
              </tr>
            ) : (
              filtered.map((item, idx) => (
                <tr key={item.id}>
                  <td style={{ color:'var(--dim)', fontFamily:'JetBrains Mono, monospace', fontSize:'.75rem' }}>
                    {String(idx + 1).padStart(2,'0')}
                  </td>
                  <td>
                    <div style={{ fontSize:'.82rem', color:'var(--muted)', fontFamily:'JetBrains Mono, monospace' }}>
                      {item.time}
                    </div>
                    <div style={{ fontSize:'.7rem', color:'var(--dim)' }}>{item.date}</div>
                  </td>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:'.7rem' }}>
                      <div className="history-thumb">🌞</div>
                      <span style={{ fontSize:'.82rem', color:'var(--text)', maxWidth:140, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                        {item.image}
                      </span>
                    </div>
                  </td>
                  <td>
                    <span style={{
                      fontSize:'.8rem', fontWeight:600,
                      color: item.severity === 'healthy' ? 'var(--green)' :
                             item.severity === 'mild' ? 'var(--blue)' :
                             item.severity === 'moderate' ? 'var(--orange)' : 'var(--red)'
                    }}>
                      {item.prediction}
                    </span>
                  </td>
                  <td>
                    <span className={`sev-pill ${item.severity}`}>
                      {item.severity === 'healthy' ? '🟢' :
                       item.severity === 'mild'    ? '🟡' :
                       item.severity === 'moderate'? '🟠' : '🔴'}
                      {' '}{SEV_LABELS[item.severity]}
                    </span>
                  </td>
                  <td>
                    <div style={{ display:'flex', alignItems:'center', gap:'.4rem' }}>
                      <div style={{
                        width: 48, height:6, borderRadius:999,
                        background:'var(--bg3)', overflow:'hidden'
                      }}>
                        <div style={{
                          width:`${item.confidence}%`, height:'100%',
                          background: item.confidence > 85 ? 'var(--green)' :
                                      item.confidence > 65 ? 'var(--orange)' : 'var(--red)',
                          borderRadius:999
                        }} />
                      </div>
                      <span style={{
                        fontSize:'.78rem', fontFamily:'JetBrains Mono, monospace',
                        color:'var(--text)', fontWeight:600
                      }}>
                        {item.confidence.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td>
                    <div style={{ display:'flex', gap:'.4rem' }}>
                      <button className="action-btn" onClick={() => addToast(`Viewing ${item.image}`, 'info')}>
                        👁 View
                      </button>
                      <button className="action-btn" onClick={() => addToast('Report exported', 'success')}>
                        📄
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Export footer */}
      {history.length > 0 && (
        <div style={{
          display:'flex', justifyContent:'flex-end', gap:'.8rem',
          marginTop:'1.2rem'
        }}>
          <button className="btn btn-outline" style={{ fontSize:'.82rem' }}
            onClick={() => addToast('Exporting history as CSV...', 'info')}>
            📥 Export CSV
          </button>
          <button className="btn btn-outline" style={{ fontSize:'.82rem' }}
            onClick={() => addToast('Generating PDF report...', 'info')}>
            📄 Export PDF
          </button>
        </div>
      )}
    </div>
  )
}
