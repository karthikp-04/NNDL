export default function Topbar({ page }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <h2>{page.title}</h2>
        <p>{page.sub}</p>
      </div>
      <div className="topbar-right">
        <div className="topbar-badge">
          <span className="status-dot" style={{ width: 6, height: 6 }}></span>
          Model Online
        </div>
        <div className="topbar-avatar">SG</div>
      </div>
    </header>
  )
}
