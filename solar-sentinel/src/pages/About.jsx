import { useApp } from '../context/AppContext'

const DATASET_FACTS = [
  { label: 'Dataset', value: 'PVEL-AD (PV EL Anomaly Detection)' },
  { label: 'Total Images', value: '26,300+ EL images' },
  { label: 'Defect Classes', value: '10 categories' },
  { label: 'Image Type', value: 'Electroluminescence (EL)' },
  { label: 'Resolution', value: '300 × 300 px (standardized)' },
  { label: 'Split', value: '70% Train · 15% Val · 15% Test' },
]

const MODEL_FACTS = [
  { label: 'DenseNet-121 Latency', value: '18.27 ms (54.7 FPS on CUDA)' },
  { label: 'DenseNet-121 Parameters', value: '6.97 Million' },
  { label: 'ResNet-50 Parameters', value: '25.56 Million' },
  { label: 'Input Resolution', value: '224 × 224 px Standardized' },
  { label: 'Test Set Accuracy', value: '91.71% (ResNet) / 87.00% (DenseNet)' },
  { label: 'PVEL-AD Test Split', value: '2,881 Stratified Images' },
]

const DEFECT_CLASSES = [
  { name:'Cell Micro Crack',    icon:'🔴', desc:'Hairline fractures within silicon cells' },
  { name:'Finger Interruption', icon:'🟠', desc:'Broken metallic contact lines on cell surface' },
  { name:'Black Core',          icon:'⚫', desc:'Inactive non-luminescent core area' },
  { name:'Crystal Dislocation', icon:'🟣', desc:'Structural crystallographic lattice dislocations' },
  { name:'Material Defect',     icon:'🟤', desc:'Silicon material impurity and inhomogeneity' },
  { name:'Star Crack',          icon:'🚨', desc:'Impact fracture star pattern' },
  { name:'Thick Line Defect',   icon:'📏', desc:'Busbar ribbon solder misalignment' },
  { name:'Healthy Panel',       icon:'🟢', desc:'Optimal electroluminescence with no defect' },
]

const TEAM = [
  { initials:'M', name:'Madhav', roll:'CB.EN.U4CSE23661', role:'Deep Learning & UI Lead', color:'#3B82F6' },
]

const REFS = [
  'Deitsch, S. et al. (2019). Automatic classification of defective photovoltaic module cells in electroluminescence images. Solar Energy.',
  'Pratt, L. et al. (2020). PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic Cell Anomaly Detection. IEEE PVSC.',
  'Selvaraju, R.R. et al. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. IJCV.',
  'He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.',
]

export default function About() {
  const { navigate } = useApp()
  return (
    <div>
      {/* Hero banner */}
      <div className="about-hero">
        <div className="about-logo-big">☀️</div>
        <div>
          <h1>SolarGuard <em style={{ fontStyle:'italic', color:'#F59E0B' }}>AI</em></h1>
          <p>
            An AI-powered solar panel defect detection system using Deep Convolutional Neural Networks
            trained on the PVEL-AD dataset. Detects 10 defect types from electroluminescence (EL)
            images and provides Grad-CAM visual explanations for every prediction.
          </p>
          <div style={{ display:'flex', gap:'.6rem', marginTop:'1.2rem' }}>
            <button className="btn btn-primary" onClick={() => navigate('upload')}>
              🔍 Try Analysis
            </button>
            <button className="btn btn-outline" onClick={() => navigate('dashboard')}>
              📈 Dashboard
            </button>
          </div>
        </div>
      </div>

      {/* Problem + Dataset */}
      <div className="about-grid">

        {/* Problem Statement */}
        <div className="card">
          <div className="about-section-title">⚡ Problem Statement</div>
          <p style={{ fontSize:'.88rem', lineHeight:1.7, marginBottom:'1rem' }}>
            Solar panels degrade over time due to manufacturing defects, environmental stress,
            and electrical faults. Manual inspection is <strong style={{ color:'var(--text)' }}>
            expensive, slow, and error-prone</strong> — especially at utility-scale installations.
          </p>
          <p style={{ fontSize:'.88rem', lineHeight:1.7 }}>
            SolarSentinel AI automates this process using electroluminescence imaging combined
            with a DCNN model to detect defects with <strong style={{ color:'var(--green)' }}>
            &gt;94% accuracy</strong> in under 2 seconds.
          </p>
          <div style={{ marginTop:'1.2rem' }}>
            {[
              { icon:'💡', text:'Reduce inspection cost by up to 80%' },
              { icon:'⚡', text:'Detect defects before power loss escalates' },
              { icon:'📊', text:'Quantify severity for prioritized maintenance' },
            ].map(b => (
              <div key={b.text} style={{
                display:'flex', alignItems:'center', gap:'.7rem',
                padding:'.5rem .8rem', borderRadius:'var(--radius-sm)',
                background:'rgba(34,197,94,0.05)', border:'1px solid rgba(34,197,94,0.15)',
                marginBottom:'.5rem', fontSize:'.85rem', color:'var(--muted)'
              }}>
                <span>{b.icon}</span> {b.text}
              </div>
            ))}
          </div>
        </div>

        {/* Dataset */}
        <div className="card">
          <div className="about-section-title">🗂️ Dataset — PVEL-AD</div>
          <div style={{ display:'flex', flexDirection:'column', gap:'.5rem' }}>
            {DATASET_FACTS.map(f => (
              <div key={f.label} style={{
                display:'flex', justifyContent:'space-between', alignItems:'center',
                padding:'.55rem .8rem', borderRadius:'var(--radius-sm)',
                background:'rgba(255,255,255,0.02)', border:'1px solid var(--border)',
                fontSize:'.83rem'
              }}>
                <span style={{ color:'var(--dim)' }}>{f.label}</span>
                <span style={{ color:'var(--text)', fontWeight:600 }}>{f.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model + Defect classes */}
      <div className="about-grid">

        {/* Model architecture */}
        <div className="card">
          <div className="about-section-title">🧠 Model Architecture</div>
          <div style={{ display:'flex', flexDirection:'column', gap:'.5rem', marginBottom:'1rem' }}>
            {MODEL_FACTS.map(f => (
              <div key={f.label} style={{
                display:'flex', justifyContent:'space-between', alignItems:'center',
                padding:'.55rem .8rem', borderRadius:'var(--radius-sm)',
                background:'rgba(255,255,255,0.02)', border:'1px solid var(--border)',
                fontSize:'.83rem'
              }}>
                <span style={{ color:'var(--dim)' }}>{f.label}</span>
                <span style={{ color:'var(--text)', fontWeight:600 }}>{f.value}</span>
              </div>
            ))}
          </div>
          <div className="about-section-title" style={{ marginTop:'1rem' }}>🛠️ Tech Stack</div>
          <div className="tech-tags">
            {['Python 3.10', 'PyTorch 2.x', 'ResNet-50', 'Grad-CAM', 'OpenCV',
              'NumPy', 'Matplotlib', 'React.js', 'Recharts', 'Vite'].map(t => (
              <span key={t} className="tech-tag">{t}</span>
            ))}
          </div>
        </div>

        {/* Defect classes */}
        <div className="card">
          <div className="about-section-title">🔍 Defect Classification</div>
          <div style={{ display:'flex', flexDirection:'column', gap:'.4rem' }}>
            {DEFECT_CLASSES.map(d => (
              <div key={d.name} style={{
                display:'flex', alignItems:'center', gap:'.7rem',
                padding:'.45rem .7rem', borderRadius:'var(--radius-sm)',
                background:'rgba(255,255,255,0.02)', border:'1px solid var(--border)',
                fontSize:'.8rem'
              }}>
                <span style={{ fontSize:'1rem', flexShrink:0 }}>{d.icon}</span>
                <div>
                  <div style={{ fontWeight:600, color:'var(--text)', fontSize:'.82rem' }}>{d.name}</div>
                  <div style={{ color:'var(--dim)', fontSize:'.72rem' }}>{d.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* How it works */}
      <div className="card" style={{ marginBottom:'1.5rem' }}>
        <div className="about-section-title">⚙️ How It Works</div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:'1rem' }}>
          {[
            { step:'01', icon:'📷', title:'Image Input', desc:'Upload EL, RGB or Infrared image of a solar panel' },
            { step:'02', icon:'🔧', title:'Preprocessing', desc:'Resize to 224×224, normalize & convert for CNN input' },
            { step:'03', icon:'🧠', title:'CNN Inference', desc:'ResNet-50 extracts features and classifies defects' },
            { step:'04', icon:'🔥', title:'Grad-CAM', desc:'Gradients highlight which regions the model focused on' },
            { step:'05', icon:'📊', title:'Report', desc:'Severity, confidence, and maintenance recommendations' },
          ].map(s => (
            <div key={s.step} style={{ textAlign:'center', padding:'1rem .5rem' }}>
              <div style={{
                width:36, height:36, borderRadius:'50%', margin:'0 auto .7rem',
                background:'var(--blue-dim)', border:'1px solid rgba(59,130,246,.3)',
                display:'flex', alignItems:'center', justifyContent:'center',
                fontSize:'.7rem', fontWeight:800, color:'var(--blue)',
                fontFamily:'JetBrains Mono, monospace'
              }}>{s.step}</div>
              <div style={{ fontSize:'1.4rem', marginBottom:'.4rem' }}>{s.icon}</div>
              <div style={{ fontSize:'.82rem', fontWeight:700, color:'var(--text)', marginBottom:'.3rem' }}>{s.title}</div>
              <div style={{ fontSize:'.72rem', color:'var(--dim)', lineHeight:1.4 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Team + References */}
      <div className="about-grid">

        {/* Team */}
        <div className="card">
          <div className="about-section-title">👥 Team</div>
          <div className="team-grid">
            {TEAM.map(m => (
              <div key={m.initials} className="team-member">
                <div className="member-avatar" style={{ background: m.color }}>
                  {m.initials}
                </div>
                <div>
                  <div className="member-name" style={{ display:'flex', alignItems:'center', gap:'.5rem' }}>
                    <span>{m.name}</span>
                    <span style={{ fontSize:'.68rem', color:'var(--dim)', background:'rgba(255,255,255,0.06)', padding:'1px 6px', borderRadius:4 }}>
                      {m.roll}
                    </span>
                  </div>
                  <div className="member-role" style={{ color: m.color }}>{m.role}</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{
            marginTop:'1.2rem', padding:'1rem',
            background:'rgba(245,158,11,0.05)', border:'1px solid rgba(245,158,11,0.2)',
            borderRadius:'var(--radius-sm)'
          }}>
            <div style={{ fontSize:'.8rem', fontWeight:700, color:'var(--orange)', marginBottom:'.3rem' }}>
              🏫 Project Info
            </div>
            <p style={{ fontSize:'.78rem', margin:0, lineHeight:1.6 }}>
              Final year B.E. project — Department of Computer Science & Engineering.
              Submitted in partial fulfillment for the degree of Bachelor of Engineering.
            </p>
          </div>
        </div>

        {/* References */}
        <div className="card">
          <div className="about-section-title">📚 References</div>
          <ul className="ref-list">
            {REFS.map((r, i) => (
              <li key={i}>
                <span className="ref-num">[{i+1}]</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>

          <div style={{ marginTop:'1.5rem' }}>
            <div className="about-section-title" style={{ marginBottom:'.8rem' }}>🔗 Quick Links</div>
            <div style={{ display:'flex', flexWrap:'wrap', gap:'.5rem' }}>
              {[
                { label:'PVEL-AD Dataset', href:'#' },
                { label:'ResNet Paper', href:'#' },
                { label:'Grad-CAM Paper', href:'#' },
                { label:'GitHub Repo', href:'#' },
              ].map(l => (
                <a key={l.label} href={l.href}
                  style={{
                    fontSize:'.78rem', fontWeight:600, padding:'.35rem .8rem',
                    borderRadius:'var(--radius-sm)', border:'1px solid rgba(59,130,246,.3)',
                    color:'var(--blue)', background:'var(--blue-dim)',
                    textDecoration:'none', transition:'all .2s'
                  }}
                  onMouseOver={e => e.currentTarget.style.background = 'rgba(59,130,246,0.2)'}
                  onMouseOut={e => e.currentTarget.style.background = 'var(--blue-dim)'}
                >
                  {l.label} ↗
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
