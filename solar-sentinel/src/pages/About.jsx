import { useApp } from '../context/AppContext'
import Badge from '../components/ui/Badge'
import { 
  Sun, 
  Zap, 
  Database, 
  Cpu, 
  Layers, 
  Activity, 
  ShieldCheck, 
  Users, 
  BookOpen, 
  ExternalLink, 
  Sparkles,
  Camera,
  Sliders,
  CheckCircle2,
  FileText
} from 'lucide-react'

const DATASET_FACTS = [
  { label: 'Dataset Name', value: 'PVEL-AD (PV EL Anomaly Detection)' },
  { label: 'Total Images', value: '26,300+ Electroluminescence frames' },
  { label: 'Defect Classes', value: '10 distinct categories' },
  { label: 'Modality', value: 'Electroluminescence (EL) Near-IR' },
  { label: 'Native Resolution', value: '300 × 300 px (standardized)' },
  { label: 'Dataset Partitioning', value: '70% Train · 15% Val · 15% Test' },
]

const MODEL_FACTS = [
  { label: 'DenseNet-121 (cGAN)', value: '18.35 ms · 48.47% Balanced Acc (+15.71% gain)' },
  { label: 'DenseNet-121 (Baseline)', value: '18.27 ms (54.7 FPS on CUDA) · 6.97M params' },
  { label: 'ResNet-50 (Diagnostic)', value: '42.10 ms · 91.71% Overall Acc · 25.56M params' },
  { label: 'EfficientNet-B0 (Edge Drone)', value: '14.80 ms (67.5 FPS) · 4.02M params · 0.39 GFLOPs' },
  { label: 'Standard Input Shape', value: '224 × 224 × 3 normalized tensor' },
  { label: 'PVEL-AD Test Split Size', value: '2,881 Stratified Evaluation Images' },
]

const DEFECT_CLASSES = [
  { name: 'Cell Micro Crack', desc: 'Hairline stress fractures within silicon crystalline wafers' },
  { name: 'Finger Interruption', desc: 'Broken silver metallization grid lines impeding current flow' },
  { name: 'Black Core', desc: 'Inactive recombination regions emitting zero luminescence' },
  { name: 'Crystal Dislocation', desc: 'Crystallographic lattice dislocations and slip lines' },
  { name: 'Material Defect', desc: 'Silicon ingot contamination and manufacturing impurities' },
  { name: 'Star Crack', desc: 'Radial point impact fracture propagated across cell center' },
  { name: 'Thick Line Defect', desc: 'Busbar ribbon solder joint misalignment and thermal hot-spots' },
  { name: 'Healthy Panel', desc: 'Optimal homogeneous electroluminescence response without defect' },
]

const TEAM = [
  { initials: 'AV', name: 'Ashin Varghese', roll: 'CB.EN.U4CSE23661', role: 'DenseNet-121 Lead (In-Line AOI & cGAN Aug)' },
  { initials: 'MS', name: 'Madhav Sreejith', roll: 'CB.EN.U4CSE23735', role: 'EfficientNet-B0 Lead (Edge Drone AI & TensorRT)' },
  { initials: 'KP', name: 'Karthik P', roll: 'CB.EN.U4CSE23522', role: 'ResNet-50 Lead (Diagnostic Web & Focal Loss Engine)' },
]

const REFS = [
  'Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.',
  'Deitsch, S. et al. (2019). Automatic classification of defective photovoltaic module cells in electroluminescence images. Solar Energy.',
  'Pratt, L. et al. (2020). PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic Cell Anomaly Detection. IEEE PVSC.',
  'Selvaraju, R.R. et al. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. IJCV.',
  'He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.',
]

export default function About() {
  const { navigate } = useApp()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Hero Banner */}
      <div className="card-raised" style={{
        padding: '2.5rem 2rem',
        borderRadius: 'var(--radius)',
        display: 'flex',
        alignItems: 'center',
        gap: '2rem',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          width: 72, height: 72, borderRadius: 'var(--radius)',
          background: 'var(--surface)', border: '1px solid var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--accent)', flexShrink: 0,
          boxShadow: '0 0 24px rgba(245,165,36,0.18)'
        }}>
          <Sun size={36} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', marginBottom: '.4rem' }}>
            <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.8rem', fontWeight: 800, margin: 0 }}>
              SolarGuard <span style={{ color: 'var(--accent)' }}>AI</span>
            </h1>
            <Badge variant="accent">v2.4 Production</Badge>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '.92rem', lineHeight: 1.6, maxWidth: 840, margin: 0 }}>
            An industrial-grade solar photovoltaic inspection suite powered by Deep Convolutional Neural Networks
            and Conditional GAN data augmentation trained on the PVEL-AD dataset. Delivers millisecond defect classification
            and Grad-CAM visual heatmaps for trustworthy verification.
          </p>
          <div style={{ display: 'flex', gap: '.75rem', marginTop: '1.25rem' }}>
            <button className="btn btn-primary" onClick={() => navigate('upload')}>
              <Sparkles size={15} /> Launch Inspection
            </button>
            <button className="btn btn-outline" onClick={() => navigate('dashboard')}>
              <Activity size={15} /> Performance Analytics
            </button>
          </div>
        </div>
      </div>

      {/* Problem Statement & Dataset */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Problem Statement */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <Zap size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Problem Statement
            </span>
          </div>
          <p style={{ fontSize: '.88rem', lineHeight: 1.65, color: 'var(--text-muted)', margin: 0 }}>
            Solar photovoltaic modules suffer severe efficiency degradation from invisible micro-cracks, finger breaks,
            and lattice dislocations. Manual electroluminescence audits are <strong style={{ color: 'var(--text)' }}>labor-intensive, slow, and subjective</strong>.
          </p>
          <p style={{ fontSize: '.88rem', lineHeight: 1.65, color: 'var(--text-muted)', margin: 0 }}>
            SolarGuard AI automates early-stage anomaly discovery, reducing inspection turnaround from hours to <strong style={{ color: 'var(--status-pass)' }}>sub-20 milliseconds</strong> per module with high confidence localization.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem', marginTop: '.5rem' }}>
            {[
              { title: '80% Lower Inspection Cost', desc: 'Replaces expensive manual laboratory audits with autonomous edge inference.' },
              { title: 'Pre-Failure Intervention', desc: 'Detects micro-fissures before thermal hot-spots cause irreversible cell burn.' },
              { title: 'Quantitative Severity Grading', desc: 'Classifies defects into Mild, Moderate, and Severe triage protocols.' },
            ].map(b => (
              <div key={b.title} style={{
                display: 'flex', gap: '.75rem', padding: '.75rem .85rem',
                borderRadius: 'var(--radius-sm)', background: 'var(--surface-raised)',
                border: '1px solid var(--border)'
              }}>
                <CheckCircle2 size={16} color="var(--accent)" style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <div style={{ fontSize: '.82rem', fontWeight: 600, color: 'var(--text)' }}>{b.title}</div>
                  <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', marginTop: 2 }}>{b.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Dataset */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <Database size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Benchmark Dataset: PVEL-AD
            </span>
          </div>
          <p style={{ fontSize: '.85rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>
            Trained and benchmarked on the international Photovoltaic Electroluminescence Anomaly Detection (PVEL-AD) dataset,
            encompassing diverse commercial multi-crystalline and mono-crystalline PV panels.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
            {DATASET_FACTS.map(f => (
              <div key={f.label} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '.6rem .85rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--surface-raised)', border: '1px solid var(--border)',
                fontSize: '.82rem'
              }}>
                <span style={{ color: 'var(--text-muted)' }}>{f.label}</span>
                <span style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{f.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Architecture & Defect Classification */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Model Architecture */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <Cpu size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Model Architecture & Benchmarks
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
            {MODEL_FACTS.map(f => (
              <div key={f.label} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '.6rem .85rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--surface-raised)', border: '1px solid var(--border)',
                fontSize: '.82rem'
              }}>
                <span style={{ color: 'var(--text-muted)' }}>{f.label}</span>
                <span style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{f.value}</span>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '.5rem' }}>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600, marginBottom: '.6rem' }}>
              Framework Stack
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem' }}>
              {['PyTorch 2.x', 'CUDA 12', 'Torchvision', 'cGAN Engine', 'Grad-CAM', 'OpenCV', 'React 18', 'Vite', 'Lucide'].map(t => (
                <Badge key={t} variant="outline" style={{ fontSize: '.75rem' }}>{t}</Badge>
              ))}
            </div>
          </div>
        </div>

        {/* Defect Classes */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <ShieldCheck size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Taxonomy of Defect Classes
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            {DEFECT_CLASSES.map(d => (
              <div key={d.name} style={{
                display: 'flex', alignItems: 'center', gap: '.75rem',
                padding: '.55rem .85rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--surface-raised)', border: '1px solid var(--border)'
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: d.name === 'Healthy Panel' ? 'var(--status-pass)' : 'var(--accent)',
                  flexShrink: 0
                }} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--text)', fontSize: '.82rem' }}>{d.name}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '.73rem' }}>{d.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Execution Pipeline Workflow */}
      <div className="card" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', marginBottom: '1.5rem' }}>
          <Layers size={18} color="var(--accent)" />
          <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
            End-to-End Inference Pipeline
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
          {[
            { step: '01', icon: Camera, title: 'EL Image Capture', desc: 'High-contrast electroluminescence or thermal capture upload' },
            { step: '02', icon: Sliders, title: 'Standardization', desc: 'Resize to 224×224 px, ImageNet normalization tensor' },
            { step: '03', icon: Cpu, title: 'DCNN Evaluation', desc: 'Forward pass through dense feature extractors on GPU/CPU' },
            { step: '04', icon: Sparkles, title: 'Grad-CAM Localization', desc: 'Gradient backpropagation to bottleneck feature activation map' },
            { step: '05', icon: FileText, title: 'Decision Synthesis', desc: 'Severity triage, confidence rating, and remediation steps' },
          ].map(s => {
            const Icon = s.icon
            return (
              <div key={s.step} style={{
                padding: '1.25rem 1rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--surface-raised)', border: '1px solid var(--border)',
                textAlign: 'center'
              }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%', margin: '0 auto .75rem',
                  background: 'var(--surface)', border: '1px solid var(--accent)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '.75rem', fontWeight: 700, color: 'var(--accent)',
                  fontFamily: 'var(--font-mono)'
                }}>
                  {s.step}
                </div>
                <Icon size={20} color="var(--accent)" style={{ margin: '0 auto .5rem', display: 'block' }} />
                <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'var(--text)', marginBottom: '.3rem' }}>
                  {s.title}
                </div>
                <div style={{ fontSize: '.74rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                  {s.desc}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Team & Academic Attribution */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Research Engineering Team */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <Users size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Project Engineering Team
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            {TEAM.map(m => (
              <div key={m.name} style={{
                display: 'flex', alignItems: 'center', gap: '.85rem',
                padding: '.75rem .9rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--surface-raised)', border: '1px solid var(--border)'
              }}>
                <div style={{
                  width: 38, height: 38, borderRadius: 'var(--radius-sm)',
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 700, fontSize: '.8rem', color: 'var(--accent)',
                  fontFamily: 'var(--font-mono)', flexShrink: 0
                }}>
                  {m.initials}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '.85rem', color: 'var(--text)' }}>{m.name}</span>
                    <span style={{ fontSize: '.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {m.roll}
                    </span>
                  </div>
                  <div style={{ fontSize: '.74rem', color: 'var(--accent)', marginTop: 2 }}>{m.role}</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{
            padding: '.85rem 1rem',
            background: 'var(--surface-raised)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)', marginTop: '.5rem'
          }}>
            <div style={{ fontSize: '.8rem', fontWeight: 600, color: 'var(--text)', marginBottom: '.25rem' }}>
              Academic Affiliation
            </div>
            <p style={{ fontSize: '.75rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
              Final Year Capstone Project — Department of Computer Science & Engineering, Amrita Vishwa Vidyapeetham.
            </p>
          </div>
        </div>

        {/* References */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '.75rem' }}>
            <BookOpen size={18} color="var(--accent)" />
            <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1rem' }}>
              Academic Literature & Citations
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
            {REFS.map((r, i) => (
              <div key={i} style={{
                display: 'flex', gap: '.6rem', fontSize: '.78rem',
                lineHeight: 1.5, color: 'var(--text-muted)'
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontWeight: 600, flexShrink: 0 }}>
                  [{i + 1}]
                </span>
                <span>{r}</span>
              </div>
            ))}
          </div>

          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '.85rem', marginTop: '.5rem' }}>
            <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600, marginBottom: '.6rem' }}>
              Dataset & Specification Resources
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem' }}>
              {[
                { label: 'PVEL-AD Benchmark', href: 'https://github.com/binyisu/PVEL-AD' },
                { label: 'DenseNet-121 Architecture', href: 'https://arxiv.org/abs/1608.06993' },
                { label: 'Grad-CAM Formulation', href: 'https://arxiv.org/abs/1610.02391' },
              ].map(l => (
                <a 
                  key={l.label} 
                  href={l.href} 
                  target="_blank" 
                  rel="noreferrer"
                  className="btn btn-outline" 
                  style={{ fontSize: '.75rem', padding: '.35rem .7rem', display: 'inline-flex', alignItems: 'center', gap: '.35rem' }}
                >
                  {l.label} <ExternalLink size={12} />
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
