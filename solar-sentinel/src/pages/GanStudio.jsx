import { useState } from 'react'
import { Play } from 'lucide-react'
import { useApp } from '../context/AppContext'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import Badge from '../components/ui/Badge'

const GAN_SPECS = [
  { label: 'Architecture', value: 'Conditional GAN (cGAN)' },
  { label: 'Latent Vector (z)', value: '100 dimensions (Gaussian)' },
  { label: 'Class Embedding', value: '50 dimensions' },
  { label: 'Synthetic Resolution', value: '64 × 64 pixels (3-ch)' },
  { label: 'Generator Params', value: '4.23M' },
  { label: 'Discriminator Params', value: '2.76M' },
  { label: 'Synthetic Dataset', value: '525 Images (75 / class)' },
  { label: 'DenseNet Bal Acc', value: '+15.71% Gain (41.89% → 48.47%)' }
]

const CLASS_DISTRIBUTION = [
  { class: 'Crack',       Real: 123, cGAN_Synthetic: 75, Total: 198 },
  { class: 'Finger',      Real: 145, cGAN_Synthetic: 75, Total: 220 },
  { class: 'Black Core',  Real: 151, cGAN_Synthetic: 75, Total: 226 },
  { class: 'Thick Line',  Real: 116, cGAN_Synthetic: 75, Total: 191 },
  { class: 'Star Crack',  Real: 19,  cGAN_Synthetic: 75, Total: 94 },
  { class: 'Dislocation', Real: 135, cGAN_Synthetic: 75, Total: 210 },
  { class: 'Material',    Real: 487, cGAN_Synthetic: 75, Total: 562 }
]

const BENCHMARK_COMPARISON = [
  { metric: 'Overall Accuracy', Baseline: '86.84%', GAN_Augmented: '86.78%', delta: '-0.07%', note: 'Preserved' },
  { metric: 'Balanced Accuracy', Baseline: '41.89%', GAN_Augmented: '48.47%', delta: '+15.71%', note: 'Significant Gain' },
  { metric: 'Macro F1-Score',  Baseline: '41.87%', GAN_Augmented: '45.60%', delta: '+8.91%', note: 'Improved' },
  { metric: 'Corner Recall',   Baseline: '0.00%',  GAN_Augmented: '100.0%', delta: '+100.0%', note: 'Starvation Resolved' },
  { metric: 'Macro Recall',    Baseline: '41.89%', GAN_Augmented: '48.47%', delta: '+15.71%', note: 'Fairness Gain' }
]

const SYNTHETIC_CATEGORIES = [
  { key: 'all', label: 'All Classes' },
  { key: 'synthetic_crack', label: 'Micro Crack', classKey: 'crack' },
  { key: 'synthetic_star_crack', label: 'Star Crack', classKey: 'star_crack' },
  { key: 'synthetic_black_core', label: 'Black Core', classKey: 'black_core' },
  { key: 'synthetic_finger', label: 'Finger Interruption', classKey: 'finger' },
  { key: 'synthetic_thick_line', label: 'Thick Line', classKey: 'thick_line' },
  { key: 'synthetic_dislocation', label: 'Dislocation', classKey: 'dislocation' },
  { key: 'synthetic_meterial', label: 'Material Inhomogeneity', classKey: 'meterial' }
]

const SAMPLE_FILES = [
  { id: 1, cat: 'synthetic_crack', file: 'synthetic_cgan_crack_0001.png', label: 'Synthetic Micro Crack #1', desc: 'Hairline stress fracture synthesized across wafer lattice' },
  { id: 2, cat: 'synthetic_crack', file: 'synthetic_cgan_crack_0002.png', label: 'Synthetic Micro Crack #2', desc: 'Sub-millimeter crack branch synthesized from latent vector' },
  { id: 3, cat: 'synthetic_star_crack', file: 'synthetic_cgan_star_crack_0001.png', label: 'Synthetic Star Crack #1', desc: 'Radial hail-impact fracture generated with multi-directional fissures' },
  { id: 4, cat: 'synthetic_star_crack', file: 'synthetic_cgan_star_crack_0002.png', label: 'Synthetic Star Crack #2', desc: 'High-severity point-impact pattern synthesized on EL background' },
  { id: 5, cat: 'synthetic_black_core', file: 'synthetic_cgan_black_core_0001.png', label: 'Synthetic Black Core #1', desc: 'Inactive non-luminescent crystal cluster with high contrast gradient' },
  { id: 6, cat: 'synthetic_black_core', file: 'synthetic_cgan_black_core_0002.png', label: 'Synthetic Black Core #2', desc: 'Center-cell dead region causing bypass diode escalation' },
  { id: 7, cat: 'synthetic_finger', file: 'synthetic_cgan_finger_0001.png', label: 'Synthetic Finger Defect #1', desc: 'Interrupted silver contact grid lines synthesized along busbar' },
  { id: 8, cat: 'synthetic_thick_line', file: 'synthetic_cgan_thick_line_0001.png', label: 'Synthetic Thick Line #1', desc: 'Soldering paste excess and ribbon thickening defect' },
  { id: 9, cat: 'synthetic_dislocation', file: 'synthetic_cgan_dislocation_0001.png', label: 'Synthetic Dislocation #1', desc: 'Silicon ingot crystal boundary anomaly with diffuse recombination' },
  { id: 10, cat: 'synthetic_meterial', file: 'synthetic_cgan_meterial_0001.png', label: 'Synthetic Material Defect #1', desc: 'Raw wafer material inhomogeneity cluster synthesized by cGAN' },
]

export default function GanStudio() {
  const { navigate, setAnalysis, setSelectedModel, addToast } = useApp()
  const [activeCategory, setActiveCategory] = useState('all')

  const filteredSamples = activeCategory === 'all'
    ? SAMPLE_FILES
    : SAMPLE_FILES.filter(s => s.cat === activeCategory)

  const handleTestSynthetic = async (sample) => {
    const imgUrl = `/gan_samples/${sample.cat}/${sample.file}`
    try {
      const res = await fetch(imgUrl)
      const blob = await res.blob()
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64data = reader.result
        setSelectedModel('densenet121_gan')
        setAnalysis({
          image: base64data,
          fileName: sample.file,
          model: 'densenet121_gan'
        })
        addToast(`Dispatching ${sample.label} to DenseNet-121 (cGAN Aug)`, 'success')
        navigate('analyzing')
      }
      reader.readAsDataURL(blob)
    } catch {
      setSelectedModel('densenet121_gan')
      setAnalysis({
        image: imgUrl,
        fileName: sample.file,
        model: 'densenet121_gan'
      })
      addToast(`Testing ${sample.label} with DenseNet-121 (cGAN Aug)`, 'info')
      navigate('analyzing')
    }
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Hero Header */}
      <div className="card" style={{ padding: '1.75rem 2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <Badge variant="outline">GENERATIVE AI & DATA AUGMENTATION</Badge>
              <Badge variant="pass">+15.71% Balanced Acc Gain</Badge>
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              cGAN Defect Synthesis & Augmentation Studio
            </h1>
            <p style={{ maxWidth: 760, fontSize: '0.88rem', lineHeight: 1.5 }}>
              Resolving class imbalance across Photovoltaic Electroluminescence (NIR EL) datasets.
              Synthesizing realistic minority defect patterns via Conditional GAN to retrain DenseNet-121 for higher recall.
            </p>
          </div>
        </div>

        {/* Spec Pill Tags */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1.25rem' }}>
          {GAN_SPECS.map((s, idx) => (
            <div key={idx} className="badge badge-outline" style={{ fontSize: '0.72rem', padding: '0.25rem 0.65rem' }}>
              <span style={{ color: 'var(--text-dim)', marginRight: '0.35rem' }}>{s.label}:</span>
              <strong className="font-mono" style={{ color: 'var(--text)' }}>{s.value}</strong>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Cards Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
        {BENCHMARK_COMPARISON.map((b, idx) => (
          <div key={idx} className="card card-raised" style={{ padding: '1rem' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontWeight: 600, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              {b.metric}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', margin: '0.35rem 0' }}>
              <span className="font-mono" style={{ fontSize: '1.45rem', fontWeight: 700, color: 'var(--text)' }}>
                {b.GAN_Augmented}
              </span>
              <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-dim)', textDecoration: 'line-through' }}>
                {b.Baseline}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="font-mono" style={{ fontSize: '0.74rem', color: 'var(--status-pass)', fontWeight: 600 }}>
                {b.delta}
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{b.note}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Class Rebalancing Bar Chart */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
              Sample Distribution: Real vs. cGAN Synthesized
            </h3>
            <p style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>
              75 synthetic samples injected per minority class to prevent gradient starvation in DenseNet-121.
            </p>
          </div>
          <Badge variant="outline">525 Synthetic Samples</Badge>
        </div>

        <div style={{ height: 260, width: '100%' }}>
          <ResponsiveContainer>
            <BarChart data={CLASS_DISTRIBUTION} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="class" stroke="var(--text-dim)" fontSize={11} tickLine={false} />
              <YAxis stroke="var(--text-dim)" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-raised)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.78rem',
                  color: 'var(--text)'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.78rem', paddingTop: 10 }} />
              <Bar dataKey="Real" name="Original PVEL-AD Images" fill="var(--text-muted)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="cGAN_Synthetic" name="cGAN Synthesized Samples" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Synthetic Gallery Section */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>
              Synthesized Electroluminescence Gallery
            </h2>
            <p style={{ fontSize: '0.82rem', marginTop: '0.2rem' }}>
              Select any generated sample to dispatch live forward pass through DenseNet-121.
            </p>
          </div>

          {/* Filter Chips */}
          <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
            {SYNTHETIC_CATEGORIES.map(cat => (
              <button
                key={cat.key}
                onClick={() => setActiveCategory(cat.key)}
                className={`preset-chip ${activeCategory === cat.key ? 'active' : ''}`}
                style={{ fontSize: '0.72rem', padding: '0.3rem 0.65rem' }}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>

        {/* Image Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '1rem' }}>
          {filteredSamples.map(sample => (
            <div
              key={sample.id}
              className="card card-interactive"
              style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}
            >
              <div style={{ position: 'relative', background: 'var(--bg)', aspectRatio: '4/3', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src={`/gan_samples/${sample.cat}/${sample.file}`}
                  alt={sample.label}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  loading="lazy"
                />
                <div style={{ position: 'absolute', top: 8, right: 8 }}>
                  <Badge variant="outline" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
                    64 × 64
                  </Badge>
                </div>
              </div>

              <div style={{ padding: '0.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.65rem', flex: 1, justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text)' }}>{sample.label}</div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{sample.desc}</div>
                </div>

                <button
                  className="btn btn-outline btn-sm"
                  onClick={() => handleTestSynthetic(sample)}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  <Play size={13} fill="currentColor" />
                  <span>Test with DenseNet-121</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
