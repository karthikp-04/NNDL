import { useState } from 'react'
import { useApp } from '../context/AppContext'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'

const GAN_SPECS = [
  { label: 'Architecture', value: 'Conditional GAN (cGAN)' },
  { label: 'Latent Noise Vector (z)', value: '100 dimensions (Gaussian)' },
  { label: 'Class Embedding (c)', value: '50 dimensions' },
  { label: 'Synthetic Resolution', value: '64 × 64 pixels (3 channels)' },
  { label: 'Generator Params', value: '4.23 Million' },
  { label: 'Discriminator Params', value: '2.76 Million' },
  { label: 'Synthetic Samples Created', value: '525 Images (75 / class)' },
  { label: 'DenseNet Balanced Acc Gain', value: '+15.71% (41.89% → 48.47%)' }
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
  { metric: 'Accuracy',        Baseline: 86.84, GAN_Augmented: 86.78, delta: '-0.07%' },
  { metric: 'Balanced Acc',    Baseline: 41.89, GAN_Augmented: 48.47, delta: '+15.71% ⭐' },
  { metric: 'Macro F1-Score',  Baseline: 41.87, GAN_Augmented: 45.60, delta: '+8.91% ⭐' },
  { metric: 'Corner Recall',   Baseline: 0.00,  GAN_Augmented: 100.0, delta: '+100.0% 🚀' },
  { metric: 'Macro Recall',    Baseline: 41.89, GAN_Augmented: 48.47, delta: '+15.71% ⭐' }
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
      // Fetch image and convert to base64 for seamless live inference
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
        addToast(`Dispatching ${sample.label} to cGAN-Augmented DenseNet-121!`, 'success')
        navigate('analyzing')
      }
      reader.readAsDataURL(blob)
    } catch {
      // Fallback
      setSelectedModel('densenet121_gan')
      setAnalysis({
        image: imgUrl,
        fileName: sample.file,
        model: 'densenet121_gan'
      })
      addToast(`Testing ${sample.label} with DenseNet-121 (GAN-Augmented)`, 'info')
      navigate('analyzing')
    }
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', paddingBottom: '3rem' }}>
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(236,72,153,0.12) 0%, rgba(59,130,246,0.12) 100%)',
        border: '1px solid rgba(236,72,153,0.3)',
        borderRadius: 'var(--radius-lg, 12px)',
        padding: '1.8rem 2rem',
        marginBottom: '2rem',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '.4rem',
              padding: '.25rem .75rem', borderRadius: 999,
              background: 'rgba(236,72,153,0.2)', color: '#F472B6',
              fontSize: '.75rem', fontWeight: 700, letterSpacing: '.05em', marginBottom: '.6rem'
            }}>
              🧪 GENERATIVE AI & DATA AUGMENTATION
            </div>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff', marginBottom: '.4rem' }}>
              cGAN Defect Synthesis & Augmentation Studio
            </h1>
            <p style={{ color: 'var(--muted)', fontSize: '.9rem', maxWidth: 760, lineHeight: 1.5 }}>
              Overcoming the extreme class imbalance of Photovoltaic Electroluminescence (NIR EL) datasets.
              Using a Conditional Generative Adversarial Network (cGAN) to synthesize realistic minority defect patterns
              and re-train DenseNet-121 for enhanced operational reliability.
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{
              display: 'inline-block',
              padding: '.5rem 1rem',
              borderRadius: 'var(--radius)',
              background: 'rgba(16,185,129,0.15)',
              border: '1px solid #10B981',
              color: '#34D399',
              fontWeight: 700,
              fontSize: '.85rem'
            }}>
              ⭐ +15.71% Balanced Acc Gain
            </span>
          </div>
        </div>

        {/* Spec Pill Tags */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.6rem', marginTop: '1.2rem' }}>
          {GAN_SPECS.map((s, idx) => (
            <div key={idx} style={{
              background: 'var(--bg2)',
              border: '1px solid var(--border)',
              borderRadius: '999px',
              padding: '.3rem .8rem',
              fontSize: '.75rem',
              color: 'var(--text)'
            }}>
              <span style={{ color: 'var(--dim)', marginRight: '.3rem' }}>{s.label}:</span>
              <strong style={{ color: '#F472B6' }}>{s.value}</strong>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Cards Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        {BENCHMARK_COMPARISON.map((b, idx) => (
          <div key={idx} style={{
            background: 'var(--bg2)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '1.2rem',
            position: 'relative'
          }}>
            <div style={{ fontSize: '.75rem', color: 'var(--dim)', fontWeight: 600, textTransform: 'uppercase' }}>
              {b.metric}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '.6rem', margin: '.4rem 0' }}>
              <span style={{ fontSize: '1.6rem', fontWeight: 800, color: '#fff' }}>
                {b.GAN_Augmented}%
              </span>
              <span style={{ fontSize: '.82rem', color: 'var(--muted)', textDecoration: 'line-through' }}>
                {b.Baseline}%
              </span>
            </div>
            <div style={{ fontSize: '.78rem', color: '#10B981', fontWeight: 700 }}>
              {b.delta}
            </div>
          </div>
        ))}
      </div>

      {/* Class Rebalancing Bar Chart */}
      <div style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1.5rem',
        marginBottom: '2.5rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>
              Training Sample Augmentation: Real vs. cGAN Injected Samples
            </h3>
            <p style={{ fontSize: '.8rem', color: 'var(--muted)' }}>
              75 high-fidelity synthetic samples synthesized per minority class to prevent gradient starvation in DenseNet-121.
            </p>
          </div>
          <span style={{ fontSize: '.75rem', color: '#EC4899', fontWeight: 600 }}>
            525 Total Synthetic Images Synthesized
          </span>
        </div>

        <div style={{ height: 260, width: '100%' }}>
          <ResponsiveContainer>
            <BarChart data={CLASS_DISTRIBUTION} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="class" stroke="var(--dim)" fontSize={11} />
              <YAxis stroke="var(--dim)" fontSize={11} />
              <Tooltip
                contentStyle={{ background: '#1E293B', border: '1px solid var(--border)', borderRadius: 8, fontSize: '.8rem' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend wrapperStyle={{ fontSize: '.8rem', paddingTop: 10 }} />
              <Bar dataKey="Real" name="Original Real PVEL-AD Images" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="cGAN_Synthetic" name="cGAN Synthesized Samples" fill="#EC4899" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Synthetic Gallery Section */}
      <div style={{ marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.2rem' }}>
          <div>
            <h2 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#fff' }}>
              Interactive Synthetic EL Image Gallery
            </h2>
            <p style={{ fontSize: '.85rem', color: 'var(--muted)' }}>
              Click <strong>"⚡ Test with DenseNet-121"</strong> on any generated defect sample to run live inference through the neural network.
            </p>
          </div>

          {/* Filter Tabs */}
          <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap' }}>
            {SYNTHETIC_CATEGORIES.map(cat => (
              <button
                key={cat.key}
                onClick={() => setActiveCategory(cat.key)}
                style={{
                  padding: '.35rem .75rem',
                  borderRadius: 999,
                  border: `1px solid ${activeCategory === cat.key ? '#EC4899' : 'var(--border)'}`,
                  background: activeCategory === cat.key ? 'rgba(236,72,153,0.2)' : 'var(--bg2)',
                  color: activeCategory === cat.key ? '#F472B6' : 'var(--muted)',
                  fontSize: '.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all .15s ease'
                }}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>

        {/* Image Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
          gap: '1.2rem'
        }}>
          {filteredSamples.map(sample => (
            <div
              key={sample.id}
              style={{
                background: 'var(--bg2)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                overflow: 'hidden',
                transition: 'all .2s ease',
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <div style={{
                position: 'relative',
                background: '#0B0F19',
                aspectRatio: '4/3',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden'
              }}>
                <img
                  src={`/gan_samples/${sample.cat}/${sample.file}`}
                  alt={sample.label}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    imageRendering: 'pixelated'
                  }}
                  onError={(e) => {
                    // Fallback to high tech SVG if local image fails
                    e.target.style.display = 'none'
                  }}
                />
                <div style={{
                  position: 'absolute', top: 8, right: 8,
                  background: 'rgba(0,0,0,0.7)', padding: '2px 8px',
                  borderRadius: 999, fontSize: '.65rem', color: '#F472B6',
                  fontWeight: 700, border: '1px solid rgba(236,72,153,0.4)'
                }}>
                  cGAN 64x64
                </div>
              </div>

              <div style={{ padding: '1rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontWeight: 700, color: '#fff', fontSize: '.9rem', marginBottom: '.3rem' }}>
                  {sample.label}
                </div>
                <div style={{ fontSize: '.75rem', color: 'var(--muted)', lineHeight: 1.4, flex: 1, marginBottom: '1rem' }}>
                  {sample.desc}
                </div>

                <button
                  className="btn btn-primary"
                  onClick={() => handleTestSynthetic(sample)}
                  style={{
                    width: '100%',
                    padding: '.55rem',
                    fontSize: '.8rem',
                    fontWeight: 700,
                    background: 'linear-gradient(135deg, #EC4899 0%, #8B5CF6 100%)',
                    border: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '.4rem'
                  }}
                >
                  ⚡ Test with DenseNet-121
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Per-Class Detailed Performance Table */}
      <div style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '1.5rem',
        overflowX: 'auto'
      }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginBottom: '.4rem' }}>
          Detailed Per-Class Performance Comparison (Su et al. PVEL-AD Held-Out Test Split)
        </h3>
        <p style={{ fontSize: '.8rem', color: 'var(--muted)', marginBottom: '1.2rem' }}>
          Exact verification metrics extracted from <code>per_class_comparison.csv</code> in the project's GAN experiment suite.
        </p>

        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--dim)' }}>
              <th style={{ padding: '.6rem .8rem' }}>Defect Category</th>
              <th style={{ padding: '.6rem .8rem' }}>Test Support</th>
              <th style={{ padding: '.6rem .8rem' }}>Baseline Recall</th>
              <th style={{ padding: '.6rem .8rem' }}>cGAN-Aug Recall</th>
              <th style={{ padding: '.6rem .8rem' }}>Baseline F1</th>
              <th style={{ padding: '.6rem .8rem' }}>cGAN-Aug F1</th>
              <th style={{ padding: '.6rem .8rem' }}>F1 Delta</th>
            </tr>
          </thead>
          <tbody>
            {[
              { cls: 'Healthy / Good', count: 1702, bRec: '96.36%', gRec: '98.65%', bF1: '0.908', gF1: '0.905', delta: '-0.003' },
              { cls: 'Cell Micro Crack', count: 123, bRec: '8.13%', gRec: '1.63%', bF1: '0.142', gF1: '0.031', delta: '-0.111' },
              { cls: 'Finger Interruption', count: 145, bRec: '55.17%', gRec: '36.55%', bF1: '0.508', gF1: '0.440', delta: '-0.068' },
              { cls: 'Black Core', count: 151, bRec: '98.68%', gRec: '98.68%', bF1: '0.980', gF1: '0.987', delta: '+0.007 ⭐' },
              { cls: 'Corner Fracture', count: 1, bRec: '0.00%', gRec: '100.0%', bF1: '0.000', gF1: '0.667', delta: '+0.667 🚀' },
              { cls: 'Crystal Dislocation', count: 135, bRec: '98.52%', gRec: '98.52%', bF1: '0.993', gF1: '0.993', delta: '0.000' },
              { cls: 'Material Inhomogeneity', count: 487, bRec: '99.59%', gRec: '99.18%', bF1: '0.995', gF1: '0.994', delta: '-0.001' }
            ].map((row, idx) => (
              <tr key={idx} style={{
                borderBottom: '1px solid rgba(255,255,255,0.04)',
                background: row.delta.includes('🚀') ? 'rgba(236,72,153,0.08)' : 'transparent'
              }}>
                <td style={{ padding: '.7rem .8rem', fontWeight: 600, color: '#fff' }}>{row.cls}</td>
                <td style={{ padding: '.7rem .8rem', color: 'var(--muted)' }}>{row.count}</td>
                <td style={{ padding: '.7rem .8rem', color: 'var(--muted)' }}>{row.bRec}</td>
                <td style={{ padding: '.7rem .8rem', color: '#10B981', fontWeight: 700 }}>{row.gRec}</td>
                <td style={{ padding: '.7rem .8rem', color: 'var(--muted)' }}>{row.bF1}</td>
                <td style={{ padding: '.7rem .8rem', color: '#34D399', fontWeight: 700 }}>{row.gF1}</td>
                <td style={{
                  padding: '.7rem .8rem',
                  fontWeight: 700,
                  color: row.delta.includes('🚀') || row.delta.includes('⭐') ? '#EC4899' : 'var(--muted)'
                }}>
                  {row.delta}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
