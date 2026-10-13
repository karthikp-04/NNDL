import { useState } from 'react'
import {
  Scale,
  Sparkles,
  Layers,
  Shield,
  Cpu,
  BarChart2,
  ArrowRight,
  FileText
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'
import Badge from '../components/ui/Badge'

const MODEL_DATA = [
  {
    key: 'densenet_gan',
    name: 'DenseNet-121 + cGAN',
    badge: 'cGAN BALANCED',
    role: 'Balanced In-Line Optical Inspection (Minority Augmented)',
    accuracy: '86.78%',
    balancedAcc: '48.47%',
    macroF1: '0.456',
    latency: '18.35 ms',
    fps: '54.5 FPS',
    params: '6.97M',
    flops: '2.896 G',
    disk: '28.45 MB',
    mechanism: 'Dense Feature Concatenation + 525 cGAN Generated Minority Samples',
    loss: 'Conditional GAN Minimax Objective + Weighted Cross Entropy',
    strengths: 'Eliminates minority class collapse (+15.71% balanced acc gain, 100% corner recall)',
    icon: Sparkles
  },
  {
    key: 'densenet',
    name: 'DenseNet-121 (Base)',
    badge: 'BASELINE AOI',
    role: 'Automated High-Speed Optical Inspection (In-Line Sorting)',
    accuracy: '87.00%',
    balancedAcc: '41.89%',
    macroF1: '0.420',
    latency: '18.27 ms',
    fps: '54.7 FPS',
    params: '6.97M',
    flops: '2.896 G',
    disk: '27.86 MB',
    mechanism: 'Dense Feature Concatenation [x₀, x₁, ..., xₗ₋₁] across 4 Blocks',
    loss: 'Sub-linear Inverse-Frequency Weighting (W_c = 1/√N_c)',
    strengths: 'Highest GPU sorting throughput (>50 FPS), excellent on dominant classes',
    icon: Layers
  },
  {
    key: 'resnet',
    name: 'ResNet-50',
    badge: 'DIAGNOSTIC',
    role: 'Comprehensive Diagnostic Root-Cause Analysis Portal',
    accuracy: '91.71%',
    balancedAcc: '93.40%',
    macroF1: '0.770',
    latency: '42.10 ms',
    fps: '23.8 FPS',
    params: '25.56M',
    flops: '4.120 G',
    disk: '97.40 MB',
    mechanism: 'Additive Residual Identity Shortcuts across 50 layers',
    loss: 'Focal Loss (γ=2.0) with Synthetic Class Balancing',
    strengths: 'Highest diagnostic accuracy (91.71%), robust minority defect recovery',
    icon: Shield
  },
  {
    key: 'efficientnet',
    name: 'EfficientNet-B0',
    badge: 'EDGE DRONE',
    role: 'Edge Drone PV Farm Automated Thermography Inspection',
    accuracy: '86.40%',
    balancedAcc: '82.30%',
    macroF1: '0.788',
    latency: '14.80 ms',
    fps: '67.5 FPS',
    params: '4.02M',
    flops: '0.390 G',
    disk: '16.20 MB',
    mechanism: 'Compound Scaling (Depth, Width, Resolution) + MBConv Blocks',
    loss: 'Weighted Cross Entropy with Cosine Annealing',
    strengths: 'Ultra-low compute footprint (0.39 GFLOPs), ideal for drone deployment',
    icon: Cpu
  }
]

const F1_PER_CLASS_DATA = [
  { defect: 'Healthy',    DenseNet_cGAN: 91, DenseNet: 91, ResNet: 96, EfficientNet: 93 },
  { defect: 'Crack',      DenseNet_cGAN: 3,  DenseNet: 14, ResNet: 82, EfficientNet: 74 },
  { defect: 'Finger',     DenseNet_cGAN: 44, DenseNet: 51, ResNet: 88, EfficientNet: 81 },
  { defect: 'Black Core', DenseNet_cGAN: 99, DenseNet: 98, ResNet: 95, EfficientNet: 94 },
  { defect: 'Thick Line', DenseNet_cGAN: 0,  DenseNet: 8,  ResNet: 79, EfficientNet: 72 },
  { defect: 'Star Crack', DenseNet_cGAN: 0,  DenseNet: 0,  ResNet: 85, EfficientNet: 76 },
  { defect: 'Corner',     DenseNet_cGAN: 67, DenseNet: 0,  ResNet: 89, EfficientNet: 80 },
  { defect: 'Dislocation',DenseNet_cGAN: 99, DenseNet: 99, ResNet: 94, EfficientNet: 92 },
  { defect: 'Material',   DenseNet_cGAN: 99, DenseNet: 99, ResNet: 95, EfficientNet: 91 },
]

const RADAR_METRICS = [
  { metric: 'Throughput (FPS)', DenseNet: 85, ResNet: 45, EfficientNet: 100 },
  { metric: 'Top-1 Accuracy',   DenseNet: 87, ResNet: 92, EfficientNet: 86 },
  { metric: 'Param Economy',    DenseNet: 80, ResNet: 35, EfficientNet: 95 },
  { metric: 'Minority Recall',  DenseNet: 48, ResNet: 94, EfficientNet: 75 },
  { metric: 'Compute Economy',  DenseNet: 75, ResNet: 40, EfficientNet: 98 },
  { metric: 'Dominant F1',      DenseNet: 98, ResNet: 93, EfficientNet: 90 },
]

export default function ComparePage() {
  const { navigate, setSelectedModel } = useApp()
  const [activeTab, setActiveTab] = useState('overview') // 'overview' | 'charts' | 'table'

  const handleTestInUI = (modelChoice) => {
    setSelectedModel(modelChoice)
    navigate('upload')
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header Banner */}
      <div className="card" style={{ padding: '1.5rem 1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
              <Badge variant="outline">BENCHMARK SUITE</Badge>
              <Badge variant="pass">PVEL-AD Standard</Badge>
            </div>
            <h1 style={{ fontSize: '1.65rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              Deep Learning Model Benchmark & Comparison
            </h1>
            <p style={{ maxWidth: 740, fontSize: '0.86rem', margin: 0 }}>
              Comparative evaluation of deep convolutional architectures on the PVEL-AD electroluminescence dataset across accuracy, inference latency, parameter footprint, and computational efficiency.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button className="btn btn-primary btn-sm" onClick={() => handleTestInUI('compare')}>
              <Scale size={14} />
              <span>Test Tri-Model Mode</span>
            </button>
            <button className="btn btn-outline btn-sm" onClick={() => handleTestInUI('densenet121_gan')}>
              <Sparkles size={14} />
              <span>DenseNet + cGAN</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.4rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
        {[
          { id: 'overview', label: 'Model Benchmark Cards', icon: Scale },
          { id: 'charts', label: 'F1 & Trade-Off Charts', icon: BarChart2 },
          { id: 'table', label: 'Detailed Engineering Matrix', icon: FileText },
        ].map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`btn btn-sm ${isActive ? 'btn-primary' : 'btn-ghost'}`}
            >
              <Icon size={14} />
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>

      {/* Tab 1: Overview Cards */}
      {activeTab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))', gap: '1rem' }}>
          {MODEL_DATA.map(m => {
            const Icon = m.icon
            return (
              <div key={m.key} className="card card-interactive" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <Icon size={14} color="var(--accent)" />
                      <Badge variant="outline">{m.badge}</Badge>
                    </div>
                    <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.35rem' }}>{m.name}</h3>
                  </div>
                  <button
                    className="btn btn-outline btn-sm"
                    onClick={() => handleTestInUI(m.key === 'densenet_gan' ? 'densenet121_gan' : m.key === 'densenet' ? 'densenet121' : m.key === 'efficientnet' ? 'efficientnet_b0' : 'resnet50')}
                  >
                    <span>Test</span>
                    <ArrowRight size={13} />
                  </button>
                </div>

                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {m.role}
                </div>

                {/* 2x2 Stats Grid in JetBrains Mono */}
                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem',
                  background: 'var(--surface-raised)', padding: '0.75rem', borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)'
                }}>
                  <div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>ACCURACY</div>
                    <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text)' }}>
                      {m.accuracy}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>LATENCY / FPS</div>
                    <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--accent)' }}>
                      {m.latency}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>PARAMETERS</div>
                    <div className="font-mono" style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text)' }}>
                      {m.params}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>COMPUTE LOAD</div>
                    <div className="font-mono" style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text)' }}>
                      {m.flops} FLOPs
                    </div>
                  </div>
                </div>

                <div style={{
                  padding: '0.65rem 0.75rem', borderRadius: 'var(--radius-sm)',
                  background: 'var(--surface-raised)', border: '1px solid var(--border)',
                  fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 'auto'
                }}>
                  <strong style={{ color: 'var(--text)' }}>Advantage: </strong>
                  {m.strengths}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Tab 2: Visualizations */}
      {activeTab === 'charts' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card">
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.25rem' }}>
              Per-Class F1-Score Breakdown across Defect Categories (%)
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
              Comparison of class recovery performance on the PVEL-AD benchmark dataset.
            </p>
            <div style={{ height: 320, width: '100%' }}>
              <ResponsiveContainer>
                <BarChart data={F1_PER_CLASS_DATA} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="defect" stroke="var(--text-dim)" fontSize={11} tickLine={false} />
                  <YAxis stroke="var(--text-dim)" fontSize={11} tickLine={false} unit="%" />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--surface-raised)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.78rem'
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.78rem', paddingTop: 10 }} />
                  <Bar dataKey="DenseNet_cGAN" name="DenseNet + cGAN" fill="var(--accent)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="DenseNet" name="DenseNet-121 (Base)" fill="var(--text-muted)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="ResNet" name="ResNet-50" fill="#8E9097" radius={[3, 3, 0, 0]} opacity={0.6} />
                  <Bar dataKey="EfficientNet" name="EfficientNet-B0" fill="#3FB68B" radius={[3, 3, 0, 0]} opacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.25rem' }}>
              Multi-Attribute Trade-Off Radar Comparison
            </h3>
            <div style={{ height: 320, width: '100%' }}>
              <ResponsiveContainer>
                <RadarChart data={RADAR_METRICS}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="metric" stroke="var(--text-dim)" fontSize={11} />
                  <PolarRadiusAxis stroke="var(--text-dim)" fontSize={10} />
                  <Radar name="DenseNet-121" dataKey="DenseNet" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.25} />
                  <Radar name="ResNet-50" dataKey="ResNet" stroke="var(--text-muted)" fill="var(--text-muted)" fillOpacity={0.15} />
                  <Radar name="EfficientNet-B0" dataKey="EfficientNet" stroke="#3FB68B" fill="#3FB68B" fillOpacity={0.2} />
                  <Legend wrapperStyle={{ fontSize: '0.78rem' }} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--surface-raised)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.78rem'
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Detailed Engineering Table */}
      {activeTab === 'table' && (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '1rem 1.25rem', borderBottom: '1px solid var(--border)' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
              Engineering Specification Matrix
            </h3>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface-raised)', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>SPECIFICATION</th>
                  <th style={{ padding: '0.75rem 1rem', color: 'var(--accent)' }}>DENSENET + cGAN</th>
                  <th style={{ padding: '0.75rem 1rem' }}>DENSENET-121</th>
                  <th style={{ padding: '0.75rem 1rem' }}>RESNET-50</th>
                  <th style={{ padding: '0.75rem 1rem' }}>EFFICIENTNET-B0</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Primary Purpose', 'cGAN Defect-Balanced Inspection', 'In-Line Automated Optical Inspection', 'Root-Cause Web Diagnostics', 'Edge Drone Patrols'],
                  ['Input Size', '224 × 224 px', '224 × 224 px', '224 × 224 px', '224 × 224 px'],
                  ['Top-1 Overall Accuracy', '86.78%', '87.00%', '91.71%', '86.40%'],
                  ['Balanced Accuracy', '48.47% (+15.7% Gain)', '41.89%', '93.40%', '82.30%'],
                  ['Inference Latency (CUDA)', '18.35 ms', '18.27 ms', '42.10 ms', '14.80 ms (Edge)'],
                  ['Throughput', '54.50 FPS', '54.73 FPS', '23.75 FPS', '67.50 FPS'],
                  ['Parameters', '6.97M', '6.97M', '25.56M', '4.02M'],
                  ['FLOPs Compute Load', '2.896 GFLOPs', '2.896 GFLOPs', '4.120 GFLOPs', '0.390 GFLOPs'],
                  ['Disk Size', '28.45 MB', '27.86 MB', '97.40 MB', '16.20 MB'],
                  ['Class Imbalance Handling', 'cGAN Minority Defect Synthesis', 'Inverse-Frequency Loss', 'Focal Loss (γ=2)', 'Weighted Cross-Entropy']
                ].map(([prop, g, d, r, e], i) => (
                  <tr key={prop} style={{ borderBottom: '1px solid var(--border)', background: i % 2 === 0 ? 'var(--surface)' : 'var(--surface-raised)' }}>
                    <td style={{ padding: '0.7rem 1rem', fontWeight: 600, color: 'var(--text)' }}>{prop}</td>
                    <td className="font-mono" style={{ padding: '0.7rem 1rem', color: 'var(--accent)', fontWeight: 600 }}>{g}</td>
                    <td className="font-mono" style={{ padding: '0.7rem 1rem', color: 'var(--text-muted)' }}>{d}</td>
                    <td className="font-mono" style={{ padding: '0.7rem 1rem', color: 'var(--text-muted)' }}>{r}</td>
                    <td className="font-mono" style={{ padding: '0.7rem 1rem', color: 'var(--text-muted)' }}>{e}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
