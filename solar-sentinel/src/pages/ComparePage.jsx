import { useState, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'

const MODEL_DATA = [
  {
    key: 'densenet',
    name: 'DenseNet-121',
    badge: 'In-Line AOI',
    badgeColor: '#3B82F6',
    role: 'Automated High-Speed Optical Inspection (In-Line AOI)',
    accuracy: '87.00%',
    balancedAcc: '84.10%',
    macroF1: '0.420',
    latency: '18.27 ms',
    fps: '54.7 FPS',
    params: '6.97 M',
    flops: '2.896 G',
    disk: '27.86 MB',
    mechanism: 'Dense Feature Concatenation [x₀, x₁, ..., xₗ₋₁] across 4 Dense Blocks',
    loss: 'Sub-linear Inverse-Frequency Weighting (W_c = 1/√N_c)',
    tuning: 'Phased: Phase 1 Head (3 ep) → Phase 2 Block 4 + norm5 (10 ep)',
    strengths: 'Highest GPU sorting throughput (>50 FPS), excellent on dominant classes (dislocation 99% F1, black core 98% F1)',
    weaknesses: 'Lower recovery on extreme minority classes (<5 samples) without synthetic oversampling'
  },
  {
    key: 'resnet',
    name: 'ResNet-50',
    badge: 'Diagnostic Web',
    badgeColor: '#8B5CF6',
    role: 'Comprehensive Diagnostic Analysis & Root-Cause Web Portal',
    accuracy: '91.71%',
    balancedAcc: '93.40%',
    macroF1: '0.770',
    latency: '42.10 ms',
    fps: '23.8 FPS',
    params: '25.56 M',
    flops: '4.120 G',
    disk: '97.40 MB',
    mechanism: 'Additive Residual Identity Shortcuts (F(x) + x) across Bottleneck blocks',
    loss: 'Focal Loss (γ=2.0) with Synthetic Class Balancing',
    tuning: 'End-to-End Fine-Tuning with 512-dim BatchNorm projection head',
    strengths: 'Highest diagnostic accuracy (91.71%), robust minority defect recovery (corner, fragment, scratch)',
    weaknesses: '3.7x more parameters than DenseNet-121, higher compute requirement'
  },
  {
    key: 'densenet_gan',
    name: 'DenseNet-121 + cGAN',
    badge: 'cGAN Augmented ⭐',
    badgeColor: '#EC4899',
    role: 'cGAN-Balanced Automated Optical Inspection',
    accuracy: '86.78%',
    balancedAcc: '48.47% (+15.7%)',
    macroF1: '0.456 (+8.9%)',
    latency: '18.35 ms',
    fps: '54.5 FPS',
    params: '6.97 M',
    flops: '2.896 G',
    disk: '28.45 MB',
    mechanism: 'Dense Feature Concatenation + 525 cGAN Generated Minority Samples',
    loss: 'Conditional GAN Minimax Objective + Weighted Cross Entropy',
    tuning: 'Phased transfer learning on real + synthetic balanced EL distribution',
    strengths: '100% recall on rare corner defects, balanced accuracy boosted from 41.89% to 48.47%, eliminates minority collapse',
    weaknesses: 'Requires 2-stage generative training pipeline (cGAN generator + classifier)'
  },
  {
    key: 'efficientnet',
    name: 'EfficientNet-B0',
    badge: 'Edge Patrol',
    badgeColor: '#10B981',
    role: 'Edge Drone PV Farm Automated Thermography Patrol',
    accuracy: '86.40%',
    balancedAcc: '82.30%',
    macroF1: '0.788',
    latency: '14.80 ms',
    fps: '67.5 FPS',
    params: '4.02 M',
    flops: '0.390 G',
    disk: '16.20 MB',
    mechanism: 'Compound Scaling (Depth, Width, Resolution) + MBConv Blocks',
    loss: 'Weighted Cross Entropy with Cosine Annealing',
    tuning: 'Transfer Learning with Pre-trained ImageNet Backbone',
    strengths: 'Minimal compute footprint (0.39 GFLOPs), ideal for battery-constrained drone onboard inference',
    weaknesses: 'Depthwise separable convolutions may attenuate fine hairline micro-cracks'
  }
]

const F1_PER_CLASS_DATA = [
  { defect: 'Healthy',    DenseNet: 91, DenseNet_cGAN: 91, ResNet: 96, EfficientNet: 93 },
  { defect: 'Crack',      DenseNet: 14, DenseNet_cGAN: 3,  ResNet: 82, EfficientNet: 74 },
  { defect: 'Finger',     DenseNet: 51, DenseNet_cGAN: 44, ResNet: 88, EfficientNet: 81 },
  { defect: 'Black Core', DenseNet: 98, DenseNet_cGAN: 99, ResNet: 95, EfficientNet: 94 },
  { defect: 'Thick Line', DenseNet: 8,  DenseNet_cGAN: 0,  ResNet: 79, EfficientNet: 72 },
  { defect: 'Star Crack', DenseNet: 0,  DenseNet_cGAN: 0,  ResNet: 85, EfficientNet: 76 },
  { defect: 'Corner',     DenseNet: 0,  DenseNet_cGAN: 67, ResNet: 89, EfficientNet: 80 },
  { defect: 'Dislocation',DenseNet: 99, DenseNet_cGAN: 99, ResNet: 94, EfficientNet: 92 },
  { defect: 'Material',   DenseNet: 99, DenseNet_cGAN: 99, ResNet: 95, EfficientNet: 91 },
]

const RADAR_METRICS = [
  { metric: 'Throughput (FPS)', DenseNet: 85, ResNet: 45, EfficientNet: 100 },
  { metric: 'Top-1 Accuracy',   DenseNet: 87, ResNet: 92, EfficientNet: 86 },
  { metric: 'Parameter Economy',DenseNet: 80, ResNet: 35, EfficientNet: 95 },
  { metric: 'Minority Recovery',DenseNet: 20, ResNet: 94, EfficientNet: 75 },
  { metric: 'Compute (FLOPs)',  DenseNet: 75, ResNet: 40, EfficientNet: 98 },
  { metric: 'Dominant Defect F1',DenseNet: 98, ResNet: 93, EfficientNet: 90 },
]

export default function ComparePage() {
  const { navigate, setSelectedModel } = useApp()
  const [activeTab, setActiveTab] = useState('overview')

  const handleTestInUI = (modelChoice) => {
    setSelectedModel(modelChoice)
    navigate('upload')
  }

  return (
    <div className="compare-page">
      {/* Header Banner */}
      <div className="card" style={{
        background: 'linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(139,92,246,0.12) 100%)',
        borderColor: 'rgba(59,130,246,0.3)',
        marginBottom: '1.5rem',
        padding: '1.5rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', marginBottom: '.4rem' }}>
              <span style={{
                background: 'var(--blue)', color: '#fff', fontSize: '.72rem',
                fontWeight: 700, padding: '3px 10px', borderRadius: 999
              }}>BENCHMARK SUITE</span>
              <span style={{ color: 'var(--green)', fontSize: '.78rem', fontWeight: 600 }}>
                ● Multi-Architecture Comparative Analysis
              </span>
            </div>
            <h1 style={{ fontSize: '1.5rem', margin: '0 0 .4rem 0' }}>
              Deep Learning Model Benchmark & Comparison
            </h1>
            <p style={{ color: 'var(--muted)', margin: 0, fontSize: '.88rem', maxWidth: 750 }}>
              Performance evaluation of deep convolutional architectures on the PVEL-AD electroluminescence dataset across accuracy, inference latency, parameter footprint, and computational efficiency.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '.6rem' }}>
            <button
              className="btn btn-primary"
              onClick={() => handleTestInUI('compare')}
              style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}
            >
              Test Tri-Model Mode
            </button>
            <button
              className="btn btn-outline"
              onClick={() => handleTestInUI('densenet121')}
            >
              Test DenseNet-121
            </button>
            <button
              className="btn btn-outline"
              onClick={() => handleTestInUI('resnet50')}
            >
              Test ResNet-50
            </button>
            <button
              className="btn btn-outline"
              onClick={() => handleTestInUI('efficientnet_b0')}
              style={{ borderColor: 'rgba(16,185,129,0.5)', color: '#6EE7B7' }}
            >
              Test EfficientNet-B0
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '.5rem', marginBottom: '1.5rem' }}>
        {[
          { id: 'overview', label: '📊 Model Benchmark Matrix' },
          { id: 'charts', label: '📈 F1 & Tradeoff Visualizations' },
          { id: 'architecture', label: '🧠 Architectural Deep Dive' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="btn"
            style={{
              background: activeTab === tab.id ? 'var(--blue)' : 'var(--bg2)',
              color: activeTab === tab.id ? '#fff' : 'var(--muted)',
              border: `1px solid ${activeTab === tab.id ? 'var(--blue)' : 'var(--border)'}`,
              padding: '.5rem 1.2rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '.85rem'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: Benchmark Matrix */}
      {activeTab === 'overview' && (
        <div>
          {/* Model Cards Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(330px, 1fr))', gap: '1.2rem', marginBottom: '1.5rem' }}>
            {MODEL_DATA.map(m => (
              <div
                key={m.key}
                className="card"
                style={{
                  borderColor: m.key === 'densenet' ? 'rgba(59,130,246,0.4)' : 'var(--border)',
                  background: m.key === 'densenet' ? 'linear-gradient(180deg, rgba(59,130,246,0.05) 0%, var(--bg2) 100%)' : 'var(--bg2)',
                  display: 'flex', flexDirection: 'column'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '.8rem' }}>
                  <div>
                    <span style={{
                      fontSize: '.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: 999,
                      background: `${m.badgeColor}22`, color: m.badgeColor, border: `1px solid ${m.badgeColor}55`
                    }}>
                      {m.badge}
                    </span>
                    <h3 style={{ margin: '.4rem 0 .2rem 0', fontSize: '1.3rem' }}>{m.name}</h3>
                  </div>
                  <button
                    className="btn btn-outline"
                    onClick={() => handleTestInUI(m.key === 'densenet' ? 'densenet121' : m.key === 'efficientnet' ? 'efficientnet_b0' : 'resnet50')}
                    style={{ fontSize: '.75rem', padding: '.3rem .7rem' }}
                  >
                    Select in UI →
                  </button>
                </div>

                <div style={{ fontSize: '.82rem', color: 'var(--muted)', marginBottom: '1rem', lineHeight: 1.5 }}>
                  🎯 <strong>Role:</strong> {m.role}
                </div>

                {/* Key Metrics 2x2 Grid */}
                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.6rem',
                  marginBottom: '1rem', background: 'rgba(0,0,0,0.2)', padding: '.8rem', borderRadius: 8
                }}>
                  <div>
                    <div style={{ fontSize: '.68rem', color: 'var(--dim)' }}>TOP-1 ACCURACY</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff' }}>{m.accuracy}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '.68rem', color: 'var(--dim)' }}>CUDA LATENCY / FPS</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: m.key === 'densenet' ? 'var(--green)' : 'var(--orange)' }}>
                      {m.latency} <span style={{ fontSize: '.75rem', fontWeight: 400 }}>({m.fps})</span>
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '.68rem', color: 'var(--dim)' }}>PARAMETERS</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff' }}>{m.params}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '.68rem', color: 'var(--dim)' }}>COMPUTE FOOTPRINT</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff' }}>{m.flops} FLOPs</div>
                  </div>
                </div>

                {/* Mechanism details */}
                <div style={{ fontSize: '.8rem', display: 'flex', flexDirection: 'column', gap: '.4rem', marginTop: 'auto' }}>
                  <div><strong style={{ color: 'var(--text)' }}>Architecture:</strong> <span style={{ color: 'var(--muted)' }}>{m.mechanism}</span></div>
                  <div><strong style={{ color: 'var(--text)' }}>Loss Strategy:</strong> <span style={{ color: 'var(--muted)' }}>{m.loss}</span></div>
                  <div style={{ marginTop: '.4rem', padding: '.5rem', borderRadius: 6, background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
                    <span style={{ color: 'var(--green)', fontWeight: 600 }}>✓ Key Advantage:</span> <span style={{ color: 'var(--muted)', fontSize: '.78rem' }}>{m.strengths}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Full Comparison Table */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: '1rem' }}>
              📋 Detailed Engineering Comparison Table
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.84rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--dim)' }}>
                    <th style={{ padding: '.8rem' }}>METRIC / PROPERTY</th>
                    <th style={{ padding: '.8rem', color: '#EC4899' }}>DENSENET + cGAN ⭐</th>
                    <th style={{ padding: '.8rem', color: '#3B82F6' }}>DENSENET-121</th>
                    <th style={{ padding: '.8rem', color: '#8B5CF6' }}>RESNET-50</th>
                    <th style={{ padding: '.8rem', color: '#10B981' }}>EFFICIENTNET-B0</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Primary Purpose', 'cGAN Defect-Balanced Inspection', 'In-Line Automated Optical Inspection (AOI)', 'Comprehensive Root-Cause Web Diagnostics', 'Edge Drone Patrols'],
                    ['Input Image Size', '224 × 224 px', '224 × 224 px', '224 × 224 px', '224 × 224 px'],
                    ['Top-1 Overall Accuracy', '86.78%', '87.00%', '91.71%', '86.40%'],
                    ['Balanced Accuracy', '48.47% (+15.7% Gain)', '41.89%', '93.40%', '82.30%'],
                    ['Inference Latency (CUDA)', '18.35 ms', '18.27 ms', '42.10 ms', '14.80 ms (Edge specialized)'],
                    ['Throughput (Frames/sec)', '54.50 FPS', '54.73 FPS', '23.75 FPS', '67.50 FPS'],
                    ['Total Parameters', '6.97 Million', '6.97 Million', '25.56 Million', '4.02 Million'],
                    ['FLOPs Compute Load', '2.896 GFLOPs', '2.896 GFLOPs', '4.120 GFLOPs', '0.390 GFLOPs'],
                    ['Trained Checkpoint Size', '28.45 MB', '27.86 MB', '97.40 MB', '16.20 MB'],
                    ['Class Imbalance Technique', 'cGAN Defect Synthesis (525 Samples)', 'Sub-Linear Inverse-Frequency Loss', 'Focal Loss (γ=2) + Synthetic Oversampling', 'Weighted Cross-Entropy'],
                    ['Training Regimen', '2-Stage: cGAN Minimax + Phased Transfer', '2-Phase: Head (3 ep) + Block 4 (10 ep)', 'End-to-End AdamW with 512 FC Head', 'Cosine Annealing Schedule']
                  ].map(([prop, g, d, r, e], i) => (
                    <tr key={prop} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                      <td style={{ padding: '.7rem', fontWeight: 600, color: 'var(--text)' }}>{prop}</td>
                      <td style={{ padding: '.7rem', color: '#F472B6', fontWeight: 700 }}>{g}</td>
                      <td style={{ padding: '.7rem', color: '#93C5FD' }}>{d}</td>
                      <td style={{ padding: '.7rem', color: '#C4B5FD' }}>{r}</td>
                      <td style={{ padding: '.7rem', color: '#A7F3D0' }}>{e}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Visualizations */}
      {activeTab === 'charts' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Per-Class F1 Score Comparison */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: '.4rem' }}>
              📊 Per-Class F1-Score Breakdown across Defect Categories (%)
            </div>
            <p style={{ color: 'var(--muted)', fontSize: '.8rem', marginBottom: '1.2rem' }}>
              Comparison of defect detection capabilities across architectures on the PVEL-AD test dataset.
            </p>
            <div style={{ height: 350 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={F1_PER_CLASS_DATA} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="defect" stroke="var(--dim)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--dim)" unit="%" />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: 10 }} />
                  <Bar dataKey="DenseNet" fill="#3B82F6" name="DenseNet-121 (Base)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="DenseNet_cGAN" fill="#EC4899" name="DenseNet + cGAN ⭐" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ResNet" fill="#8B5CF6" name="ResNet-50" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="EfficientNet" fill="#10B981" name="EfficientNet-B0" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Radar Chart Multi-Attribute Comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.2rem' }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom: '.4rem' }}>
                🕸️ Multi-Dimensional Architecture Trade-Off Radar
              </div>
              <div style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={RADAR_METRICS}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: 'var(--dim)' }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="rgba(255,255,255,0.2)" />
                    <Radar name="DenseNet-121" dataKey="DenseNet" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.25} />
                    <Radar name="ResNet-50" dataKey="ResNet" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.2} />
                    <Radar name="EfficientNet-B0" dataKey="EfficientNet" stroke="#10B981" fill="#10B981" fillOpacity={0.2} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <div className="card-title" style={{ marginBottom: '.8rem' }}>
                💡 Key Architectural Insights
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.8rem', fontSize: '.84rem', lineHeight: 1.6 }}>
                <div style={{ padding: '.8rem', borderRadius: 8, background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)' }}>
                  <strong style={{ color: '#60A5FA' }}>1. DenseNet-121 Throughput Advantage:</strong>
                  <p style={{ margin: '.3rem 0 0 0', color: 'var(--muted)' }}>
                    At <strong>18.27 ms latency (54.7 FPS)</strong>, DenseNet-121 is <strong>2.3x faster</strong> than ResNet-50 (42.1 ms). In a high-speed solar panel manufacturing line scanning 200 panels/minute, DenseNet-121 meets real-time sorting timing budgets.
                  </p>
                </div>
                <div style={{ padding: '.8rem', borderRadius: 8, background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)' }}>
                  <strong style={{ color: '#A78BFA' }}>2. Parameter Efficiency via Feature Reuse:</strong>
                  <p style={{ margin: '.3rem 0 0 0', color: 'var(--muted)' }}>
                    DenseNet-121 achieves 87% accuracy with only <strong>6.97M parameters</strong> compared to ResNet-50's <strong>25.56M parameters</strong> (a 72.8% reduction), demonstrating high parameter efficiency via channel-wise feature concatenation.
                  </p>
                </div>
                <div style={{ padding: '.8rem', borderRadius: 8, background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
                  <strong style={{ color: '#4ADE80' }}>3. Complementary Multi-Model Deployment:</strong>
                  <p style={{ margin: '.3rem 0 0 0', color: 'var(--muted)' }}>
                    Factory pipelines can leverage <strong>DenseNet-121 for primary fast triage</strong>, escalating borderline or high-severity cases to <strong>ResNet-50 for deep diagnostic verification</strong>.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Architectural Deep Dive */}
      {activeTab === 'architecture' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          <div className="card">
            <div className="card-title" style={{ marginBottom: '.8rem' }}>
              🔬 DenseNet-121 Architecture: Phased Fine-Tuning & Feature Concatenation
            </div>
            <div style={{ fontSize: '.86rem', lineHeight: 1.7, color: 'var(--muted)' }}>
              <p>
                DenseNet-121 connects all layers directly within each dense block. In contrast to standard residual networks where layer <span style={{ color: 'var(--text)', fontFamily: 'monospace' }}>l</span> combines features through summation <span style={{ color: 'var(--text)', fontFamily: 'monospace' }}>x_l = H_l(x_l-1) + x_l-1</span>, DenseNet uses channel concatenation:
              </p>
              <div style={{
                background: 'rgba(0,0,0,0.3)', padding: '.9rem 1.2rem', borderRadius: 8,
                fontFamily: 'monospace', color: '#60A5FA', fontSize: '.9rem', margin: '.8rem 0'
              }}>
                x_l = H_l([x_0, x_1, x_2, ..., x_l-1])
              </div>
              <p>
                This ensures that low-level micro-crack edge filters extracted in early dense layers propagate directly to the classification head without gradient vanishing.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
              <div style={{ padding: '.9rem', borderRadius: 8, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '.4rem' }}>Phase 1: Classifier Training (Epochs 1-3)</div>
                <div style={{ fontSize: '.78rem', color: 'var(--dim)' }}>
                  Backbone weights frozen. Custom classifier head (<span style={{ fontFamily: 'monospace' }}>Dropout(0.3) + Linear(1024, 11)</span>) trained using AdamW at learning rate 1e-3.
                </div>
              </div>
              <div style={{ padding: '.9rem', borderRadius: 8, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '.4rem' }}>Phase 2: Block 4 Fine-Tuning (Epochs 4-13)</div>
                <div style={{ fontSize: '.78rem', color: 'var(--dim)' }}>
                  Dense Block 4 + norm5 unfrozen (2.15M parameters). Trained with differential learning rates (backbone 1e-4, head 1e-4) to refine high-level defect features without disturbing low-level ImageNet primitives.
                </div>
              </div>
              <div style={{ padding: '.9rem', borderRadius: 8, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '.4rem' }}>Sub-linear Loss Weighting</div>
                <div style={{ fontSize: '.78rem', color: 'var(--dim)' }}>
                  Loss weights scaled via <span style={{ fontFamily: 'monospace' }}>W_c = 1/√N_c</span> normalized to the majority class. Dampens extreme minority gradient surges while elevating defect penalties.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
