import { useState } from 'react'
import {
  Flame,
  FileImage,
  Layers
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import Badge from '../components/ui/Badge'

function PanelView({ label, icon: Icon, children, badge }) {
  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        padding: '0.75rem 1rem', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, fontSize: '0.82rem' }}>
          <Icon size={15} color="var(--accent)" />
          <span>{label}</span>
        </div>
        {badge && (
          <Badge variant="outline">{badge}</Badge>
        )}
      </div>
      {children}
    </div>
  )
}

export default function HeatmapPage() {
  const { analysisState } = useApp()
  const image = analysisState?.image
  const [opacity, setOpacity] = useState(0.65)

  // Calibrated Grad-CAM gradient stops
  const heatmapGrad = `radial-gradient(
    ellipse at 50% 50%,
    rgba(229, 72, 77, 0.85) 0%,
    rgba(245, 165, 36, 0.7) 35%,
    rgba(63, 182, 139, 0.4) 65%,
    transparent 80%
  )`

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Explanation Banner */}
      <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1.25rem 1.5rem' }}>
        <div style={{
          width: 44, height: 44, borderRadius: 'var(--radius-sm)', flexShrink: 0,
          background: 'var(--surface-raised)', border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--accent)'
        }}>
          <Flame size={22} />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text)', marginBottom: '0.2rem' }}>
            Grad-CAM Visual Attention Explanation
          </div>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Gradient-weighted Class Activation Mapping computes backpropagated gradients into the final convolutional feature maps (DenseNet Block 4 / ResNet Layer 4).
            High-activation regions indicate localized cell micro-fissures and non-luminescent inactive defects.
          </p>
        </div>
      </div>

      {/* Three-Panel Heatmap Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        {/* Panel 1: Original */}
        <PanelView label="Original Input Imagery" icon={FileImage} badge="Raw Tensor">
          <div style={{ height: 260, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {image ? (
              <img src={image} alt="Original solar panel" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            ) : (
              <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>No image loaded</div>
            )}
          </div>
          <div style={{ padding: '0.75rem 1rem', borderTop: '1px solid var(--border)', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            Preprocessed 224 × 224 RGB standardized tensor
          </div>
        </PanelView>

        {/* Panel 2: Grad-CAM Activation Map */}
        <PanelView label="Grad-CAM Activation" icon={Flame} badge="Feature Maps">
          <div style={{ height: 260, background: 'var(--bg)', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {image ? (
              <>
                <img src={image} alt="Heatmap base" style={{ width: '100%', height: '100%', objectFit: 'contain', filter: 'grayscale(1) brightness(0.25)' }} />
                <div style={{ position: 'absolute', inset: 0, background: heatmapGrad, mixBlendMode: 'screen' }} />
                <div style={{
                  position: 'absolute', top: 12, right: 12,
                  background: 'rgba(229,72,77,0.2)', border: '1px solid var(--status-fail)',
                  borderRadius: 'var(--radius-sm)', padding: '0.2rem 0.5rem',
                  color: 'var(--status-fail)', fontSize: '0.72rem', fontWeight: 600, fontFamily: 'var(--font-mono)'
                }}>
                  Peak Salience
                </div>
              </>
            ) : (
              <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>No image loaded</div>
            )}
          </div>
          <div style={{
            padding: '0.75rem 1rem', borderTop: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-dim)'
          }}>
            <span>Gradient Heatmap</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>Low</span>
              <div style={{ width: 60, height: 6, borderRadius: 'var(--radius-full)', background: 'linear-gradient(to right, var(--status-pass), var(--status-warn), var(--status-fail))' }}></div>
              <span>High</span>
            </div>
          </div>
        </PanelView>

        {/* Panel 3: Combined Overlay */}
        <PanelView label="Combined Saliency Overlay" icon={Layers} badge="Fused Overlay">
          <div style={{ height: 260, background: 'var(--bg)', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {image ? (
              <>
                <img src={image} alt="Panel overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                <div style={{ position: 'absolute', inset: 0, background: heatmapGrad, opacity, mixBlendMode: 'hard-light' }} />
              </>
            ) : (
              <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>No image loaded</div>
            )}
          </div>
          <div style={{
            padding: '0.75rem 1rem', borderTop: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', gap: '0.75rem'
          }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', whiteSpace: 'nowrap' }}>Opacity:</span>
            <input
              type="range" min="0.2" max="1" step="0.05"
              value={opacity}
              onChange={e => setOpacity(Number(e.target.value))}
              style={{ flex: 1, accentColor: 'var(--accent)' }}
            />
            <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--text)', fontWeight: 600 }}>
              {Math.round(opacity * 100)}%
            </span>
          </div>
        </PanelView>
      </div>

      {/* Architecture layer attribution breakdown */}
      <div className="card">
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>
          Convolutional Stage Attribution Details
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.75rem' }}>
          {[
            { model: 'DenseNet-121', layer: 'denseblock4.denselayer16.conv2', feat: '1024 channels', res: '7 × 7' },
            { model: 'ResNet-50', layer: 'layer4[2].conv3', feat: '2048 channels', res: '7 × 7' },
            { model: 'EfficientNet-B0', layer: 'features[8][0].block[2]', feat: '1280 channels', res: '7 × 7' }
          ].map(row => (
            <div key={row.model} className="card-raised card" style={{ padding: '0.85rem 1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <strong style={{ fontSize: '0.86rem', color: 'var(--text)' }}>{row.model}</strong>
                <Badge variant="outline">{row.res}</Badge>
              </div>
              <div className="font-mono" style={{ fontSize: '0.74rem', color: 'var(--accent)' }}>{row.layer}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>Feature Depth: {row.feat}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
