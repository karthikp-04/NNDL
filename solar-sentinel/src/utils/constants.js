import { Layers, Sparkles, Shield, Cpu, Scale, AlertTriangle, Zap, CheckCircle2, Activity, Eye } from 'lucide-react'

// Beautiful scientific EL panel image represented as a lightweight SVG Data URL
export const DEMO_SOLAR_PANEL = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%233B82F6" stroke-width="2" stroke-opacity="0.3"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><g opacity="0.15"><rect x="15" y="15" width="115" height="105" fill="%233B82F6"/><rect x="145" y="15" width="115" height="105" fill="%233B82F6"/><rect x="275" y="15" width="115" height="105" fill="%233B82F6"/><rect x="405" y="15" width="115" height="105" fill="%233B82F6"/><rect x="15" y="135" width="115" height="105" fill="%233B82F6"/><rect x="145" y="135" width="115" height="105" fill="%233B82F6"/><rect x="275" y="135" width="115" height="105" fill="%233B82F6"/><rect x="405" y="135" width="115" height="105" fill="%233B82F6"/></g><rect x="540" y="15" width="110" height="105" fill="%23EF4444" fill-opacity="0.1" stroke="%23EF4444" stroke-width="1.5" stroke-dasharray="3"/><path d="M 550 30 Q 580 80 630 90 M 570 20 L 590 50" stroke="%23EF4444" stroke-width="2.5" fill="none" stroke-linecap="round"/><circle cx="590" cy="60" r="35" fill="none" stroke="%23EF4444" stroke-width="1" stroke-dasharray="4"/><g transform="translate(20, 470)" font-family="monospace" font-size="11" fill="%2364748B"><text x="0" y="0">EL_IMG_SCAN // ID: 88204-A</text><text x="250" y="0">RESOLUTION: 1024x1024</text><text x="500" y="0">MODE: MULTISPECTRAL_EL</text></g></svg>`

export const MODEL_CONFIGS = [
  {
    id: 'densenet121_gan',
    name: 'DenseNet-121 (cGAN)',
    badge: 'cGAN BALANCED',
    desc: 'Trained on real + synthetic minority defect manifold',
    params: '6.97M',
    latency: '18.35 ms',
    metricLabel: 'Bal Acc',
    metricValue: '48.47%',
    icon: Sparkles
  },
  {
    id: 'densenet121',
    name: 'DenseNet-121',
    badge: 'BASELINE AOI',
    desc: 'High-throughput optical inspection sorting pipeline',
    params: '6.97M',
    latency: '18.27 ms',
    metricLabel: 'Accuracy',
    metricValue: '86.84%',
    icon: Layers
  },
  {
    id: 'resnet50',
    name: 'ResNet-50',
    badge: 'DIAGNOSTIC',
    desc: 'Residual bottleneck layers with Focal Loss recovery',
    params: '25.56M',
    latency: '42.10 ms',
    metricLabel: 'Accuracy',
    metricValue: '91.71%',
    icon: Shield
  },
  {
    id: 'efficientnet_b0',
    name: 'EfficientNet-B0',
    badge: 'EDGE DRONE',
    desc: 'Mobile inverted bottlenecks with channel SE attention',
    params: '4.02M',
    latency: '14.80 ms',
    metricLabel: 'Accuracy',
    metricValue: '86.40%',
    icon: Cpu
  },
  {
    id: 'compare',
    name: 'Tri-Model Compare',
    badge: 'EVALUATION',
    desc: 'Simultaneous cross-network consensus and speedup benchmarking',
    params: '36.55M',
    latency: '75.17 ms',
    metricLabel: 'Coverage',
    metricValue: '3 Models',
    icon: Scale
  }
]

export const PRESET_SAMPLES = [
  {
    id: 'crack',
    name: 'Micro Crack Defect',
    icon: AlertTriangle,
    file: 'cell_micro_crack_EL_021.jpg',
    tag: 'Crack'
  },
  {
    id: 'black_core',
    name: 'Black Core Inactive Area',
    icon: Zap,
    file: 'black_core_inactive_045.jpg',
    tag: 'Inactive'
  },
  {
    id: 'finger',
    name: 'Finger Interruption',
    icon: Activity,
    file: 'finger_interruption_018.jpg',
    tag: 'Grid Line'
  },
  {
    id: 'dislocation',
    name: 'Crystal Dislocation',
    icon: Eye,
    file: 'dislocation_crystal_0210.jpg',
    tag: 'Ingot Boundary'
  },
  {
    id: 'healthy',
    name: 'Normal Healthy Panel',
    icon: CheckCircle2,
    file: 'healthy_normal_panel_025.jpg',
    tag: 'Healthy'
  },
  {
    id: 'gan_sample',
    name: 'cGAN Synthetic Defect',
    icon: Sparkles,
    file: 'synthetic_cgan_star_crack_0001.png',
    tag: 'Synthetic'
  }
]
