import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Upload from './pages/Upload'
import Analyzing from './pages/Analyzing'
import Results from './pages/Results'
import HeatmapPage from './pages/HeatmapPage'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import About from './pages/About'
import ComparePage from './pages/ComparePage'
import GanStudio from './pages/GanStudio'
import ToastContainer from './components/ToastContainer'
import { AppContext } from './context/AppContext'

export default function App() {
  const [activePage, setActivePage]       = useState('upload')
  const [selectedModel, setSelectedModel] = useState('densenet121_gan') // 'densenet121_gan' | 'densenet121' | 'resnet50' | 'efficientnet_b0' | 'compare'
  const [analysisState, setAnalysis]      = useState(null)          // { image, result }
  const [history, setHistory]             = useState([])            // populated by predictions
  const [toasts, setToasts]               = useState([])

  const addToast = (msg, type = 'info') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }

  const navigate = (page) => setActivePage(page)

  const pageMap = {
    upload:     { title: 'Upload & Inference',         sub: 'Deep Learning defect detection on EL / RGB / Infrared solar panel images' },
    analyzing:  { title: 'Running Model...',           sub: 'Executing neural network forward pass and latency profiling' },
    results:    { title: 'Analysis & Evaluation',      sub: 'Diagnostic results, confidence rating, severity & model metrics' },
    compare:    { title: 'Model Benchmarks & Compare', sub: 'Comparative Evaluation — DenseNet-121 (Baseline & cGAN) vs ResNet-50 vs EfficientNet-B0' },
    'gan-studio': { title: 'cGAN Synthetic Defect Studio', sub: 'Generative Adversarial Network Data Augmentation & Minority Class Balancing' },
    heatmap:    { title: 'Grad-CAM Attention Map',     sub: 'Visual explanation of neural network feature activations' },
    dashboard:  { title: 'Fleet Monitoring',           sub: 'Overview of all analyzed panels and defect statistics' },
    history:    { title: 'Inspection History',         sub: 'All previous prediction and inference records' },
    about:      { title: 'About SolarGuard AI',        sub: 'PVEL-AD Dataset, Deep Learning Architectures & Methodology' },
  }

  const hasResult = Boolean(analysisState?.result)
  const ctx = {
    activePage,
    navigate,
    selectedModel,
    setSelectedModel,
    analysisState,
    setAnalysis,
    history,
    setHistory,
    addToast,
    hasResult
  }

  return (
    <AppContext.Provider value={ctx}>
      <div className="app-shell">
        <Sidebar activePage={activePage} navigate={navigate} historyCount={history.length} hasResult={hasResult} />
        <div className="main-content">
          <Topbar page={pageMap[activePage] || pageMap.upload} />
          <div className="page-wrapper">
            {activePage === 'upload'     && <Upload />}
            {activePage === 'analyzing'  && <Analyzing />}
            {activePage === 'results'    && <Results />}
            {activePage === 'compare'    && <ComparePage />}
            {activePage === 'gan-studio' && <GanStudio />}
            {activePage === 'heatmap'    && <HeatmapPage />}
            {activePage === 'dashboard'  && <Dashboard />}
            {activePage === 'history'    && <History />}
            {activePage === 'about'      && <About />}
          </div>
        </div>
        <ToastContainer toasts={toasts} />
      </div>
    </AppContext.Provider>
  )
}
