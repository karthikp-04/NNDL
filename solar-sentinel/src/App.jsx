import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Upload from './pages/Upload'
import Analyzing from './pages/Analyzing'
import Results from './pages/Results'
import HeatmapPage from './pages/HeatmapPage'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import About from './pages/About'
import ToastContainer from './components/ToastContainer'
import { AppContext } from './context/AppContext'
// SolarSentinel AI App entry point

// Beautiful scientific EL panel image represented as a lightweight SVG Data URL
export const DEMO_SOLAR_PANEL = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="%230F172A"/><rect x="10" y="10" width="780" height="480" fill="%231E293B" rx="10" stroke="%233B82F6" stroke-width="2" stroke-opacity="0.3"/><g stroke="%23334155" stroke-width="2" stroke-opacity="0.8"><line x1="140" y1="10" x2="140" y2="490"/><line x1="270" y1="10" x2="270" y2="490"/><line x1="400" y1="10" x2="400" y2="490"/><line x1="530" y1="10" x2="530" y2="490"/><line x1="660" y1="10" x2="660" y2="490"/><line x1="10" y1="130" x2="790" y2="130"/><line x1="10" y1="250" x2="790" y2="250"/><line x1="10" y1="370" x2="790" y2="370"/></g><g opacity="0.15"><rect x="15" y="15" width="115" height="105" fill="%233B82F6"/><rect x="145" y="15" width="115" height="105" fill="%233B82F6"/><rect x="275" y="15" width="115" height="105" fill="%233B82F6"/><rect x="405" y="15" width="115" height="105" fill="%233B82F6"/><rect x="15" y="135" width="115" height="105" fill="%233B82F6"/><rect x="145" y="135" width="115" height="105" fill="%233B82F6"/><rect x="275" y="135" width="115" height="105" fill="%233B82F6"/><rect x="405" y="135" width="115" height="105" fill="%233B82F6"/></g><rect x="540" y="15" width="110" height="105" fill="%23EF4444" fill-opacity="0.1" stroke="%23EF4444" stroke-width="1.5" stroke-dasharray="3"/><path d="M 550 30 Q 580 80 630 90 M 570 20 L 590 50" stroke="%23EF4444" stroke-width="2.5" fill="none" stroke-linecap="round"/><circle cx="590" cy="60" r="35" fill="none" stroke="%23EF4444" stroke-width="1" stroke-dasharray="4"/><g transform="translate(20, 470)" font-family="monospace" font-size="11" fill="%2364748B"><text x="0" y="0">EL_IMG_SCAN // ID: 88204-A</text><text x="250" y="0">RESOLUTION: 1024x1024</text><text x="500" y="0">MODE: MULTISPECTRAL_EL</text></g></svg>`

const INITIAL_RESULT = {
  prediction: ['Micro Crack', 'Finger Interruption'],
  severity: 'moderate',
  confidence: 94.7,
  recs: [
    { icon: '🔧', title: 'Schedule Maintenance', desc: 'Clean affected cell area within 30 days', primary: true },
    { icon: '📡', title: 'Increase Monitoring', desc: 'Set alerts to daily scan frequency' },
    { icon: '📋', title: 'Document Finding', desc: 'Log to maintenance record for warranty claim' },
  ]
}

const MOCK_HISTORY = [
  { id: 1, time: '10:31 AM', date: 'Jul 17', image: 'solar_panel_EL_001.jpg', prediction: 'Micro Crack', severity: 'moderate', confidence: 94.7 },
  { id: 2, time: '10:34 AM', date: 'Jul 17', image: 'solar_panel_EL_002.jpg', prediction: 'Healthy',    severity: 'healthy',  confidence: 98.1 },
  { id: 3, time: '10:48 AM', date: 'Jul 17', image: 'solar_panel_EL_003.jpg', prediction: 'Hotspot',    severity: 'severe',   confidence: 94.6 },
  { id: 4, time: '11:02 AM', date: 'Jul 17', image: 'solar_panel_EL_004.jpg', prediction: 'PID Effect', severity: 'mild',     confidence: 76.2 },
  { id: 5, time: '11:15 AM', date: 'Jul 17', image: 'solar_panel_EL_005.jpg', prediction: 'Healthy',    severity: 'healthy',  confidence: 99.0 },
  { id: 6, time: '11:29 AM', date: 'Jul 17', image: 'solar_panel_EL_006.jpg', prediction: 'Finger Interruption', severity: 'moderate', confidence: 81.5 },
  { id: 7, time: '11:43 AM', date: 'Jul 17', image: 'solar_panel_EL_007.jpg', prediction: 'Corrosion',  severity: 'severe',   confidence: 92.8 },
  { id: 8, time: '12:05 PM', date: 'Jul 17', image: 'solar_panel_EL_008.jpg', prediction: 'Healthy',    severity: 'healthy',  confidence: 97.4 },
  { id: 9, time: '12:18 PM', date: 'Jul 17', image: 'solar_panel_EL_009.jpg', prediction: 'Micro Crack',severity: 'mild',     confidence: 83.1 },
  { id: 10, time: '12:31 PM', date: 'Jul 17', image: 'solar_panel_EL_010.jpg', prediction: 'Delamination',severity:'severe',   confidence: 91.7 },
]

export default function App() {
  const [activePage, setActivePage]     = useState('upload')
  const [analysisState, setAnalysis]    = useState(null)      // { image, result }
  const [history, setHistory]           = useState([])        // starts empty — populated by real user uploads
  const [toasts, setToasts]             = useState([])

  const addToast = (msg, type = 'info') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }

  const navigate = (page) => setActivePage(page)

  const pageMap = {
    upload:    { title: 'Upload Image',    sub: 'Upload EL / RGB / Infrared solar panel images for analysis' },
    analyzing: { title: 'Analyzing...',    sub: 'Running CNN inference and Grad-CAM heatmap generation' },
    results:   { title: 'Analysis Results',sub: 'Defect detection results, severity rating & recommendations' },
    heatmap:   { title: 'Grad-CAM Heatmap',sub: 'Visual explanation of where the CNN focused' },
    dashboard: { title: 'Dashboard',       sub: 'Overview of all analyzed panels and defect statistics' },
    history:   { title: 'History',         sub: 'All previous prediction records' },
    about:     { title: 'About',           sub: 'SolarGuard AI — Dataset, model and team information' },
  }

  const hasResult = Boolean(analysisState?.result)
  const ctx = { activePage, navigate, analysisState, setAnalysis, history, setHistory, addToast, hasResult }

  return (
    <AppContext.Provider value={ctx}>
      <div className="app-shell">
        <Sidebar activePage={activePage} navigate={navigate} historyCount={history.length} hasResult={hasResult} />
        <div className="main-content">
          <Topbar page={pageMap[activePage] || pageMap.upload} />
          <div className="page-wrapper">
            {activePage === 'upload'    && <Upload />}
            {activePage === 'analyzing' && <Analyzing />}
            {activePage === 'results'   && <Results />}
            {activePage === 'heatmap'   && <HeatmapPage />}
            {activePage === 'dashboard' && <Dashboard />}
            {activePage === 'history'   && <History />}
            {activePage === 'about'     && <About />}
          </div>
        </div>
        <ToastContainer toasts={toasts} />
      </div>
    </AppContext.Provider>
  )
}

