import { CheckCircle2, AlertCircle, Info } from 'lucide-react'

export default function ToastContainer({ toasts }) {
  return (
    <div className="toast-container" role="status" aria-live="polite">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`}>
          <span style={{ display: 'inline-flex', alignItems: 'center' }}>
            {t.type === 'success' && <CheckCircle2 size={16} color="var(--status-pass)" />}
            {t.type === 'error' && <AlertCircle size={16} color="var(--status-fail)" />}
            {t.type !== 'success' && t.type !== 'error' && <Info size={16} color="var(--accent)" />}
          </span>
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  )
}
