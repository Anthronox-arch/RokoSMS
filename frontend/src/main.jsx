import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  ClipboardPaste,
  ExternalLink,
  Link2,
  Loader2,
  MessageSquareText,
  RotateCcw,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  Trash2,
  Wifi,
  WifiOff,
  XCircle,
} from 'lucide-react'
import './styles.css'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const MAX_LENGTH = 5000

function formatKey(key) {
  return String(key)
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function isUsefulUrlCheck(value) {
  if (value === null || value === undefined) return false
  if (typeof value === 'string') return value.trim().length > 0
  if (typeof value === 'number' || typeof value === 'boolean') return true
  if (Array.isArray(value)) return value.length > 0
  if (typeof value === 'object') return Object.keys(value).length > 0
  return false
}

function Value({ value }) {
  if (typeof value === 'boolean') {
    return <span className={value ? 'value-bool true' : 'value-bool'}>{value ? 'Yes' : 'No'}</span>
  }
  if (Array.isArray(value)) {
    return (
      <div className="value-list">
        {value.map((item, i) => <span key={i}>{typeof item === 'object' ? JSON.stringify(item) : String(item)}</span>)}
      </div>
    )
  }
  if (value && typeof value === 'object') {
    return <pre className="json-value">{JSON.stringify(value, null, 2)}</pre>
  }
  return <span>{String(value)}</span>
}

function UrlAnalysis({ data }) {
  if (!isUsefulUrlCheck(data)) return null

  if (typeof data === 'object' && !Array.isArray(data)) {
    return (
      <section className="url-analysis" aria-labelledby="url-heading">
        <div className="section-heading">
          <div className="section-icon"><Link2 size={17} strokeWidth={1.8} /></div>
          <div>
            <p className="eyebrow">Link inspection</p>
            <h2 id="url-heading">URL analysis</h2>
          </div>
        </div>
        <div className="url-grid">
          {Object.entries(data).map(([key, value]) => (
            <div className="url-row" key={key}>
              <span>{formatKey(key)}</span>
              <Value value={value} />
            </div>
          ))}
        </div>
      </section>
    )
  }

  return (
    <section className="url-analysis" aria-labelledby="url-heading">
      <div className="section-heading">
        <div className="section-icon"><Link2 size={17} strokeWidth={1.8} /></div>
        <div><p className="eyebrow">Link inspection</p><h2 id="url-heading">URL analysis</h2></div>
      </div>
      <div className="url-single"><Value value={data} /></div>
    </section>
  )
}

function SignalList({ title, items, tone, icon }) {
  if (!items?.length) return null
  return (
    <section className={`signals ${tone}`}>
      <div className="section-heading compact">
        <div className="section-icon">{icon}</div>
        <div><p className="eyebrow">Model signal</p><h2>{title}</h2></div>
      </div>
      <ul>
        {items.map((item, i) => <li key={`${item}-${i}`}><span className="signal-dot" />{item}</li>)}
      </ul>
    </section>
  )
}

function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('checking')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    if (!API_URL) {
      setStatus('missing')
      return undefined
    }
    fetch(`${API_URL}/`, { signal: AbortSignal.timeout(8000) })
      .then((res) => {
        if (!res.ok) throw new Error('API unavailable')
        return res.json()
      })
      .then(() => { if (!cancelled) setStatus('online') })
      .catch(() => { if (!cancelled) setStatus('offline') })
    return () => { cancelled = true }
  }, [])

  const verdict = result?.label || 'idle'
  const probability = result ? Math.round(Number(result.probability) * 100) : null
  const charCount = message.length

  const stateClass = useMemo(() => {
    if (verdict === 'scam') return 'state-scam'
    if (verdict === 'legit') return 'state-legit'
    return 'state-idle'
  }, [verdict])

  const checkMessage = useCallback(async () => {
    const text = message.trim()
    if (!text || loading) return
    if (!API_URL) {
      setError('VITE_API_URL is not configured. Add it to your environment and restart the frontend.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: AbortSignal.timeout(20000),
      })

      let payload = null
      try { payload = await response.json() } catch { /* handled below */ }
      if (!response.ok) {
        const detail = payload?.detail || `Request failed with status ${response.status}.`
        throw new Error(detail)
      }
      if (!payload || !['scam', 'legit'].includes(payload.label)) {
        throw new Error('The API returned an unexpected result.')
      }
      setResult(payload)
    } catch (err) {
      if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
        setError('The request took too long. Check that the API is running and try again.')
      } else if (err instanceof TypeError) {
        setError('The API could not be reached. Check VITE_API_URL and the backend CORS settings.')
      } else {
        setError(err?.message || 'The message could not be checked.')
      }
    } finally {
      setLoading(false)
    }
  }, [loading, message])

  const clearAll = () => {
    setMessage('')
    setResult(null)
    setError('')
    requestAnimationFrame(() => textareaRef.current?.focus())
  }

  const pasteMessage = async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text) setMessage(text.slice(0, MAX_LENGTH))
    } catch {
      setError('Clipboard access was blocked. Paste directly into the message field.')
    }
  }

  const onKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault()
      checkMessage()
    }
  }

  return (
    <main className={`app ${stateClass}`}>
      <div className="ambient-glow" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#top" aria-label="RokoSMS home">
          <span className="brand-mark"><ShieldCheck size={20} strokeWidth={1.9} /></span>
          <span>Roko<span>SMS</span></span>
        </a>
        <div className="api-status" aria-live="polite">
          <span className={`status-dot ${status}`} />
          <span>{status === 'online' ? 'API online' : status === 'checking' ? 'Checking API' : status === 'missing' ? 'API URL missing' : 'API offline'}</span>
        </div>
      </header>

      <div className="workspace" id="top">
        <section className="intro">
          <div className="intro-kicker"><span /> ROMAN URDU MESSAGE CHECK</div>
          <h1>Pause before you<br /><em>trust the message.</em></h1>
          <p>Paste an SMS or WhatsApp message. RokoSMS checks its language patterns and returns the model's classification and supporting signals.</p>
        </section>

        <section className="analysis-shell" aria-label="Message analysis">
          <div className="composer-head">
            <div>
              <span className="field-label">Message</span>
              <span className="field-hint">SMS / WhatsApp</span>
            </div>
            <button className="text-action" type="button" onClick={pasteMessage} disabled={loading}>
              <ClipboardPaste size={15} /> Paste
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value.slice(0, MAX_LENGTH))}
            onKeyDown={onKeyDown}
            placeholder="e.g. Mubarak ho! Aap ne 50,000 rupees ka inaam jeeta hai..."
            aria-label="SMS or WhatsApp message"
            maxLength={MAX_LENGTH}
            disabled={loading}
          />

          <div className="composer-foot">
            <span className="counter">{charCount.toLocaleString()} / {MAX_LENGTH}</span>
            <div className="composer-actions">
              {(message || result) && (
                <button className="icon-action" type="button" onClick={clearAll} disabled={loading} aria-label="Clear message and result" title="Clear">
                  <Trash2 size={17} />
                </button>
              )}
              <button className="primary-action" type="button" onClick={checkMessage} disabled={!message.trim() || loading}>
                {loading ? <><Loader2 className="spin" size={17} /> Checking</> : <><ShieldCheck size={17} /> Check message</>}
              </button>
            </div>
          </div>
          <p className="shortcut"><kbd>Ctrl</kbd><span>+</span><kbd>Enter</kbd> to check</p>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <XCircle size={18} />
            <div><strong>Check failed</strong><span>{error}</span></div>
            <button type="button" onClick={() => setError('')} aria-label="Dismiss error"><XCircle size={16} /></button>
          </div>
        )}

        {loading && (
          <section className="result-panel loading-panel" aria-live="polite" aria-busy="true">
            <div className="loading-orbit"><Loader2 size={30} className="spin" /></div>
            <div><p className="eyebrow">Analyzing message</p><h2>Reading the signals…</h2><p>Checking the text against the RokoSMS model.</p></div>
          </section>
        )}

        {result && !loading && (
          <section className={`result-panel result-${result.label}`} aria-live="polite">
            <div className="verdict-line">
              <div className="verdict-icon">
                {result.label === 'scam' ? <ShieldAlert size={30} strokeWidth={1.8} /> : <ShieldCheck size={30} strokeWidth={1.8} />}
              </div>
              <div className="verdict-copy">
                <span className="eyebrow">RokoSMS verdict</span>
                <h2>{result.label === 'scam' ? 'Potential scam' : 'Likely legitimate'}</h2>
                <p>{result.label === 'scam' ? 'Treat this message with caution. Review the signals below before taking action.' : 'The model found stronger legitimate signals than scam signals.'}</p>
              </div>
              <div className="probability" aria-label={`Probability ${probability} percent`}>
                <strong>{probability}%</strong>
                <span>model probability</span>
              </div>
            </div>

            <div className="result-divider" />

            <div className="signal-grid">
              <SignalList title="Why it may be a scam" items={result.toward_scam} tone="danger" icon={<AlertTriangle size={17} />} />
              <SignalList title="Why it may be legitimate" items={result.toward_legit} tone="success" icon={<CheckCircle2 size={17} />} />
            </div>

            <UrlAnalysis data={result.url_check} />

            <div className="result-footer">
              <span><CircleHelp size={14} /> Model output is a signal, not proof. Verify unexpected requests independently.</span>
              <button type="button" className="run-again" onClick={() => { setResult(null); setError(''); textareaRef.current?.focus() }}>
                <RotateCcw size={14} /> Check another
              </button>
            </div>
          </section>
        )}

        {!result && !loading && !error && (
          <section className="empty-state" aria-label="No analysis yet">
            <div className="empty-icon"><MessageSquareText size={20} /></div>
            <div><strong>Your result will appear here</strong><span>Paste a message above to inspect it.</span></div>
          </section>
        )}

        <footer className="footer">
          <span><ShieldCheck size={14} /> RokoSMS</span>
          <span>Pakistan-focused Roman Urdu scam detection</span>
          <span className="footer-note"><Sparkles size={13} /> Built for cautious reading</span>
        </footer>
      </div>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
