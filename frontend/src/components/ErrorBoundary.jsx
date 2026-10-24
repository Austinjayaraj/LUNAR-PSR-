import React from 'react';

/**
 * ErrorBoundary — Catches uncaught runtime exceptions in child components
 * and prevents the entire React tree from unmounting to a blank black screen.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[LUNAR ERROR BOUNDARY] Uncaught error caught:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onRetry) {
      this.props.onRetry();
    } else {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#020306',
            color: '#f8fafc',
            fontFamily: "'Space Grotesk', -apple-system, sans-serif",
            padding: 32,
            zIndex: 99999,
            textAlign: 'center',
          }}
        >
          <div
            style={{
              padding: '24px 32px',
              maxWidth: 580,
              border: '1px solid rgba(239, 68, 68, 0.4)',
              borderRadius: 6,
              background: 'rgba(15, 23, 42, 0.85)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.8)',
            }}
          >
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                letterSpacing: '0.2em',
                color: '#ef4444',
                marginBottom: 12,
                textTransform: 'uppercase',
              }}
            >
              System Exception Caught
            </div>
            <h2
              style={{
                fontSize: 20,
                fontWeight: 600,
                letterSpacing: '0.08em',
                marginBottom: 12,
                color: '#f1f5f9',
              }}
            >
              LUNAR VISUALIZATION INITIALIZATION FAILED
            </h2>
            <p
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                color: '#94a3b8',
                lineHeight: 1.6,
                marginBottom: 20,
                wordBreak: 'break-word',
              }}
            >
              {this.state.error?.message || 'An unexpected runtime error occurred.'}
            </p>
            <button
              onClick={this.handleRetry}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                letterSpacing: '0.15em',
                padding: '10px 24px',
                background: '#38bdf8',
                color: '#020617',
                border: 'none',
                borderRadius: 3,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'opacity 0.2s',
              }}
            >
              RETRY INITIALIZATION
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
