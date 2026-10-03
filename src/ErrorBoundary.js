import React from 'react';

/**
 * Last line of defence for the whole page.
 *
 * React unmounts the entire tree when a component throws while rendering, and
 * a Create React App *production* build has no overlay to explain it - the
 * user is left with a blank white page and nothing to go on. This boundary
 * keeps that failure legible instead: what React complained about, and a way
 * to reload.
 *
 * It exists because of a real one: an API error body shaped
 * `{"error": {"code": "500", "message": "..."}}` was rendered as a child, so
 * a backend that was merely down blanked the whole site. See `messageFrom` in
 * `src/api.js` for the fix and README.md for the story.
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // The browser console is where a developer looks first, so put it there
    // in full: the boundary itself only renders the message.
    // eslint-disable-next-line no-console
    console.error('The page failed to render:', error, info && info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) {
      // eslint-disable-next-line react/prop-types
      return this.props.children;
    }
    // `error` can be anything a `throw` produced, hence the String(...).
    const detail = error && error.message ? String(error.message) : String(error);
    return (
      <div className="container py-5">
        <div className="alert alert-danger" role="alert">
          <h1 className="h5">The page failed to render.</h1>
          <p className="mb-2">
            Reload to try again - and open the browser console for the full trace. React
            reported:
          </p>
          <pre className="small mb-0">{detail}</pre>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
