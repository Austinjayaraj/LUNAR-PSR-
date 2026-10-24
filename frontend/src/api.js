const API_BASE = 'http://127.0.0.1:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/`, { method: 'GET' });
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return await res.json();
  } catch (err) {
    throw new Error(`Backend unreachable: ${err.message}`);
  }
}

export async function getSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/system-status`, { method: 'GET' });
    if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
    return await res.json();
  } catch (err) {
    throw new Error(`Status check failed: ${err.message}`);
  }
}

export async function runPipeline(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/pipeline`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Pipeline failed' }));
    throw new Error(errorData.detail || `Pipeline failed: ${res.status}`);
  }

  return await res.json();
}

export async function enhanceImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/enhance`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Enhancement failed: ${res.status}`);
  }

  return await res.blob();
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Analysis failed: ${res.status}`);
  }

  return await res.json();
}
