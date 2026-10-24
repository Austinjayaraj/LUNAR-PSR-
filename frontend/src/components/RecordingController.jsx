/**
 * RecordingController — captures the Three.js canvas via MediaRecorder.
 * The recording is the actual rendered mission output.
 *
 * Lifecycle:
 *   isRecording = true  → find canvas, start stream + MediaRecorder
 *   isRecording = false → stop recorder, stop all stream tracks (no MediaStream leak)
 *   On unmount          → same cleanup
 */
import { useRef, useEffect } from 'react';
import useStore from '../store';

export default function RecordingController() {
  const isRecording     = useStore((s) => s.isRecording);
  const setRecording    = useStore((s) => s.setRecording);
  const setRecordingUrl = useStore((s) => s.setRecordingUrl);
  const addEvent        = useStore((s) => s.addEvent);

  const mrRef     = useRef(null);  // MediaRecorder
  const streamRef = useRef(null);  // MediaStream — must be stopped to release resources
  const chunksRef = useRef([]);

  const stopAll = () => {
    if (mrRef.current && mrRef.current.state !== 'inactive') {
      try {
        mrRef.current.stop();
      } catch (_) {}
    }
    mrRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    if (isRecording) {
      const canvas = document.querySelector('.terrain-canvas canvas')
        ?? document.querySelector('canvas');

      if (!canvas) {
        addEvent('Recording: no canvas found', 'error');
        setRecording(false);
        return;
      }

      if (typeof canvas.captureStream !== 'function') {
        addEvent('Recording: captureStream not supported in this browser', 'warn');
        setRecording(false);
        return;
      }

      const mimeType = ['video/webm;codecs=vp9', 'video/webm'].find(
        (m) => MediaRecorder.isTypeSupported(m)
      );
      if (!mimeType) {
        addEvent('Recording: no supported video codec', 'warn');
        setRecording(false);
        return;
      }

      chunksRef.current = [];
      try {
        const stream = canvas.captureStream(30);
        streamRef.current = stream;
        const mr = new MediaRecorder(stream, { mimeType });
        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
        };
        mr.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          const url = URL.createObjectURL(blob);
          setRecordingUrl(url);
          addEvent('Recording completed and ready for export', 'info');
        };
        mr.start(250);
        mrRef.current = mr;
        addEvent('Mission recording started (30 FPS)', 'info');
      } catch (err) {
        addEvent(`Recording initialization error: ${err.message}`, 'error');
        setRecording(false);
      }
    } else {
      stopAll();
    }
    return () => stopAll();
  }, [isRecording, addEvent, setRecording, setRecordingUrl]);

  return null;
}
