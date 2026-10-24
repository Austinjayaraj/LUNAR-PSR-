import { useEffect, useRef } from 'react';
import useStore from '../store';

export default function DescentSimulation() {
  const descentMode = useStore((state) => state.descentMode);
  const updateDescentTelemetry = useStore((state) => state.updateDescentTelemetry);
  const setDescentPhase = useStore((state) => state.setDescentPhase);
  const confirmTouchdown = useStore((state) => state.confirmTouchdown);
  const stopDescent = useStore((state) => state.stopDescent);
  const addEvent = useStore((state) => state.addEvent);

  const animationRef = useRef(null);
  const startTimeRef = useRef(null);
  const stateRef = useRef({ altitude: 500, vVel: -10, hVel: 0.5, targetError: 15, time: 0 });

  useEffect(() => {
    if (!descentMode) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      startTimeRef.current = null;
      stateRef.current = { altitude: 500, vVel: -10, hVel: 0.5, targetError: 15, time: 0 };
      return;
    }

    startTimeRef.current = Date.now();
    stateRef.current = { altitude: 500, vVel: -10, hVel: 0.5, targetError: 15, time: 0 };

    const animate = () => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      const s = stateRef.current;

      const dt = 1 / 60;
      s.altitude += s.vVel * dt;
      s.targetError = Math.max(0, s.targetError - 0.05 * dt);
      s.time = elapsed;

      if (s.altitude > 200) {
        setDescentPhase('APPROACH');
        s.vVel = -10;
        s.hVel = 0.5;
      } else if (s.altitude > 100) {
        setDescentPhase('ALIGN');
        s.vVel = -5;
        s.hVel = 0.3;
      } else if (s.altitude > 20) {
        setDescentPhase('DESCENT');
        s.vVel = -3;
        s.hVel = 0.1;
      } else if (s.altitude > 1) {
        setDescentPhase('FINAL_APPROACH');
        s.vVel = -1;
        s.hVel = 0.02;
      } else {
        s.altitude = 0;
        s.vVel = 0;
        s.hVel = 0;
        updateDescentTelemetry({ ...s });
        confirmTouchdown();
        addEvent(`Touchdown. Landing error: ${s.targetError.toFixed(2)} m (SIMULATED)`, 'success');
        stopDescent();
        return;
      }

      s.altitude = Math.max(0, s.altitude);

      updateDescentTelemetry({
        altitude: s.altitude,
        vVel: s.vVel,
        hVel: s.hVel,
        targetError: s.targetError,
        time: s.time,
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [descentMode]);

  return null;
}
