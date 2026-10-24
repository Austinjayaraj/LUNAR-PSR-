/**
 * App — top-level scene router.
 *
 * Renders:
 *   SpaceScene     — full-screen cinematic Moon (LANDING_PAGE, MOON_INTERACTION)
 *   MissionScene   — contextual mission scenes (MISSION_SETUP → SCIENCE)
 *   MissionHUD     — minimal persistent overlay (hidden on landing page)
 *   DescentSimulation — headless simulation controller
 *   RecordingController — headless recording controller
 */
import { useEffect } from 'react';
import useStore from './store';
import SpaceScene          from './components/SpaceScene';
import MissionScene        from './components/MissionScene';
import MissionHUD          from './components/MissionHUD';
import DescentSimulation   from './components/DescentSimulation';
import RecordingController from './components/RecordingController';
import './App.css';

export default function App() {
  const checkBackend = useStore((s) => s.checkBackend);
  const addEvent     = useStore((s) => s.addEvent);

  useEffect(() => {
    addEvent('Lunar PSR Terminal initialized', 'info');
    checkBackend();
    const id = setInterval(checkBackend, 8000);
    return () => clearInterval(id);
  }, [checkBackend, addEvent]);

  return (
    <div className="app">
      {/* Scene 0–1: Cinematic Moon */}
      <SpaceScene />

      {/* Scene 2–10: Mission flow */}
      <MissionScene />

      {/* Persistent minimal HUD (hidden on landing page) */}
      <MissionHUD />

      {/* Headless controllers */}
      <DescentSimulation />
      <RecordingController />
    </div>
  );
}
