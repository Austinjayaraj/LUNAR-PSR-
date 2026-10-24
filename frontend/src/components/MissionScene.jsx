/**
 * MissionScene — the contextual scene controller that renders the right
 * experience for each missionScene value.
 *
 * MISSION_SETUP       → mission brief + upload
 * ANALYSIS            → pipeline running (full-screen cinematic)
 * ANALYSIS_RESULTS    → results with layer switcher (full-screen)
 * TERRAIN+            → TerrainViewport (3D world)
 *
 * Existing components are REUSED:
 *   AnalysisViewport (upload / processing / results states)
 *   TerrainViewport  (3D world, lander, rover, EVA, science)
 *   ResearchDrawer   (diagnostic pull-out)
 *   RecordingController
 *   DescentSimulation
 */
import useStore from '../store';
import AnalysisViewport from './AnalysisViewport';
import TerrainViewport  from './TerrainViewport';
import ResearchDrawer   from './ResearchDrawer';

const TERRAIN_SCENES = new Set([
  'TERRAIN','LANDING','TOUCHDOWN','ROVER','EVA','SCIENCE'
]);
const ANALYSIS_SCENES = new Set([
  'MISSION_SETUP','ANALYSIS','ANALYSIS_RESULTS'
]);

export default function MissionScene() {
  const missionScene = useStore((s) => s.missionScene);

  if (!missionScene || missionScene === 'LANDING_PAGE' || missionScene === 'MOON_INTERACTION') {
    return null;
  }

  if (ANALYSIS_SCENES.has(missionScene)) {
    return (
      <div className="mission-scene mission-scene--analysis">
        {/* Full-screen analysis viewport */}
        <AnalysisViewport />
        {/* Research drawer — bottom of screen, slides up on demand */}
        <ResearchDrawer />
      </div>
    );
  }

  if (TERRAIN_SCENES.has(missionScene)) {
    return (
      <div className="mission-scene mission-scene--terrain">
        <TerrainViewport />
      </div>
    );
  }

  return null;
}
