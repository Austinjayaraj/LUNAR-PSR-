/**
 * ResearchDrawer — secondary scientific/diagnostic data panel.
 *
 * Hidden by default. Only opened when the user requests it.
 * All values sourced from backend pipelineTelemetry and existing store state.
 * Nothing is hard-coded.
 */
import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import useStore from '../store';
import { HAZARD_MODE_METRICS } from '../types';

export default function ResearchDrawer() {
  const [open, setOpen] = useState(false);
  const pipelineTelemetry = useStore((s) => s.pipelineTelemetry);
  const hazardMode        = useStore((s) => s.hazardMode);
  const missionStatus     = useStore((s) => s.missionStatus);
  const processingTimeline = useStore((s) => s.processingTimeline);
  const scientificLimitations = useStore((s) => s.scientificLimitations);
  const imageStats        = useStore((s) => s.imageStats);

  // Only show when there's data
  const hasData = missionStatus === 'COMPLETE' || missionStatus === 'ANALYSIS_RESULTS';
  if (!hasData && !pipelineTelemetry) return null;

  const restMeta  = pipelineTelemetry?.restoration;
  const hazardTel = pipelineTelemetry?.hazard;
  const modeM     = HAZARD_MODE_METRICS[hazardMode];
  const testM     = hazardTel?.validated_test_metrics;

  return (
    <div className="research-drawer">
      <button
        className={`research-toggle${open ? ' open' : ''}`}
        onClick={() => setOpen((v) => !v)}
      >
        <span>RESEARCH DATA</span>
        {open ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
      </button>

      {open && (
        <div className="research-content">

          {/* Restoration */}
          {restMeta && (
            <section className="rd-section">
              <div className="rd-section-title">RESTORATION MODEL</div>
              <Row k="ARCHITECTURE" v="LunarUNet v2" />
              <Row k="CHECKPOINT"   v={restMeta.checkpoint?.split('/').pop()} />
              <Row k="EPOCH"        v={restMeta.epoch} />
              <Row k="VAL LOSS"     v={restMeta.val_loss} />
              <Row k="PARAMETERS"   v={restMeta.parameter_count?.toLocaleString()} />
              <Row k="TRAINING DATA" v="Chandrayaan-2 OHRC 0.25m imagery" />
            </section>
          )}

          {/* Hazard engine */}
          {hazardTel && (
            <section className="rd-section">
              <div className="rd-section-title">HAZARD ENGINE</div>
              <Row k="ENGINE"   v={hazardTel.engine} />
              <Row k="MODE"     v={hazardTel.mode?.toUpperCase()} />
              <Row k="REFERENCE" v="Derived photometric (not human GT)" dim />
              {testM && (
                <>
                  <Row k="PRECISION†" v={`${testM.test_precision?.toFixed(2)}%`} />
                  <Row k="RECALL†"    v={`${testM.test_recall?.toFixed(2)}%`} />
                  <Row k="F1†"        v={`${testM.test_f1?.toFixed(2)}%`} />
                  <Row k="IoU†"       v={`${testM.test_iou?.toFixed(2)}%`} />
                  <div className="rd-footnote">† Validated against derived photometric reference — not physical ground truth</div>
                </>
              )}
              {!testM && modeM && (
                <>
                  <Row k="F1 (val)"  v={`${modeM.f1.toFixed(2)}%`} />
                  <Row k="IoU (val)" v={`${modeM.iou.toFixed(2)}%`} />
                  <div className="rd-footnote">† System evaluation metrics — not per-image results</div>
                </>
              )}
            </section>
          )}

          {/* Image stats */}
          {imageStats && Object.keys(imageStats).length > 0 && (
            <section className="rd-section">
              <div className="rd-section-title">IMAGE STATISTICS</div>
              {[
                ['BRIGHTNESS MEAN', imageStats.brightness_mean?.toFixed(1)],
                ['BRIGHTNESS STD',  imageStats.brightness_std?.toFixed(1)],
                ['DYNAMIC RANGE',   imageStats.dynamic_range],
                ['EDGE DENSITY',    imageStats.edge_density != null ? `${(imageStats.edge_density * 100).toFixed(1)}%` : null],
              ].filter(([,v]) => v != null).map(([k, v]) => (
                <Row key={k} k={k} v={v} />
              ))}
              <div className="rd-footnote">Computed from restored image</div>
            </section>
          )}

          {/* Processing timing */}
          {processingTimeline && Object.keys(processingTimeline).length > 0 && (
            <section className="rd-section">
              <div className="rd-section-title">PROCESSING TIMELINE</div>
              {Object.entries(processingTimeline)
                .filter(([, v]) => typeof v === 'number' && v > 0)
                .map(([k, v]) => (
                  <Row key={k}
                    k={k.replace(/_ms$/,'').replace(/_/g,' ').toUpperCase()}
                    v={v > 1000 ? `${(v/1000).toFixed(2)}s` : `${Math.round(v)}ms`}
                  />
                ))}
            </section>
          )}

          {/* Scientific limitations */}
          {scientificLimitations.length > 0 && (
            <section className="rd-section">
              <div className="rd-section-title">SCIENTIFIC LIMITATIONS</div>
              {scientificLimitations.map((item, i) => (
                <div key={i} className="rd-limitation">{item}</div>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ k, v, dim }) {
  return (
    <div className="rd-row">
      <span className="rd-key">{k}</span>
      <span className={`rd-val${dim ? ' rd-val--dim' : ''}`}>{v ?? '—'}</span>
    </div>
  );
}
