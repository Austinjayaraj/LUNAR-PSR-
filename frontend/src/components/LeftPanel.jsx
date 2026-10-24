import { Upload, Play, RefreshCw, X } from 'lucide-react';
import useStore from '../store';
import ImageComparison from './ImageComparison';

export default function LeftPanel() {
  const inputFile = useStore((state) => state.inputFile);
  const inputInfo = useStore((state) => state.inputInfo);
  const rawImageUrl = useStore((state) => state.rawImageUrl);
  const restoredImageUrl = useStore((state) => state.restoredImageUrl);
  const missionStatus = useStore((state) => state.missionStatus);
  const viewMode = useStore((state) => state.viewMode);
  const analysisLayer = useStore((state) => state.analysisLayer);

  const setInputFile = useStore((state) => state.setInputFile);
  const runPipeline = useStore((state) => state.runPipeline);
  const reset = useStore((state) => state.reset);
  const setViewMode = useStore((state) => state.setViewMode);
  const setAnalysisLayer = useStore((state) => state.setAnalysisLayer);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setInputFile(file);
    }
  };

  const handleRunPipeline = () => {
    if (inputFile) {
      runPipeline(inputFile);
    }
  };

  const isProcessing = missionStatus === 'PROCESSING';

  return (
    <div className="left-panel">
      <div className="panel-card">
        <h3 className="panel-card-title">
          <Upload size={14} />
          IMAGE UPLOAD
        </h3>

        {!inputFile ? (
          <div className="upload-zone">
            <Upload size={32} className="upload-icon" />
            <p className="upload-text">DROP IMAGE HERE</p>
            <input
              type="file"
              id="file-input"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              disabled={isProcessing}
            />
            <label htmlFor="file-input" className="upload-button">
              BROWSE FILES
            </label>
          </div>
        ) : (
          <div className="file-preview">
            <img src={rawImageUrl} alt="Input" className="file-thumbnail" />
            <div className="file-info">
              <div className="file-name">{inputInfo?.filename || 'Unknown'}</div>
              <div className="file-meta">
                {inputInfo?.width || 0} × {inputInfo?.height || 0} px
              </div>
              <div className="file-meta">{((inputInfo?.file_size_bytes || 0) / 1024).toFixed(1)} KB</div>
            </div>
            <button className="remove-button" onClick={() => setInputFile(null)} disabled={isProcessing}>
              <X size={16} />
            </button>
          </div>
        )}
      </div>

      <div className="panel-card">
        <h3 className="panel-card-title">PIPELINE CONTROL</h3>
        <div className="control-buttons">
          <button className="primary-button" onClick={handleRunPipeline} disabled={!inputFile || isProcessing}>
            {isProcessing ? (
              <>
                <RefreshCw size={16} className="rotating" />
                PROCESSING...
              </>
            ) : (
              <>
                <Play size={16} />
                RUN PIPELINE
              </>
            )}
          </button>
          <button className="secondary-button" onClick={reset} disabled={isProcessing}>
            RESET MISSION
          </button>
        </div>
      </div>

      <div className="panel-card">
        <h3 className="panel-card-title">ANALYSIS LAYERS</h3>
        <div className="layer-options">
          {['RAW', 'RESTORED', 'EDGE', 'DANGER', 'SAFE_REGION', 'LANDING_ZONES'].map((layer) => (
            <label key={layer} className="layer-option">
              <input
                type="radio"
                name="analysis-layer"
                value={layer}
                checked={analysisLayer === layer}
                onChange={(e) => setAnalysisLayer(e.target.value)}
              />
              <span className="layer-label">{layer.replace('_', ' ')}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="panel-card">
        <h3 className="panel-card-title">VIEW MODE</h3>
        <div className="view-mode-toggle">
          <button
            className={`view-mode-button ${viewMode === '2D' ? 'active' : ''}`}
            onClick={() => setViewMode('2D')}
          >
            2D VIEW
          </button>
          <button
            className={`view-mode-button ${viewMode === '3D' ? 'active' : ''}`}
            onClick={() => setViewMode('3D')}
          >
            3D VIEW
          </button>
        </div>
      </div>

      {rawImageUrl && restoredImageUrl && (
        <div className="panel-card">
          <h3 className="panel-card-title">IMAGE COMPARISON</h3>
          <ImageComparison rawUrl={rawImageUrl} restoredUrl={restoredImageUrl} />
        </div>
      )}
    </div>
  );
}
