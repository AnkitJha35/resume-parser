import React from 'react';
import type { ConfigResponse, HealthResponse } from '../types';
import { Activity, Cpu, Layers, Server } from 'lucide-react';

interface HeaderProps {
  health: HealthResponse | null;
  config: ConfigResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ health, config }) => {
  const isHealthy = health?.status === 'ok';

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="logo-badge">RP</div>
        <div>
          <h1 className="header-title">Resume Parser</h1>
          <span className="header-subtitle">Local Test Harness & Diagnostic Suite</span>
        </div>
      </div>

      <div className="header-status-group">
        <div className={`status-pill ${isHealthy ? 'status-ok' : 'status-err'}`}>
          <Activity size={14} />
          <span>{isHealthy ? 'Server Online' : 'Connecting...'}</span>
        </div>

        {config && (
          <>
            <div className="status-pill status-info">
              <Server size={14} />
              <span>Provider: <strong>{config.provider}</strong></span>
            </div>

            <div className="status-pill status-info">
              <Cpu size={14} />
              <span>Model: <strong>{config.model}</strong></span>
            </div>

            <div className="status-pill status-secondary">
              <Layers size={14} />
              <span>Rep: <strong>{config.representation}</strong></span>
            </div>
          </>
        )}
      </div>
    </header>
  );
};
