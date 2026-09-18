import React, { useState } from 'react';
import type { Resume } from '../types';
import {
  User,
  Briefcase,
  GraduationCap,
  FolderGit2,
  Cpu,
  Award,
  Globe,
  Code,
  Calendar,
  MapPin,
  Mail,
  Phone,
  ExternalLink,
  Copy,
  Check,
} from 'lucide-react';

interface ResumeViewerProps {
  resume: Resume;
}

type TabType =
  | 'overview'
  | 'personal'
  | 'experience'
  | 'education'
  | 'projects'
  | 'skills'
  | 'certifications'
  | 'languages'
  | 'raw';

export const ResumeViewer: React.FC<ResumeViewerProps> = ({ resume }) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [copied, setCopied] = useState(false);

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(resume, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="resume-viewer card">
      <div className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <User size={15} />
          <span>Overview</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'personal' ? 'active' : ''}`}
          onClick={() => setActiveTab('personal')}
        >
          <User size={15} />
          <span>Personal</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'experience' ? 'active' : ''}`}
          onClick={() => setActiveTab('experience')}
        >
          <Briefcase size={15} />
          <span>Experience ({resume.experience?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'education' ? 'active' : ''}`}
          onClick={() => setActiveTab('education')}
        >
          <GraduationCap size={15} />
          <span>Education ({resume.education?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'projects' ? 'active' : ''}`}
          onClick={() => setActiveTab('projects')}
        >
          <FolderGit2 size={15} />
          <span>Projects ({resume.projects?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'skills' ? 'active' : ''}`}
          onClick={() => setActiveTab('skills')}
        >
          <Cpu size={15} />
          <span>Skills ({resume.skills?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'certifications' ? 'active' : ''}`}
          onClick={() => setActiveTab('certifications')}
        >
          <Award size={15} />
          <span>Certifications ({resume.certifications?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'languages' ? 'active' : ''}`}
          onClick={() => setActiveTab('languages')}
        >
          <Globe size={15} />
          <span>Languages ({resume.languages?.length || 0})</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
          onClick={() => setActiveTab('raw')}
        >
          <Code size={15} />
          <span>Raw JSON</span>
        </button>
      </div>

      <div className="tab-content">
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="overview-section">
            <div className="overview-header">
              <div className="candidate-avatar">
                {(resume.personal?.name || 'U').charAt(0).toUpperCase()}
              </div>
              <div>
                <h2>{resume.personal?.name || 'Candidate Name Not Found'}</h2>
                <div className="candidate-contact-row">
                  {resume.personal?.email && (
                    <span>
                      <Mail size={14} /> {resume.personal.email}
                    </span>
                  )}
                  {resume.personal?.phone && (
                    <span>
                      <Phone size={14} /> {resume.personal.phone}
                    </span>
                  )}
                  {resume.personal?.location && (
                    <span>
                      <MapPin size={14} /> {resume.personal.location}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {resume.summary && (
              <div className="summary-box">
                <h4>Professional Summary</h4>
                <p>{resume.summary}</p>
              </div>
            )}

            <div className="overview-metrics-grid">
              <div className="metric-box">
                <span className="metric-num">{resume.experience?.length || 0}</span>
                <span className="metric-name">Work Positions</span>
              </div>
              <div className="metric-box">
                <span className="metric-num">{resume.education?.length || 0}</span>
                <span className="metric-name">Educational Records</span>
              </div>
              <div className="metric-box">
                <span className="metric-num">{resume.projects?.length || 0}</span>
                <span className="metric-name">Projects</span>
              </div>
              <div className="metric-box">
                <span className="metric-num">{resume.skills?.length || 0}</span>
                <span className="metric-name">Skills Extracted</span>
              </div>
            </div>

            {resume.skills && resume.skills.length > 0 && (
              <div className="overview-skills">
                <h4>Top Skills Sample</h4>
                <div className="badge-wrap">
                  {resume.skills.slice(0, 15).map((skill, i) => (
                    <span key={i} className="badge badge-skill">
                      {skill}
                    </span>
                  ))}
                  {resume.skills.length > 15 && (
                    <span className="badge badge-more">
                      +{resume.skills.length - 15} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* PERSONAL TAB */}
        {activeTab === 'personal' && (
          <div className="details-grid">
            <div className="detail-item">
              <span className="detail-label">Full Name</span>
              <span className="detail-value">{resume.personal?.name || '—'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Email Address</span>
              <span className="detail-value">{resume.personal?.email || '—'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Phone Number</span>
              <span className="detail-value">{resume.personal?.phone || '—'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Location</span>
              <span className="detail-value">{resume.personal?.location || '—'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">LinkedIn</span>
              <span className="detail-value">
                {resume.personal?.linkedin ? (
                  <a
                    href={resume.personal.linkedin.startsWith('http') ? resume.personal.linkedin : `https://${resume.personal.linkedin}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <ExternalLink size={14} /> {resume.personal.linkedin}
                  </a>
                ) : (
                  '—'
                )}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">GitHub</span>
              <span className="detail-value">
                {resume.personal?.github ? (
                  <a
                    href={resume.personal.github.startsWith('http') ? resume.personal.github : `https://${resume.personal.github}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <ExternalLink size={14} /> {resume.personal.github}
                  </a>
                ) : (
                  '—'
                )}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Portfolio / Website</span>
              <span className="detail-value">
                {resume.personal?.portfolio ? (
                  <a
                    href={resume.personal.portfolio.startsWith('http') ? resume.personal.portfolio : `https://${resume.personal.portfolio}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <ExternalLink size={14} /> {resume.personal.portfolio}
                  </a>
                ) : (
                  '—'
                )}
              </span>
            </div>
          </div>
        )}

        {/* EXPERIENCE TAB */}
        {activeTab === 'experience' && (
          <div className="timeline-container">
            {resume.experience && resume.experience.length > 0 ? (
              resume.experience.map((exp, idx) => (
                <div key={idx} className="timeline-card">
                  <div className="timeline-header">
                    <div>
                      <h3 className="timeline-title">{exp.designation || 'Untitled Role'}</h3>
                      <h4 className="timeline-subtitle">{exp.company || 'Unknown Company'}</h4>
                    </div>
                    <div className="timeline-badge-group">
                      <span className="date-badge">
                        <Calendar size={13} />
                        {exp.startDate || '?'} — {exp.current ? 'Present' : exp.endDate || '?'}
                      </span>
                      {exp.current && <span className="badge badge-current">Current</span>}
                    </div>
                  </div>

                  {exp.location && (
                    <div className="timeline-location">
                      <MapPin size={13} /> {exp.location}
                    </div>
                  )}

                  {exp.description && (
                    <div className="timeline-desc">
                      <p>{exp.description}</p>
                    </div>
                  )}

                  {exp.skills && exp.skills.length > 0 && (
                    <div className="timeline-tags">
                      {exp.skills.map((s, i) => (
                        <span key={i} className="badge badge-tag">
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <p className="empty-state">No experience entries extracted.</p>
            )}
          </div>
        )}

        {/* EDUCATION TAB */}
        {activeTab === 'education' && (
          <div className="timeline-container">
            {resume.education && resume.education.length > 0 ? (
              resume.education.map((edu, idx) => (
                <div key={idx} className="timeline-card">
                  <div className="timeline-header">
                    <div>
                      <h3 className="timeline-title">{edu.degree || 'Degree'}</h3>
                      <h4 className="timeline-subtitle">{edu.institution || 'Unknown Institution'}</h4>
                    </div>
                    <span className="date-badge">
                      <Calendar size={13} />
                      {edu.startDate || '?'} — {edu.endDate || '?'}
                    </span>
                  </div>

                  {edu.fieldOfStudy && (
                    <p className="edu-field">
                      <strong>Field:</strong> {edu.fieldOfStudy}
                    </p>
                  )}

                  {edu.grade && (
                    <p className="edu-grade">
                      <strong>Grade / GPA:</strong> {edu.grade}
                    </p>
                  )}
                </div>
              ))
            ) : (
              <p className="empty-state">No education entries extracted.</p>
            )}
          </div>
        )}

        {/* PROJECTS TAB */}
        {activeTab === 'projects' && (
          <div className="timeline-container">
            {resume.projects && resume.projects.length > 0 ? (
              resume.projects.map((prj, idx) => (
                <div key={idx} className="timeline-card">
                  <div className="timeline-header">
                    <h3 className="timeline-title">{prj.name || 'Untitled Project'}</h3>
                    {prj.url && (
                      <a
                        href={prj.url.startsWith('http') ? prj.url : `https://${prj.url}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="project-link"
                      >
                        <ExternalLink size={14} /> Link
                      </a>
                    )}
                  </div>

                  {prj.description && <p className="timeline-desc">{prj.description}</p>}

                  {prj.technologies && prj.technologies.length > 0 && (
                    <div className="timeline-tags">
                      {prj.technologies.map((t, i) => (
                        <span key={i} className="badge badge-tag">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <p className="empty-state">No project entries extracted.</p>
            )}
          </div>
        )}

        {/* SKILLS TAB */}
        {activeTab === 'skills' && (
          <div className="skills-container">
            {resume.skills && resume.skills.length > 0 ? (
              <div className="badge-wrap">
                {resume.skills.map((skill, idx) => (
                  <span key={idx} className="badge badge-skill-large">
                    {skill}
                  </span>
                ))}
              </div>
            ) : (
              <p className="empty-state">No skills extracted.</p>
            )}
          </div>
        )}

        {/* CERTIFICATIONS TAB */}
        {activeTab === 'certifications' && (
          <div className="list-container">
            {resume.certifications && resume.certifications.length > 0 ? (
              resume.certifications.map((cert, idx) => (
                <div key={idx} className="list-item-card">
                  <Award size={18} className="list-icon" />
                  <div>
                    <h4 className="list-title">{cert.name}</h4>
                    {cert.issuingOrganization && (
                      <p className="list-sub">{cert.issuingOrganization}</p>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p className="empty-state">No certifications extracted.</p>
            )}
          </div>
        )}

        {/* LANGUAGES TAB */}
        {activeTab === 'languages' && (
          <div className="list-container">
            {resume.languages && resume.languages.length > 0 ? (
              <div className="badge-wrap">
                {resume.languages.map((lang, idx) => (
                  <span key={idx} className="badge badge-lang">
                    <Globe size={13} /> {lang}
                  </span>
                ))}
              </div>
            ) : (
              <p className="empty-state">No languages extracted.</p>
            )}
          </div>
        )}

        {/* RAW JSON TAB */}
        {activeTab === 'raw' && (
          <div className="raw-json-view">
            <div className="raw-json-header">
              <span>Canonical Resume JSON</span>
              <button className="btn btn-sm btn-ghost" onClick={handleCopyJson}>
                {copied ? <Check size={14} /> : <Copy size={14} />}
                <span>{copied ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre className="json-code">
              {JSON.stringify(resume, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
