/**
 * ============================================================================
 * SLEEPER AGENT // TACTICAL SURVEILLANCE CLIENT
 * High-Performance WebSocket Stream Receiver, Biometric HUD Telemetry,
 * Real-Time Audit Ingestion, and Classified Subject Dossier Controller
 * ============================================================================
 */

class CCTVClient {
    constructor(options = {}) {
        this.videoImg = document.getElementById(options.videoImgId || 'video-feed');
        this.fpsEl = document.getElementById(options.fpsId || 'stat-fps');
        this.profilesEl = document.getElementById(options.profilesId || 'stat-profiles');
        this.tracksEl = document.getElementById(options.tracksId || 'stat-tracks');
        this.clockTimeEl = document.getElementById(options.clockTimeId || 'clock-time');
        this.clockDateEl = document.getElementById(options.clockDateId || 'clock-date');
        this.telemetryContainer = document.getElementById(options.telemetryContainerId || 'telemetry-cards-container');
        this.auditListEl = document.getElementById(options.auditListId || 'audit-list');
        this.auditSearchEl = document.getElementById(options.auditSearchId || 'audit-search');
        this.statusIndicator = document.getElementById(options.statusIndicatorId || 'connection-pill');

        this.ws = null;
        this.isWsConnected = false;
        this.reconnectTimer = null;
        this.audioCtx = null;
        this.audioEnabled = true;
        this.lastAuditedTracks = new Set();
        this.fpsCounter = 0;
        this.lastFpsTimestamp = performance.now();
        this.scanlinesEnabled = true;
        this.lastTelemetryRenderTime = 0;
    }

    init() {
        this.startClock();
        this.initAudioContext();
        this.connectWebSocket();
        this.initEventListeners();
        this.startTelemetryPollingFallback();
        this.fetchInitialStats();
    }

    startClock() {
        const update = () => {
            const now = new Date();
            if (this.clockTimeEl) {
                this.clockTimeEl.textContent = now.toLocaleTimeString('en-US', { hour12: false }) + ' ICT';
            }
            if (this.clockDateEl) {
                this.clockDateEl.textContent = now.toISOString().split('T')[0];
            }
        };
        update();
        setInterval(update, 1000);
    }

    initAudioContext() {
        const unlock = () => {
            if (!this.audioCtx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    this.audioCtx = new AudioContext();
                }
            }
            if (this.audioCtx && this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }
            window.removeEventListener('click', unlock);
        };
        window.addEventListener('click', unlock);
    }

    playTacticalChime(type = 'verified') {
        if (!this.audioEnabled || !this.audioCtx) return;
        try {
            if (this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            const now = this.audioCtx.currentTime;
            if (type === 'verified') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, now); // D5
                osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.12); // A5
                gain.gain.setValueAtTime(0.04, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
                osc.start(now);
                osc.stop(now + 0.25);
            } else if (type === 'alert') {
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(220, now);
                osc.frequency.setValueAtTime(196, now + 0.08);
                gain.gain.setValueAtTime(0.06, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
                osc.start(now);
                osc.stop(now + 0.3);
            }
        } catch (_) {}
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/video`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.isWsConnected = true;
                if (this.statusIndicator) {
                    this.statusIndicator.innerHTML = '<span class="status-badge verified">● WEBSOCKET LIVE</span>';
                }
            };

            this.ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.image && this.videoImg) {
                            this.videoImg.src = data.image;
                        }
                        if (data.fps !== undefined && this.fpsEl) {
                            this.fpsEl.textContent = Number(data.fps).toFixed(1);
                        }
                        if (data.active_tracks && Array.isArray(data.active_tracks)) {
                            if (this.tracksEl) this.tracksEl.textContent = data.active_tracks.length;
                            const now = performance.now();
                            if (now - this.lastTelemetryRenderTime >= 350) {
                                this.lastTelemetryRenderTime = now;
                                this.renderTrackTelemetry(data.active_tracks);
                            }
                        }
                        this.calculateFPS();
                    } catch (err) {
                        console.warn('[CCTVClient] JSON parse error:', err);
                    }
                } else if (event.data instanceof Blob) {
                    const url = URL.createObjectURL(event.data);
                    if (this.videoImg) {
                        const oldUrl = this.videoImg.src;
                        this.videoImg.src = url;
                        if (oldUrl && oldUrl.startsWith('blob:')) {
                            URL.revokeObjectURL(oldUrl);
                        }
                    }
                    this.calculateFPS();
                }
            };

            this.ws.onclose = () => {
                this.isWsConnected = false;
                if (this.statusIndicator) {
                    this.statusIndicator.innerHTML = '<span class="status-badge scanning">● HTTP FALLBACK</span>';
                }
                this.fallbackToHttpStream();
                this.scheduleReconnect();
            };

            this.ws.onerror = () => {
                this.ws.close();
            };
        } catch (_) {
            this.fallbackToHttpStream();
            this.scheduleReconnect();
        }
    }

    fallbackToHttpStream() {
        if (this.videoImg && !this.videoImg.src.includes('/video_feed')) {
            this.videoImg.src = '/video_feed';
        }
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => {
            this.connectWebSocket();
        }, 2500);
    }

    calculateFPS() {
        this.fpsCounter++;
        const now = performance.now();
        if (now - this.lastFpsTimestamp >= 1000) {
            const currentFps = (this.fpsCounter * 1000) / (now - this.lastFpsTimestamp);
            if (this.fpsEl && !this.isWsConnected) {
                this.fpsEl.textContent = currentFps.toFixed(1);
            }
            this.fpsCounter = 0;
            this.lastFpsTimestamp = now;
        }
    }

    async fetchInitialStats() {
        try {
            const res = await fetch('/api/stats');
            if (res.ok) {
                const stats = await res.json();
                if (stats.registered_faces !== undefined && this.profilesEl) {
                    this.profilesEl.textContent = stats.registered_faces;
                }
            }
        } catch (_) {}
    }

    startTelemetryPollingFallback() {
        setInterval(async () => {
            try {
                const res = await fetch('/api/telemetry');
                if (res.ok) {
                    const data = await res.json();
                    if (data.total_profiles !== undefined && this.profilesEl) {
                        this.profilesEl.textContent = data.total_profiles;
                    }
                    if (!this.isWsConnected && data.active_tracks) {
                        this.renderTrackTelemetry(data.active_tracks);
                    }
                    if (data.events && Array.isArray(data.events)) {
                        this.renderAuditEvents(data.events);
                    }
                }
            } catch (_) {}
        }, 1500);
    }

    escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    renderTrackTelemetry(tracks) {
        if (!this.telemetryContainer) return;

        if (!tracks || tracks.length === 0) {
            this.telemetryContainer.innerHTML = `
                <div style="padding: 24px 12px; text-align: center; color: var(--text-muted); font-size: 0.75rem; font-family: var(--font-mono); border: 1px dashed var(--border-medium); border-radius: var(--radius-md);">
                    NO SUBJECTS IN ACTIVE FIELD OF VIEW
                </div>
            `;
            this._lastTrackSignature = '';
            return;
        }

        // Skip DOM thrashing if telemetry content is unchanged
        const signature = tracks.map(t => `${t.track_id}:${t.name}:${t.match_distance}:${t.meta?.status}`).join('|');
        if (this._lastTrackSignature === signature) {
            return;
        }

        const now = Date.now();
        // If user is hovering or interacting with cards, delay re-render by up to 1.5s
        if (this._isHoveringTelemetry && this.lastTelemetryRenderTime && (now - this.lastTelemetryRenderTime < 1500)) {
            return;
        }
        this._lastTrackSignature = signature;
        this.lastTelemetryRenderTime = now;

        const html = tracks.map(t => {
            const name = t.name || 'Unknown';
            const meta = t.meta || {};
            const isConfirmed = name !== 'Unknown' && name !== 'Scanning...';
            const trackId = t.track_id !== undefined ? t.track_id : '?';
            
            // Confidence calculation
            const dist = t.match_distance !== undefined ? Number(t.match_distance) : 0.45;
            let confidencePct = Math.max(10, Math.min(99, Math.round((1.0 - (dist * 0.7)) * 100)));
            if (!isConfirmed) confidencePct = 35;

            const badgeClass = isConfirmed ? 'verified' : 'scanning';
            const statusText = isConfirmed ? (meta.status || 'CONFIRMED') : 'TRACKING';
            const clearanceText = meta.clearance || 'LEVEL 1';
            const roleText = meta.role || meta.occupation || (isConfirmed ? 'REGISTERED PERSONNEL' : 'UNIDENTIFIED SUBJECT');
            const avatarSrc = isConfirmed ? `/api/subject/avatar/${encodeURIComponent(name)}` : '/logo.png';

            // Audio notification on new verified identification
            if (isConfirmed && !this.lastAuditedTracks.has(trackId)) {
                this.lastAuditedTracks.add(trackId);
                this.playTacticalChime('verified');
            }

            const barColorClass = confidencePct >= 75 ? '' : (confidencePct >= 50 ? 'warn' : 'alert');

            return `
                <div class="hud-telemetry-card ${badgeClass}" 
                     data-track-id="${encodeURIComponent(trackId)}"
                     data-subject-name="${this.escapeHtml(name)}"
                     onclick="window.cctvClient && window.cctvClient.openDossierModal('${this.escapeHtml(name)}', '${encodeURIComponent(trackId)}')"
                     style="cursor: pointer;"
                     title="Click to view Classified Intelligence Dossier">
                    
                    <!-- Tactical HUD Corner Brackets -->
                    <div class="hud-corner top-left"></div>
                    <div class="hud-corner top-right"></div>
                    <div class="hud-corner bottom-left"></div>
                    <div class="hud-corner bottom-right"></div>

                    <!-- Card Header -->
                    <div class="hud-card-top">
                        <span class="hud-card-prompt">TRACK #${encodeURIComponent(trackId)}</span>
                        <span class="status-badge ${badgeClass}" style="font-size: 0.62rem;">${this.escapeHtml(statusText)}</span>
                    </div>

                    <!-- Card Body -->
                    <div class="hud-card-body">
                        <div class="hud-portrait-container">
                            <img src="${avatarSrc}" alt="${this.escapeHtml(name)}" onerror="this.src='/logo.png'">
                        </div>

                        <div class="hud-info-column">
                            <div class="info-row">
                                <span class="info-lbl">SUBJECT</span>
                                <span class="info-val" style="color: ${isConfirmed ? 'var(--cyan-accent)' : 'var(--text-secondary)'}; font-weight: 700;">${this.escapeHtml(name)}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-lbl">CLEARANCE</span>
                                <span class="info-val">${this.escapeHtml(clearanceText)}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-lbl">ASSIGNMENT</span>
                                <span class="info-val" style="font-size: 0.65rem;">${this.escapeHtml(roleText)}</span>
                            </div>

                            <!-- Face Match Confidence Meter -->
                            <div class="hud-match-block" style="margin-top: 4px;">
                                <div class="meter-label">
                                    <span>ARCFACE MATCH</span>
                                    <span>${confidencePct}%</span>
                                </div>
                                <div class="bar-bg">
                                    <div class="bar-fill ${barColorClass}" style="width: ${confidencePct}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        this.telemetryContainer.innerHTML = html;
    }

    renderAuditEvents(events) {
        if (!this.auditListEl) return;
        const query = (this.auditSearchEl ? this.auditSearchEl.value : '').toLowerCase().trim();

        const filtered = events.filter(e => {
            if (!query) return true;
            const name = (e.person_name || e.name || '').toLowerCase();
            const action = (e.action || '').toLowerCase();
            const track = String(e.track_id || '');
            return name.includes(query) || action.includes(query) || track.includes(query);
        });

        if (filtered.length === 0) {
            this.auditListEl.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 0.72rem; font-family: var(--font-mono);">NO RECENT AUDIT ENTRIES</div>';
            return;
        }

        this.auditListEl.innerHTML = filtered.slice(0, 20).map(e => {
            const name = e.person_name || e.name || 'Subject';
            const action = e.action || 'VERIFIED';
            const clearance = e.meta?.clearance || e.clearance || 'LEVEL 1';
            const timeStr = e.timestamp ? (e.timestamp.split('T')[1]?.split('.')[0] || e.timestamp) : 'NOW';
            const isAlert = action.includes('DENIED') || action.includes('UNKNOWN');

            return `
                <div class="audit-item" style="cursor: pointer;" onclick="window.cctvClient && window.cctvClient.openDossierModal('${this.escapeHtml(name)}', '${encodeURIComponent(e.track_id || '')}')">
                    <div>
                        <div class="audit-title" style="color: ${isAlert ? 'var(--rose-threat)' : 'var(--text-primary)'};">
                            ${this.escapeHtml(name)} // ${this.escapeHtml(action)}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.65rem; margin-top: 2px;">
                            Track #${encodeURIComponent(e.track_id || '?')} • Clearance: ${this.escapeHtml(clearance)}
                        </div>
                    </div>
                    <div class="audit-time">${this.escapeHtml(timeStr)}</div>
                </div>
            `;
        }).join('');
    }

    async openDossierModal(name, trackId = null) {
        const modal = document.getElementById('dossier-modal');
        if (!modal) return;

        const isUnverified = !name || name === 'Unknown' || name === 'Scanning...' || name === 'Unverified';

        // Reset fields to loading state
        const nameEl = document.getElementById('modal-fullname');
        const accessEl = document.getElementById('modal-access');
        const statusEl = document.getElementById('modal-status');
        const occEl = document.getElementById('modal-occupation');
        const phoneEl = document.getElementById('modal-phone');
        const addressEl = document.getElementById('modal-address');
        const dobEl = document.getElementById('modal-dob');
        const galleryEl = document.getElementById('dossier-gallery');
        const notesEl = document.getElementById('modal-intel-notes');

        if (isUnverified) {
            const trackLabel = trackId ? `TRACK #${trackId}` : 'UNIDENTIFIED SUBJECT';
            if (nameEl) nameEl.textContent = `${trackLabel} (UNVERIFIED)`;
            if (accessEl) {
                accessEl.textContent = 'RESTRICTED / PENDING IDENTIFICATION';
                accessEl.className = 'd-val';
                accessEl.style.color = 'var(--amber-warning)';
            }
            if (statusEl) statusEl.textContent = 'LIVE SCANNING / TRACKING IN PROGRESS';
            if (occEl) occEl.textContent = 'UNENROLLED CANDIDATE';
            if (phoneEl) phoneEl.textContent = 'N/A';
            if (addressEl) addressEl.textContent = 'SURVEILLANCE FIELD OF VIEW';
            if (dobEl) dobEl.textContent = 'N/A';
            if (notesEl) {
                notesEl.textContent = 'Target is actively tracked by spatial pose detection and ByteTrack. Biometric consensus has not yet reached the verification threshold against enrolled profiles. Click "Enroll Subject" in the navigation bar to register this individual into the biometric repository.';
            }
            if (galleryEl) {
                galleryEl.innerHTML = `
                    <div class="dossier-thumb" style="display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.03); border: 1px dashed var(--border-medium); padding: 16px; border-radius: var(--radius-sm); text-align: center; width: 100%; height: 100%;">
                        <img src="/logo.png" alt="Unverified" style="width: 48px; height: 48px; opacity: 0.6; margin-bottom: 8px;">
                        <span style="font-size: 0.68rem; color: var(--text-muted); font-family: var(--font-mono);">NO ENROLLED RECORD</span>
                        <button type="button" class="btn-primary-action" onclick="window.cctvClient && window.cctvClient.closeDossierModal(); window.appDrawer && window.appDrawer.openDrawer();" style="margin-top: 10px; font-size: 0.7rem; padding: 6px 12px; width: auto;">ENROLL SUBJECT</button>
                    </div>
                `;
            }
            modal.classList.add('active');
            return;
        }

        // For verified / known subject
        if (nameEl) nameEl.textContent = name;
        if (accessEl) {
            accessEl.textContent = 'QUERYING BIOMETRIC REPOSITORY...';
            accessEl.className = 'd-val access-granted';
            accessEl.style.color = '';
        }
        if (galleryEl) {
            galleryEl.innerHTML = `
                <div class="dossier-thumb">
                    <img src="/api/subject/avatar/${encodeURIComponent(name)}" alt="${this.escapeHtml(name)}" onerror="this.src='/logo.png'">
                </div>
            `;
        }

        modal.classList.add('active');

        try {
            const res = await fetch(`/api/subject/${encodeURIComponent(name)}`);
            if (res.ok) {
                const data = await res.json();
                const meta = data.meta || data;
                if (nameEl) nameEl.textContent = data.name || meta.name || name;
                if (accessEl) {
                    accessEl.textContent = meta.clearance || data.clearance || 'LEVEL 1 GRANTED';
                    accessEl.className = 'd-val access-granted';
                    accessEl.style.color = '';
                }
                if (statusEl) statusEl.textContent = 'ACTIVE BIOMETRIC INDEX';
                if (occEl) occEl.textContent = meta.occupation || data.occupation || 'REGISTERED PERSONNEL';
                if (phoneEl) phoneEl.textContent = meta.phone || data.phone || 'N/A';
                if (addressEl) addressEl.textContent = meta.address || data.address || 'HEADQUARTERS';
                if (dobEl) dobEl.textContent = meta.dob || data.dob || '2026';
                if (notesEl) {
                    const threat = meta.threat_level ? `Threat Level: ${meta.threat_level}. ` : '';
                    notesEl.textContent = `${threat}Authorized subject for automated facial matching. Exemplar vectors indexed in FAISS repository. Real-time consensus confirmed.`;
                }

                if (galleryEl && data.images && Array.isArray(data.images) && data.images.length > 0) {
                    galleryEl.innerHTML = data.images.map(img => `
                        <div class="dossier-thumb">
                            <img src="${this.escapeHtml(img.url)}" alt="${this.escapeHtml(img.filename)}" onerror="this.src='/logo.png'">
                        </div>
                    `).join('');
                }
            }
        } catch (err) {
            console.warn('[CCTVClient] Failed to load dossier:', err);
        }
    }

    closeDossierModal() {
        const modal = document.getElementById('dossier-modal');
        if (modal) modal.classList.remove('active');
    }

    initEventListeners() {
        // Snapshot button
        const snapshotBtn = document.getElementById('btn-snapshot');
        if (snapshotBtn && this.videoImg) {
            snapshotBtn.addEventListener('click', () => {
                const canvas = document.createElement('canvas');
                canvas.width = this.videoImg.naturalWidth || 1280;
                canvas.height = this.videoImg.naturalHeight || 720;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(this.videoImg, 0, 0, canvas.width, canvas.height);
                const a = document.createElement('a');
                a.href = canvas.toDataURL('image/jpeg');
                a.download = `SleeperAgent_Snapshot_${Date.now()}.jpg`;
                a.click();
            });
        }

        // Fullscreen button
        const fullscreenBtn = document.getElementById('btn-fullscreen');
        const viewportContainer = document.getElementById('viewport-container');
        if (fullscreenBtn && viewportContainer) {
            fullscreenBtn.addEventListener('click', () => {
                if (!document.fullscreenElement) {
                    viewportContainer.requestFullscreen().catch(() => {});
                } else {
                    document.exitFullscreen().catch(() => {});
                }
            });
        }

        // Scanlines toggle button
        const scanlinesBtn = document.getElementById('btn-scanlines');
        const scanlinesLayer = document.getElementById('scanlines-layer');
        if (scanlinesBtn && scanlinesLayer) {
            scanlinesBtn.addEventListener('click', () => {
                this.scanlinesEnabled = !this.scanlinesEnabled;
                scanlinesLayer.style.display = this.scanlinesEnabled ? 'block' : 'none';
                scanlinesBtn.classList.toggle('active', this.scanlinesEnabled);
            });
        }

        // Audio toggle button
        const audioBtn = document.getElementById('btn-audio-toggle');
        if (audioBtn) {
            audioBtn.addEventListener('click', () => {
                this.audioEnabled = !this.audioEnabled;
                audioBtn.classList.toggle('active', this.audioEnabled);
                if (this.audioEnabled) this.playTacticalChime('verified');
            });
        }

        // Dossier close buttons and backdrop click
        const dossierCloseBtn = document.getElementById('dossier-close-btn');
        if (dossierCloseBtn) {
            dossierCloseBtn.addEventListener('click', () => this.closeDossierModal());
        }

        const dossierModal = document.getElementById('dossier-modal');
        if (dossierModal) {
            dossierModal.addEventListener('click', (e) => {
                if (e.target === dossierModal) this.closeDossierModal();
            });
        }

        // Delegated click listener on telemetry cards container
        if (this.telemetryContainer) {
            this.telemetryContainer.addEventListener('click', (e) => {
                const card = e.target.closest('.hud-telemetry-card');
                if (card) {
                    const subjectName = card.getAttribute('data-subject-name') || '';
                    const trackId = card.getAttribute('data-track-id') || '';
                    this.openDossierModal(subjectName, trackId);
                }
            });
            this.telemetryContainer.addEventListener('mouseenter', () => { this._isHoveringTelemetry = true; });
            this.telemetryContainer.addEventListener('mouseleave', () => { this._isHoveringTelemetry = false; });
        }

        // ESC key to close modal or drawer
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeDossierModal();
                if (window.appDrawer) window.appDrawer.closeDrawer();
            }
        });
    }
}

window.CCTVClient = CCTVClient;
