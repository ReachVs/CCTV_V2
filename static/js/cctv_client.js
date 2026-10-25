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

        // Course & timetable tracking
        this.selectedCourseId = null;
        this.coursesList = [];
        this.courseSelect = document.getElementById('att-course-select');

        // Apple Date Picker state (§1, §4, §12)
        const todayObj = new Date();
        const year = todayObj.getFullYear();
        const month = String(todayObj.getMonth() + 1).padStart(2, '0');
        const day = String(todayObj.getDate()).padStart(2, '0');
        this.selectedDate = `${year}-${month}-${day}`;

        // Teacher override modal state
        this.activeOverrideStudent = null;
        this.overrideSelectedStatus = 'PRESENT';
        this.overrideModal = document.getElementById('attendance-override-modal');

        // Unusual person / intruder state
        this.unusualAlerts = [];
        this.unusualAlertsContainer = document.getElementById('unusual-alert-container');
    }

    init() {
        this.startClock();
        this.initAudioContext();
        this.connectWebSocket();
        this.initEventListeners();
        this.setupDateControls();
        this.startTelemetryPollingFallback();
        this.fetchInitialStats();
        this.fetchAttendanceConfig();
        this.fetchCourses();
        this.fetchUnusualAlerts();
        this.fetchAttendanceChecklist();
    }

    startClock() {
        const update = () => {
            const now = new Date();
            const cambodiaTz = 'Asia/Phnom_Penh';
            if (this.clockTimeEl) {
                this.clockTimeEl.textContent = now.toLocaleTimeString('en-US', {
                    timeZone: cambodiaTz,
                    hour: 'numeric',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: true
                }) + ' ICT';
            }
            if (this.clockDateEl) {
                this.clockDateEl.textContent = now.toLocaleDateString('en-CA', {
                    timeZone: cambodiaTz
                });
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
            } else if (type === 'tap' || type === 'click') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(1400, now);
                osc.frequency.exponentialRampToValueAtTime(800, now + 0.015);
                gain.gain.setValueAtTime(0.02, now);
                gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.015);
                osc.start(now);
                osc.stop(now + 0.015);
            } else if (type === 'alert') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(320, now);
                osc.frequency.setValueAtTime(260, now + 0.08);
                gain.gain.setValueAtTime(0.04, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
                osc.start(now);
                osc.stop(now + 0.22);
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

    formatTo12Hour(timeStr, includeSeconds = false) {
        if (!timeStr || typeof timeStr !== 'string') return '--:--';
        const clean = timeStr.trim();
        if (clean.includes('AM') || clean.includes('PM') || clean.includes('am') || clean.includes('pm')) {
            return clean;
        }
        const parts = clean.split(':');
        if (parts.length < 2) return clean;
        let hours = parseInt(parts[0], 10);
        const minutes = parts[1].padStart(2, '0');
        const seconds = parts[2] ? parts[2].padStart(2, '0') : '';
        if (isNaN(hours)) return clean;
        const period = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        if (hours === 0) hours = 12;
        if (includeSeconds && seconds) {
            return `${hours}:${minutes}:${seconds} ${period}`;
        }
        return `${hours}:${minutes} ${period}`;
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

            // Periodically refresh attendance checklist and unusual person alerts
            try {
                await this.fetchAttendanceChecklist();
                await this.fetchUnusualAlerts();
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

        const now = performance.now();
        // Suspend destructive DOM re-renders while user is hovering over telemetry cards to prevent flicker
        if (this._isHoveringTelemetry) {
            return;
        }
        this._lastTrackSignature = signature;
        this.lastTelemetryRenderTime = now;

        const html = tracks.map(t => {
            const name = t.name || 'Unknown';
            const meta = t.meta || {};
            const isConfirmed = name !== 'Unknown' && !name.startsWith('Scanning') && !name.startsWith('Unverified');
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

            // Audio notification and immediate attendance check-in update on new verified identification
            if (isConfirmed && !this.lastAuditedTracks.has(trackId)) {
                this.lastAuditedTracks.add(trackId);
                this.playTacticalChime('verified');
                // Trigger immediate checklist refresh to mark subject as PRESENT without polling delay
                this.fetchAttendanceChecklist();
            }

            const barColorClass = confidencePct >= 75 ? '' : (confidencePct >= 50 ? 'warn' : 'alert');
            const clickableClass = isConfirmed ? 'clickable' : '';
            const clickAttr = isConfirmed ? 'data-clickable="true"' : 'data-clickable="false"';
            const cursorStyle = isConfirmed ? 'cursor: pointer;' : 'cursor: default;';
            const clickHandler = isConfirmed ? `onclick="window.cctvClient && window.cctvClient.openDossierModal('${this.escapeHtml(name)}', '${encodeURIComponent(trackId)}')"` : '';

            return `
                <div class="hud-telemetry-card ${badgeClass} ${clickableClass}" 
                     data-track-id="${encodeURIComponent(trackId)}"
                     data-subject-name="${this.escapeHtml(name)}"
                     ${clickAttr}
                     ${clickHandler}
                     style="${cursorStyle}">
                    
                    <!-- Card Header -->
                    <div class="hud-card-top">
                        <span class="hud-card-prompt">TRACK #${encodeURIComponent(trackId)}</span>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            ${isConfirmed ? '<span class="hud-dossier-pill">DOSSIER ↗</span>' : ''}
                            <span class="status-badge ${badgeClass}" style="font-size: 0.62rem;">${this.escapeHtml(statusText)}</span>
                        </div>
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
            let rawTime = e.timestamp ? (e.timestamp.split('T')[1]?.split('.')[0] || e.timestamp) : 'NOW';
            if (rawTime.includes(' ')) {
                const spaceParts = rawTime.split(' ');
                rawTime = spaceParts[spaceParts.length - 1];
            }
            const timeStr = this.formatTo12Hour(rawTime, true);
            const isAlert = action.includes('DENIED') || action.includes('UNKNOWN');
            const isKnown = name && name !== 'Subject' && name !== 'Unknown' && !name.startsWith('Scanning') && !name.startsWith('Unverified');
            const auditClick = isKnown ? `onclick="window.cctvClient && window.cctvClient.openDossierModal('${this.escapeHtml(name)}', '${encodeURIComponent(e.track_id || '')}')" style="cursor: pointer;"` : `style="cursor: default;"`;

            return `
                <div class="audit-item" ${auditClick}>
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
        // Only confirmed identified personnel are allowed to open the classified dossier
        if (!name || name === 'Unknown' || name.startsWith('Scanning') || name.startsWith('Unverified')) {
            return;
        }

        const modal = document.getElementById('dossier-modal');
        if (!modal) return;

        // Reset fields to loading state
        const nameEl = document.getElementById('modal-fullname');
        const accessEl = document.getElementById('modal-access');
        const statusEl = document.getElementById('modal-status');
        const occEl = document.getElementById('modal-occupation');
        const phoneEl = document.getElementById('modal-phone');
        const addressEl = document.getElementById('modal-address');
        const dobEl = document.getElementById('modal-dob');
        const notesEl = document.getElementById('modal-intel-notes');

        if (nameEl) nameEl.textContent = name;
        if (accessEl) {
            accessEl.textContent = 'QUERYING BIOMETRIC REPOSITORY...';
            accessEl.className = 'd-val access-granted';
            accessEl.style.color = '';
        }
        if (statusEl) statusEl.textContent = 'CONFIRMED CONSENSUS LOCK';
        if (occEl) occEl.textContent = 'REGISTERED PERSONNEL';
        if (notesEl) notesEl.textContent = 'Retrieving classified multi-vector biometric profile...';
        
        // Initialize carousel with avatar fallback while fetching full set
        this.setupDossierSlider([
            { url: `/api/subject/avatar/${encodeURIComponent(name)}`, filename: 'Frontal.jpg' }
        ]);

        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');

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

                if (data.images && Array.isArray(data.images) && data.images.length > 0) {
                    this.setupDossierSlider(data.images);
                }
            }
        } catch (err) {
            console.warn('[CCTVClient] Failed to load dossier:', err);
        }
    }

    formatAngleLabel(filename = '') {
        const clean = filename.toLowerCase();
        if (clean.includes('frontal')) return 'FRONTAL VIEW';
        if (clean.includes('side_left') || clean.includes('side left')) return 'SIDE PROFILE (LEFT)';
        if (clean.includes('side_right') || clean.includes('side right')) return 'SIDE PROFILE (RIGHT)';
        if (clean.includes('left')) return 'LEFT 45° ANGLE';
        if (clean.includes('right')) return 'RIGHT 45° ANGLE';
        if (clean.includes('tilt_up') || clean.includes('tilt up')) return 'TILT UP (CHIN)';
        if (clean.includes('tilt_down') || clean.includes('tilt down')) return 'TILT DOWN';
        return filename.replace(/\.[^/.]+$/, '').replace(/_/g, ' ').toUpperCase() || 'BIOMETRIC ANGLE';
    }

    setupDossierSlider(images) {
        this.sliderImages = (images && images.length > 0) ? images : [{ url: '/logo.png', filename: 'Profile.jpg' }];
        this.currentSlideIndex = 0;
        const total = this.sliderImages.length;

        const track = document.getElementById('dossier-slider-track');
        const thumbsContainer = document.getElementById('slider-thumbs');
        const prevBtn = document.getElementById('slider-btn-prev');
        const nextBtn = document.getElementById('slider-btn-next');
        const viewport = document.getElementById('slider-viewport');

        if (!track) return;

        // 1. Populate Track Slides
        track.innerHTML = this.sliderImages.map((img, idx) => {
            const label = this.formatAngleLabel(img.filename || '');
            return `
                <div class="dossier-slide" data-slide-index="${idx}">
                    <img src="${this.escapeHtml(img.url)}" alt="${this.escapeHtml(label)}" onerror="this.src='/logo.png'">
                </div>
            `;
        }).join('');

        // 2. Wire < (Previous) and > (Next) Buttons
        if (prevBtn) {
            prevBtn.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.goToSlide(this.currentSlideIndex - 1);
            };
        }

        if (nextBtn) {
            nextBtn.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.goToSlide(this.currentSlideIndex + 1);
            };
        }

        // 3. Populate Mini Thumbnails Strip
        if (thumbsContainer) {
            thumbsContainer.innerHTML = this.sliderImages.map((img, idx) => {
                const label = this.formatAngleLabel(img.filename || '');
                return `
                    <div class="slider-thumb-item ${idx === 0 ? 'active' : ''}" data-index="${idx}" title="${this.escapeHtml(label)}">
                        <img src="${this.escapeHtml(img.url)}" alt="${this.escapeHtml(label)}" onerror="this.src='/logo.png'">
                    </div>
                `;
            }).join('');

            thumbsContainer.onclick = (e) => {
                const thumb = e.target.closest('.slider-thumb-item');
                if (thumb) {
                    const idx = parseInt(thumb.getAttribute('data-index'), 10);
                    if (!isNaN(idx)) this.goToSlide(idx);
                }
            };
        }

        // 4. Touch Swipe on Mobile/Tablet
        if (viewport && !viewport._sliderTouchBound) {
            viewport._sliderTouchBound = true;
            let touchStartX = 0;
            let touchEndX = 0;

            viewport.addEventListener('touchstart', (e) => {
                if (e.touches && e.touches.length === 1) {
                    touchStartX = e.touches[0].clientX;
                    touchEndX = e.touches[0].clientX;
                }
            }, { passive: true });

            viewport.addEventListener('touchmove', (e) => {
                if (e.touches && e.touches.length === 1) {
                    touchEndX = e.touches[0].clientX;
                }
            }, { passive: true });

            viewport.addEventListener('touchend', () => {
                const diff = touchEndX - touchStartX;
                if (diff < -40) this.goToSlide(this.currentSlideIndex + 1);
                else if (diff > 40) this.goToSlide(this.currentSlideIndex - 1);
            }, { passive: true });
        }

        this.goToSlide(0);
    }

    goToSlide(index) {
        if (!this.sliderImages || this.sliderImages.length === 0) return;
        const total = this.sliderImages.length;
        let newIndex = index;
        if (newIndex < 0) newIndex = total - 1;
        if (newIndex >= total) newIndex = 0;

        this.currentSlideIndex = newIndex;

        // Slide Track
        const track = document.getElementById('dossier-slider-track');
        if (track) {
            track.style.transform = `translateX(-${newIndex * 100}%)`;
        }

        // Counter
        const counterEl = document.getElementById('slider-counter');
        if (counterEl) {
            counterEl.textContent = `${newIndex + 1} / ${total}`;
        }

        // Angle label
        const angleEl = document.getElementById('slider-angle-label');
        if (angleEl && this.sliderImages[newIndex]) {
            angleEl.textContent = this.formatAngleLabel(this.sliderImages[newIndex].filename || '');
        }

        // Update Thumbnails
        const thumbs = document.querySelectorAll('.slider-thumbs-strip .slider-thumb-item');
        thumbs.forEach((thumb, idx) => {
            const isActive = idx === newIndex;
            thumb.classList.toggle('active', isActive);
            if (isActive) {
                thumb.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
            }
        });
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
                if (card && card.getAttribute('data-clickable') === 'true') {
                    const subjectName = card.getAttribute('data-subject-name') || '';
                    const trackId = card.getAttribute('data-track-id') || '';
                    this.openDossierModal(subjectName, trackId);
                }
            });
            this.telemetryContainer.addEventListener('mouseenter', () => { this._isHoveringTelemetry = true; });
            this.telemetryContainer.addEventListener('mouseleave', () => { this._isHoveringTelemetry = false; });
        }

        // Attendance Save Schedule button (Class Start & End in 12-Hour AM/PM)
        const saveCutoffBtn = document.getElementById('att-btn-save-cutoff');
        const start12hInput = document.getElementById('att-start-12h');
        const startAmpmBtn = document.getElementById('att-start-ampm');
        const end12hInput = document.getElementById('att-end-12h');
        const endAmpmBtn = document.getElementById('att-end-ampm');
        const graceInput = document.getElementById('att-grace-input');

        // Toggle AM/PM on click
        const toggleAmpm = (btn) => {
            if (!btn) return;
            const cur = btn.textContent.trim().toUpperCase();
            if (cur === 'PM') {
                btn.textContent = 'AM';
                btn.classList.add('am');
            } else {
                btn.textContent = 'PM';
                btn.classList.remove('am');
            }
        };

        if (startAmpmBtn) startAmpmBtn.addEventListener('click', () => toggleAmpm(startAmpmBtn));
        if (endAmpmBtn) endAmpmBtn.addEventListener('click', () => toggleAmpm(endAmpmBtn));

        if (saveCutoffBtn) {
            const handleSave = () => {
                const parseTo24h = (timeVal, ampmVal) => {
                    let [hh, mm] = (timeVal || '12:00').trim().split(':');
                    let h = parseInt(hh, 10) || 12;
                    let m = parseInt(mm, 10) || 0;
                    const isPm = (ampmVal || 'PM').toUpperCase() === 'PM';
                    if (isPm && h < 12) h += 12;
                    if (!isPm && h === 12) h = 0;
                    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:00`;
                };

                const startVal = parseTo24h(start12hInput ? start12hInput.value : '05:09', startAmpmBtn ? startAmpmBtn.textContent : 'PM');
                const endVal = parseTo24h(end12hInput ? end12hInput.value : '05:10', endAmpmBtn ? endAmpmBtn.textContent : 'PM');
                const grace = parseInt(graceInput ? graceInput.value : '15', 10) || 0;

                // Client-side validation: Class End Time must be strictly after Class Start Time
                if (endVal <= startVal) {
                    alert(`Invalid Schedule Conflict:\nClass End Time (${this.formatTo12Hour(endVal)}) cannot be earlier than or equal to Class Start Time (${this.formatTo12Hour(startVal)}).\n\nPlease verify that your START and END times, including AM/PM, are configured in chronological order.`);
                    if (end12hInput) {
                        end12hInput.style.borderColor = '#ef4444';
                        end12hInput.focus();
                        setTimeout(() => { if (end12hInput) end12hInput.style.borderColor = ''; }, 3000);
                    }
                    return;
                }

                this.saveAttendanceConfig(startVal, endVal, grace);
            };
            saveCutoffBtn.addEventListener('click', handleSave);
            if (start12hInput) {
                start12hInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') handleSave();
                });
            }
            if (end12hInput) {
                end12hInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') handleSave();
                });
            }
        }

        // Apple Date Control Bar Handlers
        const prevDateBtn = document.getElementById('att-date-prev');
        if (prevDateBtn) {
            prevDateBtn.addEventListener('click', () => this.shiftSelectedDate(-1));
        }
        const nextDateBtn = document.getElementById('att-date-next');
        if (nextDateBtn) {
            nextDateBtn.addEventListener('click', () => this.shiftSelectedDate(1));
        }
        const todayDateBtn = document.getElementById('att-date-today-btn');
        if (todayDateBtn) {
            todayDateBtn.addEventListener('click', () => this.resetDateToToday());
        }

        // Attendance Check All button (Marks all enrolled students in class)
        const checkAllBtn = document.getElementById('att-btn-check-all');
        if (checkAllBtn) {
            checkAllBtn.addEventListener('click', () => {
                this.checkAllAttendance('PRESENT');
            });
        }

        // Attendance Export CSV button
        const exportCsvBtn = document.getElementById('att-btn-export-csv');
        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => {
                this.exportAttendanceCSV();
            });
        }

        // Attendance Reset button
        const resetAttBtn = document.getElementById('att-btn-reset');
        if (resetAttBtn) {
            resetAttBtn.addEventListener('click', () => {
                if (confirm("Are you sure you want to reset today's attendance checklist?")) {
                    this.resetAttendance();
                }
            });
        }

        // Course Selector dropdown
        if (this.courseSelect) {
            this.courseSelect.addEventListener('change', () => {
                const val = this.courseSelect.value;
                this.selectedCourseId = val ? parseInt(val, 10) : null;
                this.fetchAttendanceChecklist();
            });
        }

        // Pinned Unusual Person Alerts actions
        if (this.unusualAlertsContainer) {
            this.unusualAlertsContainer.addEventListener('click', (e) => {
                const disBtn = e.target.closest('.btn-unusual-dismiss');
                if (disBtn) {
                    const trackId = disBtn.getAttribute('data-track-id');
                    if (trackId) this.dismissUnusualAlert(trackId);
                    return;
                }
                const enrollBtn = e.target.closest('.btn-unusual-enroll');
                if (enrollBtn) {
                    const trackId = enrollBtn.getAttribute('data-track-id');
                    if (trackId) this.enrollUnusualAlert(trackId);
                    return;
                }
            });
        }

        // Delegated click for attendance status override modal and row checkmarks
        const rosterContainer = document.getElementById('attendance-roster-list');
        if (rosterContainer) {
            rosterContainer.addEventListener('click', (e) => {
                const checkBtn = e.target.closest('.att-check-toggle');
                if (checkBtn) {
                    const name = checkBtn.getAttribute('data-name');
                    const curStatus = checkBtn.getAttribute('data-status');
                    if (name) {
                        this.manualToggleAttendance(name, curStatus || 'ABSENT');
                    }
                    return;
                }

                const btn = e.target.closest('.att-status-toggle');
                if (btn) {
                    const name = btn.getAttribute('data-name');
                    const curStatus = btn.getAttribute('data-status');
                    if (name) {
                        this.openOverrideModal(name, curStatus || 'PENDING');
                    }
                }
            });
        }

        // Teacher Override Modal event listeners
        const statusBtns = document.querySelectorAll('#override-status-buttons .override-btn-status');
        statusBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                statusBtns.forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.overrideSelectedStatus = btn.getAttribute('data-status') || 'PRESENT';
            });
        });

        const closeOverrideBtn = document.getElementById('btn-close-override-modal');
        if (closeOverrideBtn) {
            closeOverrideBtn.addEventListener('click', () => this.closeOverrideModal());
        }
        const cancelOverrideBtn = document.getElementById('btn-cancel-override');
        if (cancelOverrideBtn) {
            cancelOverrideBtn.addEventListener('click', () => this.closeOverrideModal());
        }
        const saveOverrideBtn = document.getElementById('btn-save-override');
        if (saveOverrideBtn) {
            saveOverrideBtn.addEventListener('click', () => this.saveOverride());
        }
        if (this.overrideModal) {
            this.overrideModal.addEventListener('click', (e) => {
                if (e.target === this.overrideModal) {
                    this.closeOverrideModal();
                }
            });
        }

        // ESC key to close modal or drawer, Arrow keys to slide images
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeDossierModal();
                this.closeOverrideModal();
                if (window.appDrawer) window.appDrawer.closeDrawer();
                return;
            }

            const modal = document.getElementById('dossier-modal');
            if (modal && modal.classList.contains('active')) {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    this.goToSlide(this.currentSlideIndex - 1);
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    this.goToSlide(this.currentSlideIndex + 1);
                }
            }
        });
    }

    async fetchAttendanceConfig() {
        try {
            const res = await fetch('/api/attendance/config');
            if (res.ok) {
                const data = await res.json();
                const startTime = data.class_start_time || data.cutoff_time || '18:00:00';
                const endTime = data.class_end_time || '21:00:00';

                const schedDisplay = document.getElementById('att-schedule-display');
                const badgeCutoff = document.getElementById('att-badge-cutoff');
                if (schedDisplay) {
                    if (startTime >= endTime) {
                        schedDisplay.textContent = `CONFLICT: ${this.formatTo12Hour(startTime)} - ${this.formatTo12Hour(endTime)}`;
                        if (badgeCutoff) {
                            badgeCutoff.classList.remove('verified');
                            badgeCutoff.style.color = '#ef4444';
                            badgeCutoff.style.borderColor = '#ef4444';
                            badgeCutoff.style.background = 'rgba(239, 68, 68, 0.15)';
                        }
                    } else {
                        schedDisplay.textContent = `${this.formatTo12Hour(startTime)} - ${this.formatTo12Hour(endTime)}`;
                        if (badgeCutoff) {
                            badgeCutoff.classList.add('verified');
                            badgeCutoff.style.color = '';
                            badgeCutoff.style.borderColor = '';
                            badgeCutoff.style.background = '';
                        }
                    }
                }
                const cutoffDisplay = document.getElementById('att-cutoff-display');
                if (cutoffDisplay) {
                    cutoffDisplay.textContent = this.formatTo12Hour(startTime);
                }

                const set12hInput = (inputEl, ampmBtn, time24) => {
                    if (!time24 || !inputEl) return;
                    const parts = time24.split(':');
                    let h = parseInt(parts[0], 10) || 12;
                    const m = (parts[1] || '00').padStart(2, '0');
                    const isPm = h >= 12;
                    h = h % 12;
                    if (h === 0) h = 12;
                    if (document.activeElement !== inputEl) {
                        inputEl.value = `${String(h).padStart(2, '0')}:${m}`;
                    }
                    if (ampmBtn) {
                        ampmBtn.textContent = isPm ? 'PM' : 'AM';
                        if (isPm) ampmBtn.classList.remove('am');
                        else ampmBtn.classList.add('am');
                    }
                };

                const start12hInput = document.getElementById('att-start-12h');
                const startAmpmBtn = document.getElementById('att-start-ampm');
                set12hInput(start12hInput, startAmpmBtn, startTime);

                const end12hInput = document.getElementById('att-end-12h');
                const endAmpmBtn = document.getElementById('att-end-ampm');
                set12hInput(end12hInput, endAmpmBtn, endTime);

                const graceInput = document.getElementById('att-grace-input');
                if (graceInput && data.grace_period_minutes !== undefined && document.activeElement !== graceInput) {
                    graceInput.value = data.grace_period_minutes;
                }
            }
        } catch (err) {
            console.warn('[CCTVClient] Error fetching attendance config:', err);
        }
    }

    async saveAttendanceConfig(startTime, endTime = null, graceMinutes = 0) {
        try {
            const payload = {
                class_start_time: startTime,
                class_end_time: endTime || '21:00:00',
                cutoff_time: startTime,
                grace_period_minutes: graceMinutes
            };
            const res = await fetch('/api/attendance/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                const data = await res.json();
                const st = data.class_start_time || startTime;
                const et = data.class_end_time || endTime || '11:30:00';
                const schedDisplay = document.getElementById('att-schedule-display');
                const badgeCutoff = document.getElementById('att-badge-cutoff');
                if (schedDisplay) {
                    if (st >= et) {
                        schedDisplay.textContent = `CONFLICT: ${this.formatTo12Hour(st)} - ${this.formatTo12Hour(et)}`;
                        if (badgeCutoff) {
                            badgeCutoff.classList.remove('verified');
                            badgeCutoff.style.color = '#ef4444';
                            badgeCutoff.style.borderColor = '#ef4444';
                            badgeCutoff.style.background = 'rgba(239, 68, 68, 0.15)';
                        }
                    } else {
                        schedDisplay.textContent = `${this.formatTo12Hour(st)} - ${this.formatTo12Hour(et)}`;
                        if (badgeCutoff) {
                            badgeCutoff.classList.add('verified');
                            badgeCutoff.style.color = '';
                            badgeCutoff.style.borderColor = '';
                            badgeCutoff.style.background = '';
                        }
                    }
                }
                const cutoffDisplay = document.getElementById('att-cutoff-display');
                if (cutoffDisplay) {
                    cutoffDisplay.textContent = this.formatTo12Hour(st);
                }

                const set12hInput = (inputEl, ampmBtn, time24) => {
                    if (!time24 || !inputEl) return;
                    const parts = time24.split(':');
                    let h = parseInt(parts[0], 10) || 12;
                    const m = (parts[1] || '00').padStart(2, '0');
                    const isPm = h >= 12;
                    h = h % 12;
                    if (h === 0) h = 12;
                    inputEl.value = `${String(h).padStart(2, '0')}:${m}`;
                    if (ampmBtn) {
                        ampmBtn.textContent = isPm ? 'PM' : 'AM';
                        if (isPm) ampmBtn.classList.remove('am');
                        else ampmBtn.classList.add('am');
                    }
                };

                const start12hInput = document.getElementById('att-start-12h');
                const startAmpmBtn = document.getElementById('att-start-ampm');
                set12hInput(start12hInput, startAmpmBtn, st);

                const end12hInput = document.getElementById('att-end-12h');
                const endAmpmBtn = document.getElementById('att-end-ampm');
                set12hInput(end12hInput, endAmpmBtn, et);

                await this.fetchAttendanceChecklist();
            } else {
                const errData = await res.json().catch(() => ({}));
                alert(`Error setting schedule: ${errData.detail || 'Invalid format'}`);
            }
        } catch (err) {
            console.error('[CCTVClient] Error saving attendance config:', err);
        }
    }

    setupDateControls() {
        const nativePicker = document.getElementById('att-date-native-picker');
        const dateChip = document.getElementById('att-date-chip');
        if (nativePicker) {
            nativePicker.value = this.selectedDate;
            nativePicker.addEventListener('change', (e) => {
                if (e.target.value) {
                    this.selectedDate = e.target.value;
                    this.updateDateDisplay();
                    this.fetchAttendanceChecklist();
                }
            });
        }
        if (dateChip) {
            dateChip.addEventListener('click', (e) => {
                // If user clicks anywhere on the chip, pop open the OS graphical calendar picker
                if (nativePicker && typeof nativePicker.showPicker === 'function') {
                    try {
                        nativePicker.showPicker();
                    } catch (_) {
                        nativePicker.focus();
                    }
                }
            });
            dateChip.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    if (nativePicker && typeof nativePicker.showPicker === 'function') {
                        try {
                            nativePicker.showPicker();
                        } catch (_) {
                            nativePicker.focus();
                        }
                    }
                }
            });
        }
        this.updateDateDisplay();
    }

    shiftSelectedDate(deltaDays) {
        if (!this.selectedDate) return;
        const parts = this.selectedDate.split('-').map(Number);
        const curDate = new Date(parts[0], parts[1] - 1, parts[2]);
        curDate.setDate(curDate.getDate() + deltaDays);
        const y = curDate.getFullYear();
        const m = String(curDate.getMonth() + 1).padStart(2, '0');
        const d = String(curDate.getDate()).padStart(2, '0');
        this.selectedDate = `${y}-${m}-${d}`;
        const nativePicker = document.getElementById('att-date-native-picker');
        if (nativePicker) nativePicker.value = this.selectedDate;
        this.updateDateDisplay();
        this.fetchAttendanceChecklist();
    }

    resetDateToToday() {
        const todayObj = new Date();
        const y = todayObj.getFullYear();
        const m = String(todayObj.getMonth() + 1).padStart(2, '0');
        const d = String(todayObj.getDate()).padStart(2, '0');
        this.selectedDate = `${y}-${m}-${d}`;
        const nativePicker = document.getElementById('att-date-native-picker');
        if (nativePicker) nativePicker.value = this.selectedDate;
        this.updateDateDisplay();
        this.fetchAttendanceChecklist();
    }

    updateDateDisplay() {
        const labelEl = document.getElementById('att-date-display-label');
        const dateLbl = document.getElementById('att-date-label');
        if (!labelEl && !dateLbl) return;

        const todayObj = new Date();
        const todayStr = `${todayObj.getFullYear()}-${String(todayObj.getMonth() + 1).padStart(2, '0')}-${String(todayObj.getDate()).padStart(2, '0')}`;

        let display = this.selectedDate;
        if (this.selectedDate === todayStr) {
            display = 'TODAY';
        } else {
            const parts = this.selectedDate.split('-').map(Number);
            const d = new Date(parts[0], parts[1] - 1, parts[2]);
            const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            display = `${monthNames[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
        }

        if (labelEl) labelEl.textContent = display;
        if (dateLbl) dateLbl.textContent = display;
    }

    async checkAllAttendance(status = 'PRESENT') {
        if (this._isAttendanceBusy) return;
        this._isAttendanceBusy = true;
        try {
            const payload = {
                status: status,
                course_id: this.selectedCourseId,
                date: this.selectedDate
            };
            const res = await fetch('/api/attendance/check-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                await this.fetchAttendanceChecklist();
            } else {
                const errData = await res.json().catch(() => ({}));
                alert(`Error marking all students: ${errData.detail || 'Operation failed'}`);
            }
        } catch (err) {
            console.error('[CCTVClient] Error in checkAllAttendance:', err);
        } finally {
            this._isAttendanceBusy = false;
        }
    }

    async fetchCourses() {
        try {
            const res = await fetch('/api/courses');
            if (res.ok) {
                const data = await res.json();
                this.coursesList = data.courses || [];
                const courseSelect = document.getElementById('att-course-select');
                if (courseSelect) {
                    let html = '<option value="">Global Roster</option>';
                    this.coursesList.forEach(c => {
                        const sel = (this.selectedCourseId === c.id) ? ' selected' : '';
                        html += `<option value="${c.id}"${sel}>${this.escapeHtml(c.code)} - ${this.escapeHtml(c.name)}</option>`;
                    });
                    courseSelect.innerHTML = html;
                }
            }
        } catch (err) {
            console.warn('[CCTVClient] Error fetching courses:', err);
        }
    }

    async fetchUnusualAlerts() {
        const container = document.getElementById('unusual-alert-container');
        if (!container) return;
        try {
            const res = await fetch('/api/attendance/unusual');
            if (res.ok) {
                const data = await res.json();
                this.unusualAlerts = data.alerts || [];
                this.renderUnusualAlerts();
            }
        } catch (err) {
            console.warn('[CCTVClient] Error fetching unusual alerts:', err);
        }
    }

    renderUnusualAlerts() {
        const container = document.getElementById('unusual-alert-container');
        if (!container) return;
        if (!this.unusualAlerts || this.unusualAlerts.length === 0) {
            container.style.display = 'none';
            container.innerHTML = '';
            return;
        }

        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.gap = '8px';

        container.innerHTML = this.unusualAlerts.map(alert => {
            const thumbUrl = alert.snapshot_url || (alert.snapshot_path ? `/data/unusual_events/${alert.snapshot_path.split('/').pop()}` : '/logo.png');
            const seenTime = alert.last_seen || alert.first_seen || 'Active';
            return `
                <div class="unusual-alert-card" data-track-id="${alert.track_id}">
                    <div class="unusual-alert-header">
                        <div class="unusual-alert-badge">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                            UNUSUAL PERSON // INTRUDER (TRACK #${alert.track_id})
                        </div>
                        <span style="font-size: 0.60rem; color: #fca5a5; font-family: var(--font-mono);">${seenTime}</span>
                    </div>
                    <div class="unusual-alert-body">
                        <img src="${thumbUrl}" alt="Intruder crop" class="unusual-thumbnail" onerror="this.src='/logo.png'">
                        <div style="flex: 1; min-width: 0;">
                            <div style="font-size: 0.72rem; color: #fff; font-weight: 600;">Unidentified Non-Enrolled Subject</div>
                            <div style="font-size: 0.62rem; color: #fca5a5; font-family: var(--font-mono); margin-top: 2px;">
                                Sustained presence in surveillance field (&ge;15 scans)
                            </div>
                            <div class="unusual-alert-actions">
                                <button type="button" class="tool-btn btn-unusual-dismiss" data-track-id="${alert.track_id}" style="padding: 3px 8px; font-size: 0.64rem; color: #f87171; border-color: rgba(244,63,94,0.4);">
                                    Dismiss
                                </button>
                                <button type="button" class="tool-btn btn-accent btn-unusual-enroll" data-track-id="${alert.track_id}" style="padding: 3px 8px; font-size: 0.64rem;">
                                    + Quick Enroll
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    async dismissUnusualAlert(trackId) {
        try {
            const res = await fetch('/api/attendance/unusual/dismiss', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_id: parseInt(trackId, 10) })
            });
            if (res.ok) {
                await this.fetchUnusualAlerts();
            }
        } catch (err) {
            console.error('[CCTVClient] Error dismissing alert:', err);
        }
    }

    async enrollUnusualAlert(trackId) {
        const studentName = prompt('Enter student full name to enroll this subject:');
        if (!studentName || !studentName.trim()) return;

        try {
            const res = await fetch('/api/attendance/unusual/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    track_id: parseInt(trackId, 10),
                    person_name: studentName.trim()
                })
            });
            if (res.ok) {
                alert(`Successfully enrolled ${studentName.trim()}! ArcFace embeddings committed and FAISS reloaded.`);
                await this.fetchUnusualAlerts();
                await this.fetchAttendanceChecklist();
                await this.fetchInitialStats();
            } else {
                const errData = await res.json().catch(() => ({}));
                alert(`Enrollment failed: ${errData.detail || 'Could not enroll subject'}`);
            }
        } catch (err) {
            console.error('[CCTVClient] Error enrolling unusual person:', err);
        }
    }

    async fetchAttendanceChecklist() {
        try {
            const params = new URLSearchParams();
            if (this.selectedCourseId) {
                params.append('course_id', this.selectedCourseId);
            }
            if (this.selectedDate) {
                params.append('date', this.selectedDate);
            }
            const queryString = params.toString() ? `?${params.toString()}` : '';
            const res = await fetch(`/api/attendance/checklist${queryString}`);
            if (res.ok) {
                const data = await res.json();
                this.renderAttendanceChecklist(data);
            }
        } catch (err) {
            console.warn('[CCTVClient] Error fetching attendance checklist:', err);
        }
    }

    renderAttendanceChecklist(data) {
        if (!data) return;

        // Update KPIs (4 statuses: Present, Late, Pending, Absent)
        const presEl = document.getElementById('att-stat-present');
        if (presEl) presEl.textContent = data.present_count || 0;

        const lateEl = document.getElementById('att-stat-late');
        if (lateEl) lateEl.textContent = data.late_count || 0;

        const pendEl = document.getElementById('att-stat-pending');
        if (pendEl) pendEl.textContent = data.pending_count || 0;

        const absEl = document.getElementById('att-stat-absent');
        if (absEl) absEl.textContent = data.absent_count || 0;

        // Total enrolled students
        const totalEl = document.getElementById('att-stat-total');
        if (totalEl) totalEl.textContent = data.total_enrolled || 0;

        // Top bar metric pill: show attended count / total enrolled students
        const topRatePill = document.getElementById('stat-attendance-rate');
        if (topRatePill) {
            const attended = (data.present_count || 0) + (data.late_count || 0);
            topRatePill.textContent = `${attended} / ${data.total_enrolled || 0}`;
        }

        const dateLbl = document.getElementById('att-date-label');
        if (dateLbl && data.date) dateLbl.textContent = data.date;

        const schedDisplay = document.getElementById('att-schedule-display');
        const badgeCutoff = document.getElementById('att-badge-cutoff');
        if (schedDisplay && data.class_start_time && data.class_end_time) {
            if (data.class_start_time >= data.class_end_time) {
                schedDisplay.textContent = `CONFLICT: ${this.formatTo12Hour(data.class_start_time)} - ${this.formatTo12Hour(data.class_end_time)}`;
                if (badgeCutoff) {
                    badgeCutoff.classList.remove('verified');
                    badgeCutoff.style.color = '#ef4444';
                    badgeCutoff.style.borderColor = '#ef4444';
                    badgeCutoff.style.background = 'rgba(239, 68, 68, 0.15)';
                }
            } else {
                schedDisplay.textContent = `${this.formatTo12Hour(data.class_start_time)} - ${this.formatTo12Hour(data.class_end_time)}`;
                if (badgeCutoff) {
                    badgeCutoff.classList.add('verified');
                    badgeCutoff.style.color = '';
                    badgeCutoff.style.borderColor = '';
                    badgeCutoff.style.background = '';
                }
            }
        }
        const cutoffDisplay = document.getElementById('att-cutoff-display');
        if (cutoffDisplay && (data.class_start_time || data.cutoff_time)) {
            cutoffDisplay.textContent = this.formatTo12Hour(data.class_start_time || data.cutoff_time);
        }

        const rosterContainer = document.getElementById('attendance-roster-list');
        if (!rosterContainer) return;

        if (!data.checklist || data.checklist.length === 0) {
            rosterContainer.innerHTML = `
                <div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 0.72rem; font-family: var(--font-mono);">
                    NO ENROLLED STUDENTS FOUND FOR SELECTED ROSTER
                </div>
            `;
            return;
        }

        const rowsHtml = data.checklist.map(item => {
            const status = (item.status || 'ABSENT').toUpperCase();
            const isPresent = status === 'PRESENT';
            const isLate = status === 'LATE';
            const isPending = status === 'PENDING';
            const isAbsent = status === 'ABSENT';

            let badgeBg = 'rgba(244, 63, 94, 0.08)';
            let badgeColor = '#f87171';
            let badgeBorder = 'rgba(244, 63, 94, 0.25)';
            let iconSvg = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
            let checkToggleClass = 'unchecked';
            let checkIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg>`;

            if (isPresent) {
                badgeBg = 'rgba(16, 185, 129, 0.15)';
                badgeColor = 'var(--emerald-verified)';
                badgeBorder = 'rgba(16, 185, 129, 0.4)';
                iconSvg = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`;
                checkToggleClass = 'checked';
                checkIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`;
            } else if (isLate) {
                badgeBg = 'rgba(245, 158, 11, 0.15)';
                badgeColor = 'var(--amber-warning)';
                badgeBorder = 'rgba(245, 158, 11, 0.4)';
                iconSvg = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
                checkToggleClass = 'checked-late';
                checkIcon = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`;
            } else if (isPending) {
                badgeBg = 'rgba(234, 179, 8, 0.15)';
                badgeColor = '#facc15';
                badgeBorder = 'rgba(234, 179, 8, 0.4)';
                iconSvg = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 14 14"/></svg>`;
            }

            const avatarUrl = `/api/subject/avatar/${encodeURIComponent(item.person_name)}`;
            const timeDisplay = item.check_in_time ? this.formatTo12Hour(item.check_in_time, true) : '--:--:--';
            const occupationDisplay = (item.occupation && item.occupation !== 'SECURITY SPECIALIST') ? this.escapeHtml(item.occupation) : 'STUDENT';
            const manualTag = item.is_manual ? `<span class="att-manual-tag" title="Teacher manual override: ${this.escapeHtml(item.override_reason || 'Manual')}">MANUAL</span>` : '';

            return `
                <div class="attendance-student-row">
                    <div style="display: flex; align-items: center; gap: 8px; min-width: 0;">
                        <button type="button" class="att-check-toggle ${checkToggleClass}" data-name="${this.escapeHtml(item.person_name)}" data-status="${item.status}" title="Click to quick toggle check-in for ${this.escapeHtml(item.person_name)}">
                            ${checkIcon}
                        </button>
                        <img src="${avatarUrl}" alt="${this.escapeHtml(item.person_name)}" style="width: 26px; height: 26px; border-radius: 50%; object-fit: cover; border: 1px solid var(--border-medium); background: #000; flex-shrink: 0;" onerror="this.src='/logo.png'">
                        <div style="min-width: 0;">
                            <div style="font-size: 0.74rem; font-weight: 600; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: flex; align-items: center;">
                                <span style="overflow: hidden; text-overflow: ellipsis;">${this.escapeHtml(item.person_name)}</span>
                                ${manualTag}
                            </div>
                            <div style="font-size: 0.62rem; color: var(--text-muted); font-family: var(--font-mono);">
                                ${occupationDisplay}
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 0.68rem; color: ${item.check_in_time ? 'var(--text-secondary)' : 'var(--text-muted)'};">
                            ${timeDisplay}
                        </span>
                        <button type="button" class="att-status-toggle" data-name="${this.escapeHtml(item.person_name)}" data-status="${item.status}" title="Click to open teacher attendance override modal" style="background: ${badgeBg}; color: ${badgeColor}; border: 1px solid ${badgeBorder};">
                            ${iconSvg}
                            ${item.status}
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        rosterContainer.innerHTML = rowsHtml;
    }

    openOverrideModal(personName, currentStatus) {
        const modal = document.getElementById('attendance-override-modal');
        if (!modal) return;
        this.activeOverrideStudent = personName;
        this.overrideSelectedStatus = (currentStatus === 'PENDING') ? 'PRESENT' : (currentStatus || 'PRESENT');

        const nameEl = document.getElementById('override-modal-name');
        if (nameEl) nameEl.textContent = personName;

        const curStatusEl = document.getElementById('override-modal-cur-status');
        if (curStatusEl) {
            curStatusEl.textContent = currentStatus || 'UNKNOWN';
            curStatusEl.style.color = (currentStatus === 'PRESENT') ? 'var(--emerald-verified)' :
                                      (currentStatus === 'LATE') ? 'var(--amber-warning)' :
                                      (currentStatus === 'PENDING') ? '#facc15' : '#f87171';
        }

        const avatarEl = document.getElementById('override-modal-avatar');
        if (avatarEl) avatarEl.src = `/api/subject/avatar/${encodeURIComponent(personName)}`;

        const reasonInput = document.getElementById('override-reason-input');
        if (reasonInput) reasonInput.value = '';

        // Select the active status button
        const buttons = document.querySelectorAll('#override-status-buttons .override-btn-status');
        buttons.forEach(btn => {
            if (btn.getAttribute('data-status') === this.overrideSelectedStatus) {
                btn.classList.add('selected');
            } else {
                btn.classList.remove('selected');
            }
        });

        modal.style.display = 'flex';
        modal.classList.add('active');
    }

    closeOverrideModal() {
        const modal = document.getElementById('attendance-override-modal');
        if (modal) {
            modal.style.display = 'none';
            modal.classList.remove('active');
        }
        this.activeOverrideStudent = null;
    }

    async saveOverride() {
        if (!this.activeOverrideStudent) return;
        const reasonInput = document.getElementById('override-reason-input');
        const reason = reasonInput ? reasonInput.value.trim() : '';

        try {
            const res = await fetch('/api/attendance/manual', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    person_name: this.activeOverrideStudent,
                    status: this.overrideSelectedStatus,
                    reason: reason || 'Teacher manual override',
                    course_id: this.selectedCourseId,
                    date: this.selectedDate
                })
            });
            if (res.ok) {
                this.closeOverrideModal();
                await this.fetchAttendanceChecklist();
            } else {
                const errData = await res.json().catch(() => ({}));
                alert(`Override failed: ${errData.detail || 'Could not update status'}`);
            }
        } catch (err) {
            console.error('[CCTVClient] Error saving override:', err);
        }
    }

    exportAttendanceCSV() {
        const params = new URLSearchParams();
        if (this.selectedCourseId) {
            params.append('course_id', this.selectedCourseId);
        }
        if (this.selectedDate) {
            params.append('date', this.selectedDate);
        }
        const queryString = params.toString() ? `?${params.toString()}` : '';
        window.open(`/api/attendance/export${queryString}`, '_blank');
    }

    async manualToggleAttendance(personName, currentStatus) {
        if (this._isAttendanceBusy) return;
        this._isAttendanceBusy = true;
        // Toggle sequence: ABSENT/PENDING -> PRESENT -> LATE -> ABSENT
        let nextStatus = 'PRESENT';
        if (currentStatus === 'PRESENT') nextStatus = 'LATE';
        else if (currentStatus === 'LATE') nextStatus = 'ABSENT';
        else if (currentStatus === 'ABSENT' || currentStatus === 'PENDING') nextStatus = 'PRESENT';

        try {
            const res = await fetch('/api/attendance/manual', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    person_name: personName,
                    status: nextStatus,
                    course_id: this.selectedCourseId,
                    date: this.selectedDate,
                    reason: 'Quick toggle via attendance checklist'
                })
            });
            if (res.ok) {
                await this.fetchAttendanceChecklist();
            }
        } catch (err) {
            console.error('[CCTVClient] Error manually toggling attendance:', err);
        } finally {
            this._isAttendanceBusy = false;
        }
    }

    async resetAttendance() {
        try {
            const res = await fetch('/api/attendance/reset', { method: 'POST' });
            if (res.ok) {
                await this.fetchAttendanceChecklist();
            }
        } catch (err) {
            console.error('[CCTVClient] Error resetting attendance:', err);
        }
    }
}

window.CCTVClient = CCTVClient;

