/**
 * ============================================================================
 * SLEEPER AGENT // SLIDE-OVER ENROLLMENT DRAWER & CURATION STUDIO
 * Zero-Friction Multi-Exemplar Biometric Ingestion Controller
 * ============================================================================
 */

class EnrollmentDrawer {
    constructor() {
        this.drawerEl = document.getElementById('enrollment-drawer');
        this.backdropEl = document.getElementById('drawer-backdrop');
        this.openBtns = document.querySelectorAll('[data-action="open-enroll-drawer"]');
        this.closeBtns = document.querySelectorAll('[data-action="close-enroll-drawer"]');

        this.dropzone = document.getElementById('drawer-dropzone');
        this.fileInput = document.getElementById('drawer-file-input');
        this.curationTray = document.getElementById('drawer-curation-tray');
        this.curationGrid = document.getElementById('drawer-curation-grid');
        this.commitBtn = document.getElementById('drawer-commit-btn');
        this.clearBtn = document.getElementById('drawer-clear-btn');
        this.statusIndicator = document.getElementById('drawer-status-indicator');
        this.toastMsg = document.getElementById('drawer-toast');

        this.uploadedPreviews = []; // { crop_id, image_b64, sharpness_score, excluded: false }
    }

    init() {
        this.bindDrawerTriggers();
        this.bindDropzone();
        this.bindCommitAction();
    }

    bindDrawerTriggers() {
        this.openBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                // If on /enroll standalone page, don't prevent navigation unless it has data-inline="true"
                if (btn.getAttribute('data-inline') === 'true' || this.drawerEl) {
                    e.preventDefault();
                    this.open();
                }
            });
        });

        this.closeBtns.forEach(btn => {
            btn.addEventListener('click', () => this.close());
        });

        if (this.backdropEl) {
            this.backdropEl.addEventListener('click', () => this.close());
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen()) {
                this.close();
            }
        });
    }

    open() {
        if (!this.drawerEl) return;
        this.drawerEl.classList.add('active');
        if (this.backdropEl) this.backdropEl.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    close() {
        if (!this.drawerEl) return;
        this.drawerEl.classList.remove('active');
        if (this.backdropEl) this.backdropEl.classList.remove('active');
        document.body.style.overflow = '';
    }

    isOpen() {
        return this.drawerEl && this.drawerEl.classList.contains('active');
    }

    bindDropzone() {
        if (!this.dropzone) return;

        this.dropzone.addEventListener('click', () => {
            if (this.fileInput) this.fileInput.click();
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropzone.classList.remove('dragover');
            });
        });

        this.dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                this.processFiles(dt.files);
            }
        });

        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.processFiles(e.target.files);
                }
            });
        }

        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => {
                this.uploadedPreviews = [];
                this.renderPreviews();
            });
        }
    }

    async processFiles(files) {
        const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (imageFiles.length === 0) {
            this.showToast('Please select valid image files (JPG, PNG, WEBP).', 'error');
            return;
        }

        if (this.statusIndicator) {
            this.statusIndicator.style.display = 'block';
            this.statusIndicator.textContent = `Analyzing ${imageFiles.length} images with YuNet 5-point alignment...`;
        }

        const b64List = [];
        for (const file of imageFiles) {
            try {
                const b64 = await this.readFileAsDataURL(file);
                b64List.push(b64);
            } catch (_) {}
        }

        try {
            const res = await fetch('/api/enroll/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images: b64List })
            });

            const data = await res.json();
            if (res.ok && data.previews && data.previews.length > 0) {
                data.previews.forEach(p => {
                    this.uploadedPreviews.push({
                        crop_id: p.crop_id,
                        image_b64: p.image_b64,
                        sharpness_score: p.sharpness_score,
                        is_recommended: p.is_recommended,
                        status_label: p.status_label,
                        excluded: false
                    });
                });
                this.renderPreviews();
                this.showToast(`Extracted & aligned ${data.previews.length} face templates.`, 'info');
            } else {
                this.showToast(data.message || 'No detectable faces found in uploaded images.', 'error');
            }
        } catch (err) {
            this.showToast(`Error analyzing images: ${err.message}`, 'error');
        } finally {
            if (this.statusIndicator) this.statusIndicator.style.display = 'none';
        }
    }

    readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    renderPreviews() {
        if (!this.curationGrid) return;

        if (this.uploadedPreviews.length === 0) {
            if (this.curationTray) this.curationTray.style.display = 'none';
            if (this.commitBtn) this.commitBtn.disabled = true;
            return;
        }

        if (this.curationTray) this.curationTray.style.display = 'flex';
        this.curationGrid.innerHTML = this.uploadedPreviews.map((p, idx) => `
            <div class="curation-card ${p.excluded ? 'excluded' : 'approved'}" data-index="${idx}" title="Click to ${p.excluded ? 'include' : 'exclude'}">
                <img src="${p.image_b64}" alt="Face Crop">
                <div class="curation-sharpness-badge">${p.sharpness_score} Q</div>
            </div>
        `).join('');

        this.curationGrid.querySelectorAll('.curation-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const idx = parseInt(card.getAttribute('data-index'), 10);
                if (!isNaN(idx) && this.uploadedPreviews[idx]) {
                    this.uploadedPreviews[idx].excluded = !this.uploadedPreviews[idx].excluded;
                    this.renderPreviews();
                }
            });
        });

        const activeCount = this.uploadedPreviews.filter(p => !p.excluded).length;
        if (this.commitBtn) {
            this.commitBtn.disabled = activeCount === 0;
            this.commitBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
                Commit Enrollment (${activeCount} Templates)
            `;
        }
    }

    bindCommitAction() {
        if (!this.commitBtn) return;

        this.commitBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const nameInput = document.getElementById('drawer-full-name');
            const name = nameInput ? nameInput.value.trim() : '';

            if (!name) {
                this.showToast('Please enter the subject full name.', 'error');
                if (nameInput) nameInput.focus();
                return;
            }

            const approvedCrops = this.uploadedPreviews.filter(p => !p.excluded).map(p => p.image_b64);
            if (approvedCrops.length === 0) {
                this.showToast('Please keep at least one approved face crop.', 'error');
                return;
            }

            this.commitBtn.disabled = true;
            this.commitBtn.textContent = 'Extracting 512D ArcFace Vectors...';

            const payload = {
                name: name,
                occupation: (document.getElementById('drawer-occupation')?.value || 'SECURITY SPECIALIST').trim(),
                clearance: (document.getElementById('drawer-clearance')?.value || 'LEVEL 1 GRANTED').trim(),
                phone: (document.getElementById('drawer-phone')?.value || 'N/A').trim(),
                dob: (document.getElementById('drawer-dob')?.value || 'N/A').trim(),
                address: (document.getElementById('drawer-address')?.value || 'SYSTEM DATABASE').trim(),
                threat_level: (document.getElementById('drawer-threat')?.value || 'LOW').trim(),
                approved_crops: approvedCrops
            };

            try {
                const res = await fetch('/api/enroll/commit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await res.json();
                if (res.ok && data.status === 'success') {
                    this.showToast(`Enrolled '${name}' with ${data.exemplar_count} templates! Hot-reloaded live index.`, 'info');
                    
                    // Update global profile count badge if present on dashboard
                    const profilesEl = document.getElementById('stat-profiles');
                    if (profilesEl && data.total_profiles) {
                        profilesEl.textContent = data.total_profiles;
                    }

                    // Reset form after 1.5s
                    setTimeout(() => {
                        this.uploadedPreviews = [];
                        this.renderPreviews();
                        if (nameInput) nameInput.value = '';
                        this.close();
                    }, 1400);
                } else {
                    this.showToast(data.message || 'Error committing enrollment to database.', 'error');
                    this.commitBtn.disabled = false;
                }
            } catch (err) {
                this.showToast(`Enrollment failed: ${err.message}`, 'error');
                this.commitBtn.disabled = false;
            }
        });
    }

    showToast(message, type = 'info') {
        if (!this.toastMsg) {
            alert(message);
            return;
        }
        this.toastMsg.textContent = message;
        this.toastMsg.style.display = 'block';
        this.toastMsg.style.color = type === 'error' ? '#f43f5e' : '#10b981';
        this.toastMsg.style.border = type === 'error' ? '1px solid rgba(244,63,94,0.3)' : '1px solid rgba(16,185,129,0.3)';
        this.toastMsg.style.background = type === 'error' ? 'rgba(244,63,94,0.1)' : 'rgba(16,185,129,0.1)';
        this.toastMsg.style.padding = '10px 14px';
        this.toastMsg.style.borderRadius = '6px';
        this.toastMsg.style.fontSize = '0.78rem';
        this.toastMsg.style.fontFamily = 'var(--font-mono)';

        setTimeout(() => {
            if (this.toastMsg) this.toastMsg.style.display = 'none';
        }, 5000);
    }
}

window.EnrollmentDrawer = EnrollmentDrawer;
