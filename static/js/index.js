(function () {
    const recordBtn      = document.getElementById('record-btn');
    const recordLabel    = document.getElementById('record-label');
    const soundwave      = document.getElementById('soundwave');
    const uploadBtn      = document.getElementById('upload-btn');
    const fileInput      = document.getElementById('file-input');
    const toast          = document.getElementById('toast');
    const progressOverlay = document.getElementById('progress-overlay');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressStatus  = document.getElementById('progress-status');
    const dayInfoCard     = document.getElementById('day-info-card');

    let mediaRecorder  = null;
    let recordedChunks = [];
    let recording      = false;
    let toastTimer     = null;
    let analyzing      = false;

    /* ── German labels for internal status strings ───────────── */
    const STATUS_LABELS = {
        'Vorbereitung...':               'Wird vorbereitet\u2026',
        'Hochgeladen. Analyse startet\u2026': 'Analyse startet\u2026',
        'Audio wird geladen...':         'Audio wird geladen\u2026',
        'Merkmale werden extrahiert...': 'Sprachmerkmale werden analysiert\u2026',
        'Baseline wird berechnet...':    'Pers\u00f6nliche Baseline wird berechnet\u2026',
        'Klassifizierung l\u00e4uft...': 'Stimmungsmuster werden erkannt\u2026',
        'Score wird berechnet...':       'Score wird berechnet\u2026',
        'Ergebnis wird gespeichert...':  'Ergebnis wird gespeichert\u2026',
        'Analyse abgeschlossen.':        'Analyse abgeschlossen.',
        'done':                          'Fertig!',
    };

    /* ── Toast helper ───────────────────────────────────────── */
    function showToast(msg, isError) {
        toast.textContent = msg;
        toast.className = 'toast show' + (isError ? ' toast-error' : '');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3500);
    }

    /* ── Progress UI helpers ────────────────────────────────── */
    function showProgress() {
        progressBarFill.style.width = '0%';
        progressOverlay.classList.remove('progress-overlay--hidden');
        soundwave.classList.add('soundwave--hidden');
        dayInfoCard.classList.add('day-info-card--hidden');
        analyzing = true;
        setActionButtons(false);
    }

    function hideProgress() {
        progressOverlay.classList.add('progress-overlay--hidden');
        soundwave.classList.remove('soundwave--hidden');
        dayInfoCard.classList.remove('day-info-card--hidden');
        analyzing = false;
        setActionButtons(true);
    }

    function setProgress(pct, rawStatus) {
        progressBarFill.style.width = Math.min(100, pct) + '%';
        progressStatus.textContent = STATUS_LABELS[rawStatus] || rawStatus;
    }

    function setActionButtons(enabled) {
        recordBtn.disabled  = !enabled;
        uploadBtn.disabled  = !enabled;
        recordBtn.classList.toggle('btn--disabled', !enabled);
        uploadBtn.classList.toggle('btn--disabled', !enabled);
    }

    /* ── Recording ──────────────────────────────────────────── */
    recordBtn.addEventListener('click', async () => {
        if (analyzing) return;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast('Mikrofonzugriff wird von diesem Browser nicht unterst\u00fctzt.', true);
            return;
        }
        if (typeof MediaRecorder === 'undefined') {
            showToast('Aufnahme wird von diesem Browser nicht unterst\u00fctzt.', true);
            return;
        }

        if (!recording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                recordedChunks = [];
                mediaRecorder = new MediaRecorder(stream);
                let start = 0;

                mediaRecorder.onstart  = () => { start = Date.now(); };
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
                };
                mediaRecorder.onstop = () => {
                    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
                    recordedChunks = [];
                    if (Date.now() - start < 4500) {
                        showToast('Bitte mindestens 5 Sekunden an Tonmaterial bereitstellen.', true);
                    } else {
                        sendToBackend(blob, 'aufnahme.webm');
                    }
                };

                mediaRecorder.start();
                recording = true;
                recordBtn.classList.add('recording');
                recordBtn.setAttribute('aria-label', 'Aufnahme stoppen');
                recordLabel.textContent = 'Stopp';
                soundwave.classList.remove('idle');
            } catch (err) {
                console.error('Mikrofonzugriff verweigert:', err);
                showToast('Mikrofonzugriff nicht m\u00f6glich. Bitte Einstellungen pr\u00fcfen.', true);
            }
        } else {
            if (mediaRecorder) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                mediaRecorder = null;
            }
            recording = false;
            recordBtn.classList.remove('recording');
            recordBtn.setAttribute('aria-label', 'Aufnahme starten');
            recordLabel.textContent = 'Aufnehmen';
            soundwave.classList.add('idle');
        }
    });

    /* ── Upload ─────────────────────────────────────────────── */
    uploadBtn.addEventListener('click', () => {
        if (!analyzing) fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            sendToBackend(file, file.name);
            fileInput.value = '';
        }
    });

    /* ── Send to backend + SSE progress ─────────────────────── */
    function sendToBackend(blob, filename) {
        showProgress();
        setProgress(2, 'Vorbereitung...');

        const formData = new FormData();
        formData.append('file', blob, filename);

        fetch('/analyze', { method: 'POST', body: formData })
            .then(r => {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(data => {
                if (!data.task_id) throw new Error('Kein Task erhalten');
                listenProgress(data.task_id);
            })
            .catch(err => {
                console.error(err);
                showToast('Upload fehlgeschlagen. Bitte erneut versuchen.', true);
                hideProgress();
            });
    }

    function listenProgress(taskId) {
        const es = new EventSource('/progress/' + taskId);

        es.onmessage = (event) => {
            let data;
            try { data = JSON.parse(event.data); }
            catch { return; }

            if (data.error) {
                es.close();
                showToast('Fehler: ' + data.error, true);
                hideProgress();
                return;
            }

            setProgress(data.progress || 0, data.status || '');

            if (data.status === 'done' && data.result_id) {
                es.close();
                window.location.href = '/result/' + data.result_id;
            } else if (data.status === 'error') {
                es.close();
                showToast('Analyse fehlgeschlagen: ' + (data.error || ''), true);
                hideProgress();
            }
        };

        es.onerror = () => {
            es.close();
            showToast('Verbindung unterbrochen. Bitte erneut versuchen.', true);
            hideProgress();
        };
    }
})();
