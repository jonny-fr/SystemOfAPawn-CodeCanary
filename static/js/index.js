(function () {
    const recordBtn   = document.getElementById('record-btn');
    const recordLabel = document.getElementById('record-label');
    const soundwave   = document.getElementById('soundwave');
    const uploadBtn   = document.getElementById('upload-btn');
    const fileInput   = document.getElementById('file-input');
    const toast       = document.getElementById('toast');

    let mediaRecorder = null;
    let recordedChunks = [];
    let recording     = false;
    let toastTimer    = null;

    /* ── Toast helper ──────────────────────── */
    function showToast(msg, isError) {
      toast.textContent = msg;
      toast.className = 'toast show' + (isError ? ' toast-error' : '');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3500);
    }

    /* ── Recording ─────────────────────────── */
    recordBtn.addEventListener('click', async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showToast('Mikrofonzugriff wird von diesem Browser nicht unterstützt.', true);
        return;
      }
      if (typeof MediaRecorder === 'undefined') {
        showToast('Aufnahme wird von diesem Browser nicht unterstützt.', true);
        return;
      }

      if (!recording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          recordedChunks = [];
          mediaRecorder = new MediaRecorder(stream);
          let start = 0

          mediaRecorder.onstart = (e) => { start = Date.now(); }

          mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              recordedChunks.push(e.data);
            }
          };

          mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'audio/webm' });
            recordedChunks = [];
            // only upload recording if at least 4.5 seconds have been recorded
            if(Date.now() - start < 4500) {
                alert('Bitte mindestens 5 Sekunden an Tonmaterial bereitstellen.');
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
          showToast('Mikrofonzugriff nicht möglich. Bitte Einstellungen prüfen.', true);
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

    /* ── Upload ────────────────────────────── */
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (file) {
        sendToBackend(file, file.name);
      }
    });

    /* ── Send to backend ───────────────────── */
    function sendToBackend(blob, filename) {
      showToast('Wird analysiert…', false);
      const formData = new FormData();
      formData.append('file', blob, filename);
      fetch('/analyze', { method: 'POST', body: formData })
        .then(response => {
          if (response.ok) {
            window.location.href = response.url || '/result';
          } else {
            showToast('Analyse fehlgeschlagen. Bitte erneut versuchen.', true);
          }
        })
        .catch(() => showToast('Verbindungsfehler. Bitte erneut versuchen.', true));
    }
  })();