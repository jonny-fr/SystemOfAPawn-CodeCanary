import json
import os
import tempfile
import threading
import time
import uuid

from flask import (
    Flask, Response, abort, jsonify, redirect,
    render_template, request, url_for,
)

from database import (
    get_all_features_ordered,
    get_last_scored_result,
    get_next_day_number,
    get_result,
    get_results,
    init_db,
    save_result,
    get_adjacent_scores,
    delete_history,
    get_score_before_day,
    get_all_score_deltas,
)
from result import Result

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory task registry for async analysis
# task_id -> {progress, status, result_id, error}
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background analysis worker
# ---------------------------------------------------------------------------

def _run_analysis(task_id: str, audio_path: str, day_number: int):
    def progress(pct: int, msg: str):
        with _tasks_lock:
            _tasks[task_id]['progress'] = pct
            _tasks[task_id]['status'] = msg

    try:
        progress(5, 'Vorbereitung...')

        # Load all previous features from DB
        previous_features, previous_qualities = get_all_features_ordered()

        # Import here to avoid circular imports and heavy libs at startup
        from classifier.analyzer import analyze_single_day

        result_data = analyze_single_day(
            audio_path,
            day_number,
            previous_features,
            previous_qualities,
            progress_callback=progress,
        )

        progress(95, 'Ergebnis wird gespeichert...')

        res = Result(None)
        res.day_number = day_number
        res.is_baseline = result_data['is_baseline']
        res.score = 0 if result_data['score'] == None else result_data['score']
        res.state = result_data['state']
        res.confidence = result_data['confidence']
        res.dep_score = result_data['dep_score']
        res.man_score = result_data['man_score']
        res.quality = result_data['quality']

        feats = result_data['features']
        res.f0_mean          = feats.get('f0_mean')
        res.f0_std           = feats.get('f0_std')
        res.f0_range         = feats.get('f0_range')
        res.jitter_local     = feats.get('jitter_local')
        res.shimmer_local    = feats.get('shimmer_local')
        res.hnr              = feats.get('hnr')
        res.speech_rate      = feats.get('speech_rate')
        res.pause_ratio      = feats.get('pause_ratio')
        res.pause_mean_dur   = feats.get('pause_mean_dur')
        res.rms_energy       = feats.get('rms_energy')
        res.spectral_centroid = feats.get('spectral_centroid')
        res.mfcc_1           = feats.get('mfcc_1')
        res.mfcc_2           = feats.get('mfcc_2')
        res.mfcc_3           = feats.get('mfcc_3')
        res.mfcc_4           = feats.get('mfcc_4')

        result_id = save_result(res)

        with _tasks_lock:
            _tasks[task_id]['result_id'] = result_id
            _tasks[task_id]['progress'] = 100
            _tasks[task_id]['status'] = 'done'

    except Exception as exc:
        with _tasks_lock:
            _tasks[task_id]['status'] = 'error'
            _tasks[task_id]['error'] = str(exc)
        # Re-raise so it shows up in server logs
        raise

    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

init_db()


@app.route('/')
def index():
    day_number = get_next_day_number()
    last_result = get_last_scored_result()
    return render_template('index.html', day_number=day_number, last_result=last_result)


@app.route('/history/clear')
def clear_history():
    delete_history()
    return 1

@app.route('/history')
def history():
    results = get_results()
    prev_scores = get_all_score_deltas()
    return render_template('history.html', results=results, prev_scores=prev_scores)


@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files.get('file')
    if not audio_file:
        return jsonify({'error': 'Keine Datei empfangen.'}), 400

    # Determine day number now (before any DB write) so the worker uses it
    day_number = get_next_day_number()

    # Save upload to a temp file that the background thread will read
    suffix = os.path.splitext(audio_file.filename or '.webm')[1] or '.webm'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        audio_file.save(tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            'progress': 0,
            'status': 'Hochgeladen. Analyse startet…',
            'result_id': None,
            'error': None,
        }

    thread = threading.Thread(
        target=_run_analysis,
        args=(task_id, tmp_path, day_number),
        daemon=True,
    )
    thread.start()

    return jsonify({'task_id': task_id}), 202


@app.route('/progress/<task_id>')
def progress_stream(task_id):
    def generate():
        while True:
            with _tasks_lock:
                task = _tasks.get(task_id)

            if task is None:
                yield 'data: ' + json.dumps({'error': 'not_found'}) + '\n\n'
                return

            payload = json.dumps({
                'progress':  task['progress'],
                'status':    task['status'],
                'result_id': task['result_id'],
                'error':     task['error'],
            })
            yield 'data: ' + payload + '\n\n'

            if task['status'] in ('done', 'error'):
                return

            time.sleep(0.4)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/result/<int:result_id>')
def result(result_id):
    res = get_result(result_id)
    if res is None:
        abort(404)
    timeline = get_adjacent_scores(result_id, 7)
    currentIndex = -1
    i = 0
    for x in timeline:
        if x[1]:
            currentIndex = i
        i += 1
    prev_score = get_score_before_day(res.day_number)
    return render_template('result.html', result=res, timeline=[x[0] for x in timeline], current=currentIndex, prev_score=prev_score)


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
