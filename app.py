from flask import Flask, render_template, request, redirect, url_for, abort
from database import get_result, get_results, init_db, save_result
from result import Result

app = Flask(__name__)



def analyze_audio(_file):
    """Dummy classifier – always returns a fixed score for demonstration."""
    res = Result(None)
    res.score = 67
    res.f0_mean = 1
    res.f0_range = 1
    res.f1_mean = 1
    res.f2_mean = 1
    res.hnr = 1
    res.jitter = 1
    res.mean_pause_duration = 1
    res.mfcc_var = 1
    res.shimmer = 1
    res.speech_rate = 1
    res.pause_rate = 1
    res.mean_pause_duration = 1
    res.rms_energy = 1
    result_id = save_result(res)
    return result_id

init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    return render_template('history.html', results=get_results())


@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files.get('file')
    if not audio_file:
        return redirect(url_for('index'))
    result_id = analyze_audio(audio_file)
    return redirect(url_for('result', result_id=result_id))


@app.route('/result/<int:result_id>')
def result(result_id):
    result = get_result(result_id)
    if result is None:
        abort(404)
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run()
