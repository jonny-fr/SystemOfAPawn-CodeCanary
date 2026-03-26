from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


def analyze_audio(_file):
    """Dummy classifier – always returns a fixed score for demonstration."""
    score = 67
    return score


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files.get('file')
    if not audio_file:
        return redirect(url_for('index'))
    score = analyze_audio(audio_file)
    return redirect(url_for('result', score=score))


@app.route('/result')
def result():
    score = request.args.get('score', 0, type=int)
    return render_template('result.html', score=score)


if __name__ == '__main__':
    app.run()
