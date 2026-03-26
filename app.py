from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


def classify_file(file):  # noqa: ARG001  # file is unused in this dummy implementation
    """Dummy classifier function. Returns a hardcoded quality score."""
    score = 67
    return score


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(url_for('index'))
    score = classify_file(file)
    return redirect(url_for('result', score=score, filename=file.filename))


@app.route('/result')
def result():
    score = int(request.args.get('score', 0))
    filename = request.args.get('filename', '')
    return render_template('result.html', score=score, filename=filename)


if __name__ == '__main__':
    app.run()
