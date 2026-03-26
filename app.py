import sqlite3
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for, abort

app = Flask(__name__)

DATABASE = 'results.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                score      INTEGER NOT NULL,
                created_at TEXT    NOT NULL
            )'''
        )
        conn.commit()


def save_result(score):
    """Persist a classifier score and return the new row id."""
    created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as conn:
        cursor = conn.execute(
            'INSERT INTO results (score, created_at) VALUES (?, ?)',
            (score, created_at),
        )
        conn.commit()
        return cursor.lastrowid


def analyze_audio(_file):
    """Dummy classifier – always returns a fixed score for demonstration."""
    score = 67
    result_id = save_result(score)
    return result_id


init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT id, score, created_at FROM results ORDER BY id DESC'
        ).fetchall()
    return render_template('history.html', results=rows)


@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files.get('file')
    if not audio_file:
        return redirect(url_for('index'))
    result_id = analyze_audio(audio_file)
    return redirect(url_for('result', result_id=result_id))


@app.route('/result/<int:result_id>')
def result(result_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT id, score, created_at FROM results WHERE id = ?',
            (result_id,),
        ).fetchone()
    if row is None:
        abort(404)
    return render_template('result.html', result=row)


if __name__ == '__main__':
    app.run()
