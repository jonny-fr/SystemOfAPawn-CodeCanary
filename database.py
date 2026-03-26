from datetime import datetime, timezone
from result import Result
import sqlite3

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
    
def get_results():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT id, score, created_at FROM results ORDER BY id DESC'
        ).fetchall()
    return [Result(r) for r in rows]

def get_result(result_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT id, score, created_at FROM results WHERE id = ?',
            (result_id,),
        ).fetchone()
    return row