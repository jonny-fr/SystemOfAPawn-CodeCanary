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
                created_at TEXT    NOT NULL,
                speech_rate     DECIMAL(10,5) NOT NULL,
                pause_rate      DECIMAL(10,5) NOT NULL,
                mean_pause_duration     DECIMAL(10,5) NOT NULL,
                f0_mean         DECIMAL(10,5) NOT NULL,
                f0_range        DECIMAL(10,5) NOT NULL,
                rms_energy      DECIMAL(10,5) NOT NULL,
                jitter          DECIMAL(10,5) NOT NULL,
                shimmer         DECIMAL(10,5) NOT NULL,
                hnr             DECIMAL(10,5) NOT NULL,
                f1_mean         DECIMAL(10,5) NOT NULL,
                f2_mean         DECIMAL(10,5) NOT NULL,
                mfcc_var        DECIMAL(10,5) NOT NULL
            )'''
        )
        conn.commit()


def save_result(res: Result):
    """Persist a classifier score and return the new row id."""
    with get_db() as conn:
        sql = 'INSERT INTO results (score, created_at, speech_rate, pause_rate, mean_pause_duration, f0_mean, f0_range, rms_energy, jitter, shimmer, hnr, f1_mean, f2_mean, mfcc_var) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        cursor = conn.execute(
            sql,
            (res.score, res.created_at, res.speech_rate, res.pause_rate, res.mean_pause_duration, res.f0_mean, res.f0_range, res.rms_energy, res.jitter, res.shimmer, res.hnr, res.f1_mean, res.f2_mean, res.mfcc_var)
        )
        conn.commit()
        return cursor.lastrowid
    
def get_results():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM results ORDER BY id DESC'
        ).fetchall()
    return [Result(r) for r in rows]

def get_result(result_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM results WHERE id = ?',
            (result_id,),
        ).fetchone()
    return Result(row)

def get_adjacent_scores(result_id: int, range: int) -> list[int]:
    with get_db() as conn:
        # TODO: more robust implementation that doesn't rely on consecutive IDs
        rows = conn.execute('SELECT id, score FROM results WHERE id >= ? AND id <= ? ORDER BY id ASC').fetchall()
        return [(r['score'], r['id'] == result_id) for r in rows]