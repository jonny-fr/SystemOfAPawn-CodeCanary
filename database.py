from datetime import datetime, timezone
from result import Result
import sqlite3

DATABASE = 'results.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_db():
    """Drop and recreate the table if it uses the old schema (no day_number column)."""
    with get_db() as conn:
        cols = {row['name'] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
        if cols and 'day_number' not in cols:
            conn.execute("DROP TABLE results")
            conn.commit()


def init_db():
    _migrate_db()
    with get_db() as conn:
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                day_number        INTEGER NOT NULL,
                created_at        TEXT    NOT NULL,
                is_baseline       INTEGER NOT NULL DEFAULT 0,
                score             REAL,
                state             TEXT,
                confidence        REAL,
                dep_score         REAL,
                man_score         REAL,
                quality           TEXT,
                f0_mean           REAL,
                f0_std            REAL,
                f0_range          REAL,
                jitter_local      REAL,
                shimmer_local     REAL,
                hnr               REAL,
                speech_rate       REAL,
                pause_ratio       REAL,
                pause_mean_dur    REAL,
                rms_energy        REAL,
                spectral_centroid REAL,
                mfcc_1            REAL,
                mfcc_2            REAL,
                mfcc_3            REAL,
                mfcc_4            REAL
            )'''
        )
        conn.commit()


def get_next_day_number() -> int:
    """Returns 1-based day number for the next recording."""
    with get_db() as conn:
        row = conn.execute('SELECT COUNT(*) FROM results').fetchone()
    return (row[0] or 0) + 1


def get_all_features_ordered() -> tuple[list, list]:
    """
    Returns (features_list, qualities_list) ordered by day_number.
    Each features_list entry is a dict with all 15 feature keys + 'reliability',
    or None if the recording was rejected.
    """
    FEATURE_NAMES = [
        'f0_mean', 'f0_std', 'f0_range',
        'jitter_local', 'shimmer_local', 'hnr',
        'speech_rate', 'pause_ratio', 'pause_mean_dur',
        'rms_energy', 'spectral_centroid',
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
    ]
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM results ORDER BY day_number ASC'
        ).fetchall()

    features_list = []
    qualities_list = []
    for row in rows:
        quality = row['quality'] or 'degraded'
        qualities_list.append(quality)
        if quality == 'reject':
            features_list.append(None)
        else:
            feat = {f: row[f] for f in FEATURE_NAMES}
            # reliability: stored implicitly from quality
            feat['reliability'] = 1.0 if quality == 'clean' else 0.5
            features_list.append(feat)

    return features_list, qualities_list


def get_last_scored_result():
    """Returns the most recent Result that has a non-NULL score, or None."""
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM results WHERE score IS NOT NULL ORDER BY day_number DESC LIMIT 1'
        ).fetchone()
    if row is None:
        return None
    return Result(row)


def save_result(res: 'Result') -> int:
    """Persist a result and return the new row id."""
    with get_db() as conn:
        cursor = conn.execute(
            '''INSERT INTO results (
                day_number, created_at, is_baseline,
                score, state, confidence, dep_score, man_score, quality,
                f0_mean, f0_std, f0_range,
                jitter_local, shimmer_local, hnr,
                speech_rate, pause_ratio, pause_mean_dur,
                rms_energy, spectral_centroid,
                mfcc_1, mfcc_2, mfcc_3, mfcc_4
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?
            )''',
            (
                res.day_number,
                res.created_at if isinstance(res.created_at, str)
                    else res.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                1 if res.is_baseline else 0,
                res.score, res.state, res.confidence,
                res.dep_score, res.man_score, res.quality,
                res.f0_mean, res.f0_std, res.f0_range,
                res.jitter_local, res.shimmer_local, res.hnr,
                res.speech_rate, res.pause_ratio, res.pause_mean_dur,
                res.rms_energy, res.spectral_centroid,
                res.mfcc_1, res.mfcc_2, res.mfcc_3, res.mfcc_4,
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_results() -> list:
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM results ORDER BY day_number DESC'
        ).fetchall()
    return [Result(row) for row in rows]


def get_result(result_id: int):
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM results WHERE id = ?',
            (result_id,),
        ).fetchone()
    if row is None:
        return None
    return Result(row)


def get_adjacent_scores(result_id: int, range: int) -> list[int]:
    with get_db() as conn:
        # TODO: more robust implementation that doesn't rely on consecutive IDs
        rows = conn.execute('SELECT id, score FROM results WHERE id >= ? AND id <= ? ORDER BY id ASC',
                            (result_id - range, result_id + range)).fetchall()
        return [(r['score'], r['id'] == result_id) for r in rows]