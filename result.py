from datetime import datetime, timezone


class Result:
    def __init__(self, row):
        if row is None:
            self.id = None
            self.day_number = 0
            self.created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            self.is_baseline = False
            self.score = None
            self.state = None
            self.confidence = None
            self.dep_score = None
            self.man_score = None
            self.quality = None
            # 15 acoustic features
            self.f0_mean = self.f0_std = self.f0_range = None
            self.jitter_local = self.shimmer_local = self.hnr = None
            self.speech_rate = self.pause_ratio = self.pause_mean_dur = None
            self.rms_energy = self.spectral_centroid = None
            self.mfcc_1 = self.mfcc_2 = self.mfcc_3 = self.mfcc_4 = None
        else:
            self.id = row['id']
            self.day_number = row['day_number']
            self.created_at = datetime.strptime(
                row['created_at'], '%Y-%m-%d %H:%M:%S'
            ).replace(tzinfo=timezone.utc).astimezone(tz=None)
            self.is_baseline = bool(row['is_baseline'])
            self.score = row['score']
            self.state = row['state']
            self.confidence = row['confidence']
            self.dep_score = row['dep_score']
            self.man_score = row['man_score']
            self.quality = row['quality']
            self.f0_mean = row['f0_mean']
            self.f0_std = row['f0_std']
            self.f0_range = row['f0_range']
            self.jitter_local = row['jitter_local']
            self.shimmer_local = row['shimmer_local']
            self.hnr = row['hnr']
            self.speech_rate = row['speech_rate']
            self.pause_ratio = row['pause_ratio']
            self.pause_mean_dur = row['pause_mean_dur']
            self.rms_energy = row['rms_energy']
            self.spectral_centroid = row['spectral_centroid']
            self.mfcc_1 = row['mfcc_1']
            self.mfcc_2 = row['mfcc_2']
            self.mfcc_3 = row['mfcc_3']
            self.mfcc_4 = row['mfcc_4']

    def get_formatted_date(self):
        if isinstance(self.created_at, str):
            dt = datetime.strptime(self.created_at, '%Y-%m-%d %H:%M:%S')
        else:
            dt = self.created_at
        return dt.strftime('%d.%m.%Y %H:%M Uhr')

    def get_score_css_class(self):
        if self.score is None or self.is_baseline:
            return 'score-baseline'
        s = self.score
        if s < -60:  return 'score-danger-dep'
        if s < -25:  return 'score-warning-dep'
        if s < -8:   return 'score-concerning-dep'
        if s <= 8:   return 'score-neutral'
        if s <= 25:  return 'score-concerning-man'
        if s <= 60:  return 'score-warning-man'
        return 'score-danger-man'

    def get_state_label(self):
        _MAP = {
            'normal':            'Normal / Stabil',
            'depression-onset':  'Depressions-Beginn',
            'depression-like':   'Depressionsartig',
            'mania-onset':       'Manie-Beginn',
            'mania-like':        'Manierartig',
            'unclear':           'Unklare Veränderung',
            'reject':            'Analyse nicht möglich',
            'stable':            'Normal / Stabil',
        }
        return _MAP.get(self.state or '', self.state or '–')

    def get_quality_label(self):
        return {
            'clean':    'Gut',
            'degraded': 'Beeinträchtigt',
            'reject':   'Abgelehnt',
        }.get(self.quality or '', self.quality or '–')
