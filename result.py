from datetime import datetime, timezone

class Result:
    def __init__(self, row):
        if row == None:
            self.created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            self.score = self.speech_rate = self.pause_rate = self.mean_pause_duration = self.f0_mean = self.f0_range = self.rms_energy = self.jitter = self.shimmer = self.hnr = self.f1_mean = self.f2_mean = self.mfcc_var = 0
        else:
            self.created_at = datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).astimezone(tz=None)
            self.score = row['score']
            self.id = row['id']
            self.speech_rate = row['speech_rate']
            self.pause_rate = row['pause_rate']
            self.mean_pause_duration = row['mean_pause_duration']
            self.f0_mean = row['f0_mean']
            self.f0_range = row['f0_range']
            self.rms_energy = row['rms_energy']
            self.jitter = row['jitter']
            self.shimmer = row['shimmer']
            self.hnr = row['hnr']
            self.f1_mean = row['f1_mean']
            self.f2_mean = row['f2_mean']
            self.mfcc_var = row['mfcc_var']

    def get_formatted_date(self):
        return self.created_at.strftime('%d.%m.%Y %H:%M Uhr')
    
    def get_score_css_class(self):
        if self.score < 25: return 'score-neutral'
        if self.score < 50: return 'score-concerning'
        if self.score < 75: return 'score-warning'
        return 'score-danger'