from datetime import datetime, timezone

class Result:
    def __init__(self, row):
        self.created_at = datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).astimezone(tz=None)
        self.score = row['score']
        self.id = row['id']
    
    def get_formatted_date(self):
        return self.created_at.strftime('%d.%m.%Y %H:%M Uhr')