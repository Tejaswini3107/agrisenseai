from datetime import datetime
from extensions import db

class Weather(db.Model):
    __tablename__ = 'weather'
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(128), nullable=False)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Integer, nullable=True)
    description = db.Column(db.String(256), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'city': self.city,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }
