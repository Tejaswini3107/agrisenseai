from flask import Blueprint, request, jsonify
from extensions import db
from models import Weather
from services.weather_service import fetch_external_weather

def register_routes(app):
    bp = Blueprint('api', __name__)

    @bp.route('/weather', methods=['GET'])
    def list_weather():
        items = Weather.query.order_by(Weather.timestamp.desc()).all()
        return jsonify([w.to_dict() for w in items])

    @bp.route('/weather', methods=['POST'])
    def create_weather():
        data = request.get_json() or {}
        city = data.get('city')
        if not city:
            return jsonify({'error': 'city required'}), 400
        w = Weather(
            city=city,
            temperature=data.get('temperature'),
            humidity=data.get('humidity'),
            description=data.get('description')
        )
        db.session.add(w)
        db.session.commit()
        return jsonify({'id': w.id}), 201

    @bp.route('/weather/external', methods=['GET'])
    def external():
        city = request.args.get('city')
        if not city:
            return jsonify({'error': 'city query required'}), 400
        result = fetch_external_weather(city)
        return jsonify(result)

    app.register_blueprint(bp)
