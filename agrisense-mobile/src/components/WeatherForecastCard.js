import React from 'react';
import {Text, View, StyleSheet} from 'react-native';

const WeatherForecastCard = ({forecast}) => {
  const isRainDay = (forecast?.rainfallMm ?? 0) > 0 || /rain/i.test(forecast?.condition ?? '');
  const temperature = forecast?.temperatureC ?? '—';
  const humidity = forecast?.humidityPct ?? '—';
  const rainfall = forecast?.rainfallMm ?? '—';

  return (
    <View style={styles.card}>
      <View style={styles.topRow}>
        <Text style={styles.day}>{forecast?.day ?? 'Day'}</Text>
        <Text style={[styles.badge, isRainDay ? styles.badgeRain : styles.badgeDry]}>
          {isRainDay ? 'Rain' : 'Dry'}
        </Text>
      </View>
      <Text style={styles.temp}>{temperature}°C</Text>
      <Text style={styles.condition}>{forecast?.condition ?? '—'}</Text>

      <View style={styles.metricRow}>
        <Text style={styles.metricLabel}>Humidity</Text>
        <Text style={styles.metricValue}>{humidity}%</Text>
      </View>
      <View style={styles.metricRow}>
        <Text style={styles.metricLabel}>Rain</Text>
        <Text style={styles.metricValue}>{rainfall} mm</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    width: 168,
    borderRadius: 16,
    backgroundColor: '#FFFFFF',
    padding: 14,
    marginRight: 12,
    borderWidth: 1,
    borderColor: '#DCEADF',
    shadowColor: '#00150B',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 3,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  day: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1F3A2E',
  },
  badge: {
    fontSize: 10,
    fontWeight: '700',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 999,
  },
  badgeRain: {
    backgroundColor: '#E6F1FF',
    color: '#2E5FA8',
  },
  badgeDry: {
    backgroundColor: '#E9F7EF',
    color: '#1F754A',
  },
  temp: {
    fontSize: 26,
    fontWeight: '700',
    color: '#0A7A4D',
    marginTop: 4,
  },
  condition: {
    fontSize: 13,
    color: '#4A6A5C',
    marginTop: 2,
    marginBottom: 12,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 6,
  },
  metricLabel: {
    fontSize: 12,
    color: '#5D7569',
  },
  metricValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1F3A2E',
  },
});

export default WeatherForecastCard;
