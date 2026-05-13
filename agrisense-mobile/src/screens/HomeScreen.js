import React, {useEffect, useState} from 'react';
import {ActivityIndicator, SafeAreaView, ScrollView, StyleSheet, Text, View} from 'react-native';
import {getDashboardWeather} from '../services/api';
import WeatherForecastCard from '../components/WeatherForecastCard';

const APP_TITLE = 'AgriSense AI';

const HomeScreen = () => {
  const [dashboardWeather, setDashboardWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadForecast = async () => {
      try {
        const weather = await getDashboardWeather();
        setDashboardWeather(weather);
      } catch (error) {
        console.error('Unable to load weather forecast for dashboard', error);
        setError('Unable to load live backend data');
      } finally {
        setLoading(false);
      }
    };

    loadForecast();
  }, []);

  const todayWeather = dashboardWeather?.today ?? {};
  const forecastList = Array.isArray(dashboardWeather?.forecast) ? dashboardWeather.forecast : [];
  const weatherSummary = `${todayWeather.temperatureC ?? '—'}°C ${todayWeather.condition ?? '—'}`;
  const weatherMeta = `H ${todayWeather.humidityPct ?? '—'}%  R ${todayWeather.rainfallMm ?? '—'} mm`;
  const currentCrop = dashboardWeather?.recommended_crop ?? '—';
  const diseaseRisk = dashboardWeather?.disease_risk ?? '—';
  const irrigationNeed = dashboardWeather?.irrigation_need_litres ?? '—';
  const climateRisk = dashboardWeather?.climate_risk ?? '—';

  if (loading) {
    return (
      <View style={styles.loaderWrap}>
        <ActivityIndicator size="large" color="#0A7A4D" />
        <Text style={styles.loaderText}>Loading...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.screen}>
        <View style={styles.errorWrap}>
          <Text style={styles.errorTitle}>Live data unavailable</Text>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.headerBar}>
        <View style={styles.brandRow}>
          <View style={styles.logoWrap}>
            <Text style={styles.logoMark}>A</Text>
          </View>
          <Text style={styles.brandName}>{APP_TITLE}</Text>
        </View>

        <View style={styles.headerActions}>
          <Text style={styles.languagePill}>English</Text>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>AI</Text>
          </View>
        </View>
      </View>

      <Text style={styles.greeting}>Namaste!</Text>

      <View style={styles.tileGrid}>
        <View style={[styles.tileCard, styles.tilePest]}>
          <Text style={styles.newBadge}>{dashboardWeather?.disease_confidence ?? '—'}%</Text>
          <Text style={styles.tileTitle}>Pest Alerts</Text>
          <Text style={styles.tileSubtext}>{diseaseRisk}</Text>
        </View>

        <View style={[styles.tileCard, styles.tileIrrigation]}>
          <Text style={styles.tileTitle}>Smart Irrigation</Text>
          <Text style={styles.tileSubtext}>{irrigationNeed} L</Text>
          <Text style={styles.tileMetaText}>Need from backend</Text>
        </View>

        <View style={[styles.tileCard, styles.tileWeather]}>
          <Text style={styles.tileTitle}>Weather</Text>
          <Text style={styles.tileSubtext}>{weatherSummary}</Text>
          <Text style={styles.tileMetaText}>{weatherMeta}</Text>
          <Text style={styles.tileMetaText}>{dashboardWeather?.location_name ?? '—'}</Text>
          <Text style={styles.tileMetaText}>{currentCrop}</Text>
        </View>

        <View style={[styles.tileCard, styles.tileOffline]}>
          <Text style={styles.tileTitle}>Offline Dashboard</Text>
          <Text style={styles.tileSubtext}>{climateRisk}</Text>
        </View>
      </View>

      <View style={styles.sectionWrap}>
        <Text style={styles.sectionTitle}>Forecast</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.forecastRow}>
          {forecastList.map((forecast) => (
            <WeatherForecastCard key={`${forecast.day}-${forecast.temperatureC}`} forecast={forecast} />
          ))}
        </ScrollView>
      </View>

      <View style={styles.askButtonWrap}>
        <View style={styles.askButtonOuter}>
          <View style={styles.askButtonInner}>
            <Text style={styles.askButtonMic}>MIC</Text>
            <Text style={styles.askButtonText}>Tap to Ask</Text>
          </View>
        </View>
      </View>

      <View style={styles.bottomNav}>
        <View style={styles.navItem}>
          <Text style={[styles.navIcon, styles.navActive]}>HOME</Text>
        </View>
        <View style={styles.navItem}>
          <View style={styles.navBadge}>
            <Text style={styles.navBadgeText}>2</Text>
          </View>
          <Text style={styles.navIcon}>ALERTS</Text>
        </View>
        <View style={styles.navItem}>
          <Text style={styles.navIcon}>PROFILE</Text>
        </View>
        <View style={styles.navItem}>
          <Text style={styles.navIcon}>SETTINGS</Text>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#045E3A',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 6,
  },
  loaderWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#045E3A',
    paddingHorizontal: 20,
  },
  loaderText: {
    marginTop: 14,
    fontSize: 14,
    color: '#E7F7ED',
  },
  errorWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
    backgroundColor: '#045E3A',
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#E7F7ED',
    textAlign: 'center',
  },
  headerBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 6,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#1A7A55',
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  logoWrap: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#0ABF71',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoMark: {
    fontSize: 16,
    fontWeight: '700',
    color: '#053923',
  },
  brandName: {
    fontSize: 28,
    fontWeight: '700',
    color: '#EAFBF2',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  languagePill: {
    backgroundColor: '#0B8A55',
    color: '#D9FCE9',
    borderRadius: 999,
    fontSize: 13,
    fontWeight: '600',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#16CF79',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 14,
  },
  greeting: {
    marginTop: 12,
    fontSize: 34,
    color: '#E9FAF0',
    fontWeight: '700',
  },
  tileGrid: {
    marginTop: 20,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    rowGap: 12,
  },
  tileCard: {
    width: '48.3%',
    minHeight: 156,
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'flex-end',
  },
  tilePest: {
    backgroundColor: '#F0451F',
  },
  tileIrrigation: {
    backgroundColor: '#1897E8',
  },
  tileWeather: {
    backgroundColor: '#F3A311',
  },
  tileOffline: {
    backgroundColor: '#7C3AED',
  },
  newBadge: {
    position: 'absolute',
    right: 8,
    top: 8,
    backgroundColor: '#FFFFFF',
    color: '#D11F00',
    borderRadius: 999,
    fontSize: 11,
    fontWeight: '600',
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  tileTitle: {
    fontSize: 28,
    lineHeight: 32,
    color: '#FFFFFF',
    fontWeight: '700',
  },
  tileSubtext: {
    marginTop: 4,
    fontSize: 17,
    color: '#F6FCFF',
    fontWeight: '500',
  },
  tileMetaText: {
    marginTop: 4,
    fontSize: 12,
    color: '#F5FAFF',
    fontWeight: '600',
  },
  sectionWrap: {
    marginTop: 18,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#E9FAF0',
    marginBottom: 10,
  },
  forecastRow: {
    paddingRight: 12,
  },
  askButtonWrap: {
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 4,
  },
  askButtonOuter: {
    width: 95,
    height: 95,
    borderRadius: 47.5,
    backgroundColor: '#E9FDF1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  askButtonInner: {
    width: 87,
    height: 87,
    borderRadius: 43.5,
    borderWidth: 2,
    borderColor: '#14D17C',
    backgroundColor: '#0EC26F',
    alignItems: 'center',
    justifyContent: 'center',
  },
  askButtonMic: {
    fontSize: 12,
    fontWeight: '700',
    color: '#E6FFF3',
  },
  askButtonText: {
    marginTop: 4,
    fontSize: 14,
    color: '#F2FFF8',
    fontWeight: '600',
  },
  bottomNav: {
    height: 68,
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  navItem: {
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 70,
  },
  navIcon: {
    fontSize: 11,
    color: '#97A2A9',
    fontWeight: '700',
  },
  navActive: {
    color: '#0DAC62',
  },
  navBadge: {
    position: 'absolute',
    top: -8,
    right: 14,
    backgroundColor: '#10C86F',
    borderRadius: 999,
    minWidth: 18,
    height: 18,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  navBadgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '700',
  },
});

export default HomeScreen;
