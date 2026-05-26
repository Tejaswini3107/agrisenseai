/**
 * SmartIrrigationScreen
 *
 * Farmer inputs (interactive):
 *   • Crop type      — synced from cropStore (tap to change in HomeScreen)
 *   • Growth stage   — pill selector: Seedling / Vegetative / Flowering / Maturity
 *   • Days since last irrigation — stepper (− / number / +)
 *   • Soil feel      — pill selector: Dry / Moist / Wet (maps to 20 / 50 / 80 %)
 *
 * Auto-fetched (no farmer input needed):
 *   • temperature, humidity, rainfall_mm, wind_speed  ← /api/weather/current
 *   • ET0  ← estimated from weather via estimateEt0 in api.js
 *   • Kc   ← looked up from KC_TABLE by crop + stage in api.js
 *
 * ML endpoints called:
 *   • POST /api/irrigation/predict  → { irrigate, water_amount_mm, etc_mm_per_day }
 */
import React, {useEffect, useRef, useState} from 'react';
import {
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {buildIrrigationPayloadFromInputs, getWeatherCurrent, getWeatherForecast, predictIrrigation} from '../services/api';
import cropStore from '../services/cropStore';
import {getCurrentPosition, requestLocationPermission} from '../services/location';
import locationStore from '../services/locationStore';
import useTranslation from '../services/useTranslation';

const DEFAULT_LAT = 18.6650;
const DEFAULT_LON = 77.9046;

const GROWTH_STAGES = ['seedling', 'vegetative', 'flowering', 'maturity'];
const SOIL_FEEL_OPTIONS = [
  {key: 'dry',   label: 'Dry',   emoji: '🌵', moisture: 20},
  {key: 'moist', label: 'Moist', emoji: '🌱', moisture: 50},
  {key: 'wet',   label: 'Wet',   emoji: '💧', moisture: 80},
];

const SmartIrrigationScreen = ({navigation}) => {
  // ── Farmer inputs ──────────────────────────────────────────
  const [selectedCrop, setSelectedCrop]           = useState('rice');
  const [growthStage, setGrowthStage]             = useState('vegetative');
  const [daysSinceIrrigation, setDaysSinceIrrigation] = useState(4);
  const [soilFeel, setSoilFeel]                   = useState('moist');

  // ── Data ───────────────────────────────────────────────────
  const [irrigationData, setIrrigationData]       = useState(null);
  const [loading, setLoading]                     = useState(true);
  const [refreshing, setRefreshing]               = useState(false);
  const [error, setError]                         = useState(null);
  const [coords, setCoords]                       = useState(null);

  const {t} = useTranslation();
  const coordsRef = useRef(coords);
  useEffect(() => { coordsRef.current = coords; }, [coords]);

  // ── Sync crop from store ────────────────────────────────────
  useEffect(() => {
    cropStore.getCrop().then((v) => { if (v) setSelectedCrop(v); });
    const unsub = cropStore.subscribe((v) => { if (v) setSelectedCrop(v); });
    return unsub;
  }, []);

  // ── Resolve location ────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const saved = await locationStore.getLocation();
        if (saved?.lat && saved?.lon) { setCoords(saved); return; }
        await requestLocationPermission();
        const pos = await getCurrentPosition({timeout: 8000, maximumAge: 60000});
        const loc = {lat: pos.coords.latitude, lon: pos.coords.longitude};
        await locationStore.setLocation(loc);
        setCoords(loc);
      } catch {
        setCoords({lat: DEFAULT_LAT, lon: DEFAULT_LON});
      }
    })();
  }, []);

  // ── Main fetch ──────────────────────────────────────────────
  const loadData = async (isRefresh = false) => {
    const loc = coordsRef.current ?? {lat: DEFAULT_LAT, lon: DEFAULT_LON};
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const {lat, lon} = loc;

      const [weather, forecastResp] = await Promise.all([
        getWeatherCurrent(lat, lon),
        getWeatherForecast(lat, lon),
      ]);

      const forecastList = forecastResp?.forecast ?? [];

      // Build and run today's prediction
      const payload = buildIrrigationPayloadFromInputs({
        weather,
        cropType: selectedCrop,
        growthStage,
        daysSinceIrrigation,
        soilFeel,
      });
      const result = await predictIrrigation(payload);

      const shouldWater    = (result.irrigate ?? '').toLowerCase() === 'yes';
      const waterAmountMm  = result.water_amount_mm ?? 0;
      const etcMmPerDay    = result.etc_mm_per_day ?? 0;

      // Build 3-day schedule from real forecast.
      // IMPORTANT: idx=0 is "Today" — we reuse the main card result so the schedule
      // is never contradictory with the decision card above it.
      // Future days (idx>0) infer soil feel from forecast rainfall.
      const scheduleItems = await Promise.all(
        forecastList.slice(0, 3).map(async (day, idx) => {
          const [y, m, d] = day.date.split('-').map(Number);
          const dateObj = new Date(y, m - 1, d);
          const dayLabel = idx === 0 ? 'Today' : idx === 1 ? 'Tomorrow'
            : dateObj.toLocaleDateString('en-US', {weekday: 'long'});
          const fullDate = dateObj.toLocaleDateString('en-US', {weekday: 'long', month: 'short', day: 'numeric'});

          // Today (idx=0): reuse main-card result — no separate ML call, no contradiction
          if (idx === 0) {
            return {
              label: dayLabel,
              fullDate,
              shouldWater,
              statusLabel: shouldWater ? 'Irrigate' : 'Skip',
              rainfall: Math.round(day.rainfall_mm ?? 0),
              waterMm: waterAmountMm,
            };
          }

          // Future days: infer soil feel from forecast rainfall
          const futureSoilFeel = day.rainfall_mm > 5 ? 'wet' : day.rainfall_mm > 1 ? 'moist' : 'dry';
          const dayPayload = buildIrrigationPayloadFromInputs({
            weather: {
              temperature: day.temperature,
              humidity: day.humidity,
              rainfall_mm: day.rainfall_mm,
              wind_speed: day.wind_speed,
            },
            cropType: selectedCrop,
            growthStage,
            daysSinceIrrigation: idx + daysSinceIrrigation,
            soilFeel: futureSoilFeel,
          });
          const dayResult   = await predictIrrigation(dayPayload);
          const dayIrrigate = (dayResult.irrigate ?? '').toLowerCase() === 'yes';

          return {
            label: dayLabel,
            fullDate,
            shouldWater: dayIrrigate,
            statusLabel: dayIrrigate ? 'Irrigate' : 'Skip',
            rainfall: Math.round(day.rainfall_mm ?? 0),
            waterMm: dayResult.water_amount_mm ?? 0,
          };
        }),
      );

      const temp        = weather.temperature != null ? parseFloat(weather.temperature.toFixed(1)) : 0;
      const humidity    = Math.round(weather.humidity ?? 0);
      const rainfall    = Math.round(weather.rainfall_mm ?? 0);
      const wind        = Math.round(weather.wind_speed ?? 0);
      const soilMoisture = Math.round(weather.soil_moisture ?? 0);

      // Determine next watering label.
      // When shouldWater=false, skip idx=0 (Today) to avoid contradiction
      // with the "Don't Water" decision card.
      const nextWateringItem = shouldWater
        ? null
        : scheduleItems.find((s, i) => i > 0 && s.shouldWater);
      const nextWatering = shouldWater
        ? 'Today 6–8 AM'
        : nextWateringItem
          ? `${nextWateringItem.label} 6–8 AM`
          : 'Not needed this week';

      setIrrigationData({
        shouldWater, waterAmountMm, etcMmPerDay,
        locationName: weather.location_name ?? 'Current location',
        nextWatering,
        conditionCards: [
          {iconName: 'thermometer',    iconColor: '#F97316', label: 'Temperature', value: `${temp}°C`,      sub: weather.condition ?? 'Live'},
          {iconName: 'water-percent',  iconColor: '#3B82F6', label: 'Humidity',    value: `${humidity}%`,   sub: 'Relative humidity'},
          {iconName: 'weather-rainy',  iconColor: '#60A5FA', label: 'Rainfall',    value: `${rainfall} mm`, sub: 'Today'},
          {iconName: 'weather-windy',  iconColor: '#A78BFA', label: 'Wind',        value: `${wind} km/h`,   sub: 'Current speed'},
        ],
        soilMoisture,
        insightRows: [
          {
            iconName: 'water', dot: '#22C55E',
            title: 'Water Requirement',
            body: `Crop water demand: ${etcMmPerDay.toFixed(1)} mm/day (ETc). ${shouldWater ? `Apply ${waterAmountMm} mm today.` : 'No deficit detected.'}`,
          },
          {
            iconName: 'grain', dot: '#3B82F6',
            title: 'Soil Condition',
            body: soilMoisture > 0
              ? `Soil moisture at ${soilMoisture}%. ${soilMoisture < 30 ? 'Below optimal — irrigation recommended.' : soilMoisture > 65 ? 'Well saturated — hold off watering.' : 'Adequate moisture levels.'}`
              : rainfall > 2 ? `${rainfall} mm of rain received — moisture likely adequate.` : 'No significant rain — monitor soil closely.',
          },
          {
            iconName: 'sprout', dot: '#F97316',
            title: 'ML Recommendation',
            body: shouldWater
              ? `Irrigate now. Apply ${waterAmountMm} mm. Best time: early morning (6–8 AM) to minimise evaporation.`
              : 'No irrigation needed right now. Recheck after 24 hours or if soil feels dry.',
          },
        ],
        schedule: scheduleItems,
        liveWeather: weather,
      });
    } catch (err) {
      console.error('SmartIrrigation load error', err);
      setError('Unable to load live irrigation advice');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Trigger load once coords are resolved
  useEffect(() => {
    if (coords !== null) loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coords]);

  // Re-fetch when any farmer input changes (but coords must be set)
  useEffect(() => {
    if (coords !== null && !loading) loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCrop, growthStage, daysSinceIrrigation, soilFeel]);

  // ── Loading ─────────────────────────────────────────────────
  if (loading) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: '#0D1B3E'}}>
        <StatusBar barStyle="light-content" backgroundColor="#0D1B3E" />
        <View style={{flex: 1, alignItems: 'center', justifyContent: 'center', gap: 14}}>
          <ActivityIndicator size="large" color="#22C55E" />
          <Text style={{fontSize: 15, fontWeight: '600', color: '#FFFFFF'}}>Loading irrigation data…</Text>
          <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.45)'}}>Fetching live weather + ML prediction</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: '#0D1B3E'}}>
        <StatusBar barStyle="light-content" backgroundColor="#0D1B3E" />
        <View style={{flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24, gap: 14}}>
          <MaterialCommunityIcons name="wifi-off" size={44} color="#EF4444" />
          <Text style={{fontSize: 16, fontWeight: '700', color: '#FFFFFF', textAlign: 'center'}}>{error}</Text>
          <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.5)', textAlign: 'center'}}>Check that the backend is running</Text>
          <TouchableOpacity
            onPress={() => loadData()}
            style={{marginTop: 6, backgroundColor: '#22C55E', borderRadius: 12, paddingHorizontal: 28, paddingVertical: 12}}>
            <Text style={{color: '#FFFFFF', fontWeight: '700'}}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const {shouldWater, nextWatering, conditionCards, insightRows, schedule, liveWeather, locationName, waterAmountMm, soilMoisture, etcMmPerDay} = irrigationData || {};

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#0D1B3E'}}>
      <StatusBar barStyle="light-content" backgroundColor="#0D1B3E" />

      {/* ── Header ── */}
      <View style={{flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 14, gap: 12}}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
          style={{width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.08)'}}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#FFFFFF" />
        </TouchableOpacity>
        <View style={{flex: 1}}>
          <Text style={{color: '#FFFFFF', fontSize: 18, fontWeight: '700'}}>Smart Irrigation</Text>
          <View style={{flexDirection: 'row', alignItems: 'center', marginTop: 2, gap: 4}}>
            <MaterialCommunityIcons name="map-marker" size={12} color="#6EE7B7" />
            <Text style={{color: 'rgba(255,255,255,0.5)', fontSize: 12}}>{locationName ?? 'Locating…'}</Text>
          </View>
        </View>
        <TouchableOpacity
          onPress={() => loadData(true)}
          style={{width: 38, height: 38, alignItems: 'center', justifyContent: 'center', borderRadius: 19, backgroundColor: 'rgba(34,197,94,0.15)'}}>
          {refreshing
            ? <ActivityIndicator size="small" color="#22C55E" />
            : <MaterialCommunityIcons name="refresh" size={20} color="#22C55E" />}
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 48}}>

        {/* ── Farmer Input Card ── */}
        <View style={{backgroundColor: '#132140', borderRadius: 20, padding: 18, borderWidth: 1, borderColor: 'rgba(34,197,94,0.2)', marginBottom: 16}}>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 16}}>
            <MaterialCommunityIcons name="tune-variant" size={18} color="#22C55E" />
            <Text style={{color: '#FFFFFF', fontSize: 15, fontWeight: '700'}}>Field Inputs</Text>
            <View style={{flex: 1}} />
            <View style={{backgroundColor: 'rgba(34,197,94,0.15)', borderRadius: 10, paddingHorizontal: 10, paddingVertical: 4}}>
              <Text style={{color: '#22C55E', fontSize: 12, fontWeight: '700', textTransform: 'capitalize'}}>{selectedCrop}</Text>
            </View>
          </View>

          {/* Growth Stage */}
          <Text style={{color: 'rgba(255,255,255,0.55)', fontSize: 12, fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.6}}>Growth Stage</Text>
          <View style={{flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 16}}>
            {GROWTH_STAGES.map((stage) => (
              <TouchableOpacity
                key={stage}
                onPress={() => setGrowthStage(stage)}
                style={{
                  paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
                  backgroundColor: growthStage === stage ? '#22C55E' : 'rgba(255,255,255,0.07)',
                  borderWidth: 1,
                  borderColor: growthStage === stage ? '#22C55E' : 'rgba(255,255,255,0.12)',
                }}>
                <Text style={{fontSize: 12, fontWeight: '600', textTransform: 'capitalize',
                  color: growthStage === stage ? '#FFFFFF' : 'rgba(255,255,255,0.6)'}}>
                  {stage}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Days Since Irrigation */}
          <Text style={{color: 'rgba(255,255,255,0.55)', fontSize: 12, fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.6}}>
            Days Since Last Irrigation
          </Text>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 16, marginBottom: 16}}>
            <TouchableOpacity
              onPress={() => setDaysSinceIrrigation((v) => Math.max(0, v - 1))}
              style={{width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.1)', alignItems: 'center', justifyContent: 'center'}}>
              <MaterialCommunityIcons name="minus" size={20} color="#FFFFFF" />
            </TouchableOpacity>
            <View style={{alignItems: 'center', minWidth: 60}}>
              <Text style={{fontSize: 28, fontWeight: '800', color: '#FFFFFF'}}>{daysSinceIrrigation}</Text>
              <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 2}}>days ago</Text>
            </View>
            <TouchableOpacity
              onPress={() => setDaysSinceIrrigation((v) => Math.min(30, v + 1))}
              style={{width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.1)', alignItems: 'center', justifyContent: 'center'}}>
              <MaterialCommunityIcons name="plus" size={20} color="#FFFFFF" />
            </TouchableOpacity>
            <View style={{flex: 1, height: 6, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.1)', overflow: 'hidden'}}>
              <View style={{width: `${Math.min((daysSinceIrrigation / 14) * 100, 100)}%`, height: '100%', borderRadius: 3,
                backgroundColor: daysSinceIrrigation >= 7 ? '#EF4444' : daysSinceIrrigation >= 4 ? '#F97316' : '#22C55E'}} />
            </View>
          </View>

          {/* Soil Feel */}
          <Text style={{color: 'rgba(255,255,255,0.55)', fontSize: 12, fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.6}}>
            How Does the Soil Feel?
          </Text>
          <View style={{flexDirection: 'row', gap: 10}}>
            {SOIL_FEEL_OPTIONS.map((opt) => (
              <TouchableOpacity
                key={opt.key}
                onPress={() => setSoilFeel(opt.key)}
                style={{
                  flex: 1, alignItems: 'center', paddingVertical: 12, borderRadius: 14,
                  backgroundColor: soilFeel === opt.key ? '#22C55E' : 'rgba(255,255,255,0.07)',
                  borderWidth: 1.5,
                  borderColor: soilFeel === opt.key ? '#22C55E' : 'rgba(255,255,255,0.1)',
                }}>
                <Text style={{fontSize: 20}}>{opt.emoji}</Text>
                <Text style={{fontSize: 12, fontWeight: '700', marginTop: 4,
                  color: soilFeel === opt.key ? '#FFFFFF' : 'rgba(255,255,255,0.55)'}}>
                  {opt.label}
                </Text>
                <Text style={{fontSize: 10, color: soilFeel === opt.key ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.3)', marginTop: 2}}>
                  {opt.moisture}%
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* ── Decision Card ── */}
        <View style={{borderRadius: 20, padding: 20, marginBottom: 16,
          backgroundColor: shouldWater ? '#15803D' : '#1E3A5F',
          borderWidth: 1.5,
          borderColor: shouldWater ? '#4ADE80' : '#3B82F6'}}>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 14}}>
            <View style={{height: 56, width: 56, borderRadius: 16, backgroundColor: 'rgba(255,255,255,0.15)', alignItems: 'center', justifyContent: 'center'}}>
              <MaterialCommunityIcons name={shouldWater ? 'water' : 'hand-back-right-off'} size={30} color="#FFFFFF" />
            </View>
            <View style={{flex: 1}}>
              <Text style={{color: '#FFFFFF', fontSize: 24, fontWeight: '800'}}>
                {shouldWater ? 'Water Now' : "Don't Water"}
              </Text>
              <Text style={{color: 'rgba(255,255,255,0.6)', fontSize: 13, marginTop: 2}}>
                {shouldWater ? 'पानी दें' : 'रुकें — मिट्टी में पर्याप्त नमी है'}
              </Text>
            </View>
          </View>

          <View style={{marginTop: 14, borderRadius: 12, backgroundColor: 'rgba(0,0,0,0.2)', padding: 14, gap: 6}}>
            <Text style={{color: 'rgba(255,255,255,0.8)', fontSize: 13, fontWeight: '700'}}>Why? (क्यों?)</Text>
            <Text style={{color: 'rgba(255,255,255,0.65)', fontSize: 13, lineHeight: 20}}>
              {shouldWater
                ? `ETc demand is ${etcMmPerDay?.toFixed(1)} mm/day. Soil feel: ${soilFeel}. Days since irrigation: ${daysSinceIrrigation}.`
                : `ETc demand is ${etcMmPerDay?.toFixed(1)} mm/day. Soil feel: ${soilFeel} — moisture adequate.`}
            </Text>
            {shouldWater && waterAmountMm > 0 ? (
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4}}>
                <MaterialCommunityIcons name="water-outline" size={14} color="#6EE7B7" />
                <Text style={{color: '#6EE7B7', fontSize: 13, fontWeight: '700'}}>Apply {waterAmountMm} mm of water</Text>
              </View>
            ) : null}
          </View>
        </View>

        {/* ── Next watering ── */}
        <View style={{borderRadius: 16, backgroundColor: '#0F3D2B', padding: 16, flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 16}}>
          <View style={{height: 48, width: 48, borderRadius: 12, backgroundColor: 'rgba(34,197,94,0.2)', alignItems: 'center', justifyContent: 'center'}}>
            <MaterialCommunityIcons name="clock-time-six-outline" size={26} color="#22C55E" />
          </View>
          <View>
            <Text style={{color: 'rgba(255,255,255,0.55)', fontSize: 11, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.5}}>Next Watering</Text>
            <Text style={{color: '#FFFFFF', fontSize: 20, fontWeight: '800', marginTop: 2}}>{nextWatering}</Text>
            <Text style={{color: 'rgba(255,255,255,0.45)', fontSize: 12, marginTop: 2}}>Best time: early morning</Text>
          </View>
        </View>

        {/* ── Condition cards 2×2 ── */}
        <Text style={{color: '#FFFFFF', fontSize: 15, fontWeight: '700', marginBottom: 12}}>Today's Conditions</Text>
        <View style={{flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginBottom: 16}}>
          {conditionCards.map((card) => (
            <View key={card.label} style={{width: '47.5%', borderRadius: 16, backgroundColor: '#1A2B50', padding: 16}}>
              <View style={{height: 36, width: 36, borderRadius: 10, backgroundColor: 'rgba(255,255,255,0.08)', alignItems: 'center', justifyContent: 'center', marginBottom: 10}}>
                <MaterialCommunityIcons name={card.iconName} size={20} color={card.iconColor} />
              </View>
              <Text style={{color: 'rgba(255,255,255,0.5)', fontSize: 11, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4}}>{card.label}</Text>
              <Text style={{color: '#FFFFFF', fontSize: 22, fontWeight: '700', marginTop: 4}}>{card.value}</Text>
              <Text style={{color: 'rgba(255,255,255,0.35)', fontSize: 11, marginTop: 2}}>{card.sub}</Text>
            </View>
          ))}
        </View>

        {/* ── Soil moisture bar ── */}
        {soilMoisture > 0 ? (
          <View style={{borderRadius: 16, backgroundColor: '#1A2B50', padding: 16, marginBottom: 16}}>
            <View style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10}}>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 8}}>
                <MaterialCommunityIcons name="layers" size={18} color="#A78BFA" />
                <Text style={{color: '#FFFFFF', fontSize: 14, fontWeight: '700'}}>Soil Moisture (NASA POWER)</Text>
              </View>
              <Text style={{color: '#A78BFA', fontSize: 16, fontWeight: '800'}}>{soilMoisture}%</Text>
            </View>
            <View style={{height: 8, borderRadius: 4, backgroundColor: 'rgba(255,255,255,0.1)'}}>
              <View style={{height: 8, borderRadius: 4, width: `${Math.min(soilMoisture, 100)}%`,
                backgroundColor: soilMoisture < 30 ? '#EF4444' : soilMoisture > 65 ? '#3B82F6' : '#22C55E'}} />
            </View>
            <Text style={{color: 'rgba(255,255,255,0.4)', fontSize: 11, marginTop: 6}}>
              {soilMoisture < 30 ? 'Low — irrigation needed' : soilMoisture > 65 ? 'High — hold off watering' : 'Optimal range (30–65%)'}
            </Text>
          </View>
        ) : null}

        {/* ── ML Insights ── */}
        <Text style={{color: '#FFFFFF', fontSize: 15, fontWeight: '700', marginBottom: 12}}>ML Insights</Text>
        <View style={{borderRadius: 16, backgroundColor: '#1A2B50', padding: 16, gap: 0, marginBottom: 16}}>
          {insightRows.map((insight, index) => (
            <View key={insight.title}>
              <View style={{flexDirection: 'row', alignItems: 'flex-start', paddingVertical: 10, gap: 12}}>
                <View style={{height: 34, width: 34, borderRadius: 9, backgroundColor: 'rgba(255,255,255,0.06)', alignItems: 'center', justifyContent: 'center', marginTop: 1}}>
                  <MaterialCommunityIcons name={insight.iconName} size={17} color={insight.dot} />
                </View>
                <View style={{flex: 1}}>
                  <Text style={{color: '#FFFFFF', fontSize: 13, fontWeight: '700'}}>{insight.title}</Text>
                  <Text style={{color: 'rgba(255,255,255,0.6)', fontSize: 12, lineHeight: 18, marginTop: 3}}>{insight.body}</Text>
                </View>
              </View>
              {index < insightRows.length - 1 ? <View style={{height: 1, backgroundColor: 'rgba(255,255,255,0.06)'}} /> : null}
            </View>
          ))}
        </View>

        {/* ── 3-Day Schedule ── */}
        <Text style={{color: '#FFFFFF', fontSize: 15, fontWeight: '700', marginBottom: 12}}>3-Day Schedule</Text>
        <View style={{gap: 10, marginBottom: 24}}>
          {schedule.map((item) => (
            <View key={item.label} style={{borderRadius: 14, backgroundColor: '#1A2B50', paddingHorizontal: 16, paddingVertical: 14,
              flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
              borderLeftWidth: 3, borderLeftColor: item.shouldWater ? '#22C55E' : '#374151'}}>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 12}}>
                <View style={{height: 40, width: 40, borderRadius: 10,
                  backgroundColor: item.shouldWater ? 'rgba(34,197,94,0.15)' : 'rgba(255,255,255,0.06)',
                  alignItems: 'center', justifyContent: 'center'}}>
                  <MaterialCommunityIcons name={item.shouldWater ? 'water' : 'water-off'} size={20}
                    color={item.shouldWater ? '#22C55E' : '#6B7280'} />
                </View>
                <View>
                  <Text style={{color: '#FFFFFF', fontSize: 14, fontWeight: '700'}}>{item.label}</Text>
                  <Text style={{color: 'rgba(255,255,255,0.4)', fontSize: 11, marginTop: 1}}>{item.fullDate}</Text>
                  {item.rainfall > 0 ? (
                    <View style={{flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2}}>
                      <MaterialCommunityIcons name="weather-rainy" size={10} color="#60A5FA" />
                      <Text style={{color: '#60A5FA', fontSize: 10}}>{item.rainfall} mm rain</Text>
                    </View>
                  ) : null}
                </View>
              </View>
              <View style={{alignItems: 'flex-end', gap: 4}}>
                <View style={{paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20,
                  backgroundColor: item.shouldWater ? 'rgba(34,197,94,0.15)' : 'rgba(255,255,255,0.06)'}}>
                  <Text style={{color: item.shouldWater ? '#22C55E' : '#9CA3AF', fontSize: 12, fontWeight: '700'}}>{item.statusLabel}</Text>
                </View>
                {item.shouldWater && item.waterMm > 0 ? (
                  <Text style={{color: 'rgba(255,255,255,0.4)', fontSize: 11}}>{item.waterMm} mm</Text>
                ) : null}
              </View>
            </View>
          ))}
        </View>

      </ScrollView>
    </SafeAreaView>
  );
};

export default SmartIrrigationScreen;
