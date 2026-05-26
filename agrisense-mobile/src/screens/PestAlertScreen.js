/**
 * PestAlertScreen
 *
 * Farmer inputs (interactive):
 *   • Crop type              — synced from cropStore (shown as chip)
 *   • Growth stage           — pill selector: Seedling / Vegetative / Flowering / Maturity
 *   • Previous pest history  — Yes / No toggle
 *
 * Auto-fetched (no farmer input needed):
 *   • temperature, humidity, rainfall_mm, wind_speed  ← /api/weather/current
 *
 * ML endpoint called:
 *   • POST /api/pest/predict → { risk_level, pest_type, confidence }
 */
import React, {useEffect, useState} from 'react';
import {ActivityIndicator, Pressable, SafeAreaView, ScrollView, StatusBar, Text, TouchableOpacity, View} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {buildPestPayloadFromInputs, getWeatherCurrent, predictPest} from '../services/api';
import cropStore from '../services/cropStore';
import {getCurrentPosition, requestLocationPermission} from '../services/location';
import locationStore from '../services/locationStore';
import useTranslation from '../services/useTranslation';

const DEFAULT_LAT = 18.6650;
const DEFAULT_LON = 77.9046;

const SEVERITY_CONFIG = {
  CRITICAL: {bg: '#7F1D1D', badge: '#DC2626', label: 'CRITICAL', icon: 'alert-circle',   iconColor: '#FCA5A5'},
  WARNING:  {bg: '#78350F', badge: '#D97706', label: 'WARNING',  icon: 'alert',           iconColor: '#FCD34D'},
  INFO:     {bg: '#14532D', badge: '#16A34A', label: 'LOW RISK', icon: 'information',     iconColor: '#86EFAC'},
};

const PEST_RECOMMENDATIONS = {
  aphids:     [
    'Inspect leaf undersides early each morning',
    'Apply neem-based spray when threshold is reached',
    'Avoid excess nitrogen fertiliser — it encourages aphid colonies',
  ],
  stem_borer: [
    'Check stems and leaf sheaths daily for dead-heart symptoms',
    'Remove and destroy infested tillers immediately',
    'Use approved insecticide if 5% dead-heart threshold is crossed',
  ],
  whitefly: [
    'Set yellow sticky traps at field margins immediately',
    'Spray underside of leaves with a recommended insecticide',
    'Avoid over-irrigation and dense planting that raises humidity',
  ],
};
const FALLBACK_REC = [
  'Scout your field and nearby plots every morning',
  'Keep field boundaries and irrigation channels clear of weeds',
  'Follow local Krishi Vigyan Kendra (KVK) advisory for your district',
];

const PEST_ICONS = {
  aphids: 'bug', stem_borer: 'sprout', whitefly: 'butterfly-outline',
};

const GROWTH_STAGES = ['seedling', 'vegetative', 'flowering', 'maturity'];

const PestAlertScreen = ({navigation}) => {
  // ── Farmer inputs ──────────────────────────────────────────
  const [selectedCrop, setSelectedCrop]  = useState('rice');
  const [growthStage, setGrowthStage]    = useState('vegetative');
  const [prevPest, setPrevPest]          = useState(1);

  // ── Data ───────────────────────────────────────────────────
  const [pestData, setPestData]          = useState(null);
  const [loading, setLoading]            = useState(true);
  const [refreshing, setRefreshing]      = useState(false);
  const [error, setError]                = useState(null);
  const [coords, setCoords]              = useState(null);

  const {t} = useTranslation();

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

  // ── Sync crop from store ────────────────────────────────────
  useEffect(() => {
    cropStore.getCrop().then((v) => { if (v) setSelectedCrop(v); });
    const unsub = cropStore.subscribe((v) => { if (v) setSelectedCrop(v); });
    return unsub;
  }, []);

  // ── Fetch pest prediction ───────────────────────────────────
  const loadData = async (isRefresh = false) => {
    const loc = coords ?? {lat: DEFAULT_LAT, lon: DEFAULT_LON};
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const weather = await getWeatherCurrent(loc.lat, loc.lon);
      const payload = buildPestPayloadFromInputs({
        weather,
        cropType: selectedCrop,
        growthStage,
        previousPestOccurrence: prevPest,
      });
      const response = await predictPest(payload);

      const confidencePct = Math.round(Number(response.confidence ?? 0) * 100);
      const risk          = String(response.risk_level || 'medium').toLowerCase();
      const severity      = risk === 'high' ? 'CRITICAL' : risk === 'medium' ? 'WARNING' : 'INFO';
      const pestKey       = String(response.pest_type ?? '').toLowerCase().replace(/\s+/g, '_');
      const pestName      = String(response.pest_type ?? 'General Pest').replace(/_/g, ' ');

      setPestData({
        severity,
        pestKey,
        pestName,
        confidence: confidencePct,
        expectedDays: risk === 'high' ? '1–3 days' : risk === 'medium' ? '3–5 days' : '5–7 days',
        location: weather.location_name ?? 'Current farm location',
        crop: selectedCrop,
        riskLabel: `${risk.charAt(0).toUpperCase() + risk.slice(1)} · ${confidencePct}% confidence`,
        recommendations: PEST_RECOMMENDATIONS[pestKey] ?? FALLBACK_REC,
        weather,
      });
    } catch (e) {
      console.error('Pest load error', e);
      setError('Unable to load live pest data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    if (coords !== null) loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coords]);

  useEffect(() => {
    if (coords !== null && !loading) loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCrop, growthStage, prevPest]);

  // ── Loading ─────────────────────────────────────────────────
  if (loading) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: '#F0FFF5'}}>
        <StatusBar barStyle="dark-content" backgroundColor="#F0FFF5" />
        <View style={{flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12}}>
          <ActivityIndicator size="large" color="#16A34A" />
          <Text style={{fontSize: 15, color: '#14532D', fontWeight: '600'}}>Analysing pest risk…</Text>
          <Text style={{fontSize: 13, color: '#4A7A5C'}}>Fetching live weather + ML prediction</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={{flex: 1, backgroundColor: '#F0FFF5'}}>
        <StatusBar barStyle="dark-content" backgroundColor="#F0FFF5" />
        <View style={{flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24, gap: 14}}>
          <MaterialCommunityIcons name="wifi-off" size={44} color="#DC2626" />
          <Text style={{fontSize: 15, color: '#14532D', fontWeight: '600', textAlign: 'center'}}>{error}</Text>
          <TouchableOpacity
            onPress={() => loadData()}
            style={{marginTop: 4, backgroundColor: '#15803D', borderRadius: 12, paddingHorizontal: 28, paddingVertical: 12}}>
            <Text style={{color: '#FFFFFF', fontWeight: '700'}}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const cfg = SEVERITY_CONFIG[pestData.severity] ?? SEVERITY_CONFIG.INFO;

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#F0FFF5'}}>
      <StatusBar barStyle="dark-content" backgroundColor="#F0FFF5" />

      {/* ── Header ── */}
      <View style={{flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 10, gap: 12}}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
          style={{width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 22, backgroundColor: 'rgba(21,128,61,0.1)'}}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#15803D" />
        </TouchableOpacity>
        <View style={{flex: 1}}>
          <Text style={{fontSize: 18, fontWeight: '700', color: '#14532D'}}>Pest Alert</Text>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2}}>
            <MaterialCommunityIcons name="map-marker" size={12} color="#16A34A" />
            <Text style={{fontSize: 12, color: '#4A7A5C'}}>{pestData?.location ?? 'Locating…'}</Text>
          </View>
        </View>
        <TouchableOpacity
          onPress={() => loadData(true)}
          style={{width: 38, height: 38, alignItems: 'center', justifyContent: 'center', borderRadius: 19, backgroundColor: 'rgba(21,128,61,0.1)'}}>
          {refreshing
            ? <ActivityIndicator size="small" color="#15803D" />
            : <MaterialCommunityIcons name="refresh" size={20} color="#15803D" />}
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => navigation.navigate('VoiceAssistant')}
          style={{width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 22, backgroundColor: '#15803D'}}>
          <MaterialCommunityIcons name="microphone" size={20} color="#FFFFFF" />
        </TouchableOpacity>
      </View>

      <View style={{height: 2, backgroundColor: '#16A34A', marginHorizontal: 16, borderRadius: 1, marginBottom: 4}} />

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 40, paddingTop: 8}}>

        {/* ── Farmer Input Card ── */}
        <View style={{backgroundColor: '#FFFFFF', borderRadius: 18, padding: 18, borderWidth: 1, borderColor: '#DCFCE7',
          shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 10, elevation: 2, marginBottom: 16}}>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14}}>
            <MaterialCommunityIcons name="tune-variant" size={18} color="#15803D" />
            <Text style={{fontSize: 14, fontWeight: '700', color: '#14532D'}}>Field Inputs</Text>
            <View style={{flex: 1}} />
            {/* Crop chip */}
            <View style={{backgroundColor: '#DCFCE7', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5, flexDirection: 'row', alignItems: 'center', gap: 5}}>
              <MaterialCommunityIcons name="barley" size={13} color="#15803D" />
              <Text style={{fontSize: 12, fontWeight: '700', color: '#15803D', textTransform: 'capitalize'}}>{selectedCrop}</Text>
            </View>
          </View>

          {/* Growth Stage */}
          <Text style={{fontSize: 12, color: '#166534', fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5}}>
            Growth Stage
          </Text>
          <View style={{flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 16}}>
            {GROWTH_STAGES.map((stage) => (
              <Pressable
                key={stage}
                onPress={() => setGrowthStage(stage)}
                style={{
                  paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20,
                  backgroundColor: growthStage === stage ? '#16A34A' : '#F0FFF4',
                  borderWidth: 1,
                  borderColor: growthStage === stage ? '#16A34A' : '#BBF7D0',
                }}>
                <Text style={{fontSize: 12, fontWeight: '600', textTransform: 'capitalize',
                  color: growthStage === stage ? '#FFFFFF' : '#166534'}}>
                  {stage}
                </Text>
              </Pressable>
            ))}
          </View>

          {/* Previous Pest */}
          <Text style={{fontSize: 12, color: '#166534', fontWeight: '600', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5}}>
            Did Pests Appear Before?
          </Text>
          <View style={{flexDirection: 'row', gap: 10}}>
            {[{label: 'Yes — had pests', val: 1, emoji: '⚠️'}, {label: 'No — all clear', val: 0, emoji: '✅'}].map((opt) => (
              <Pressable
                key={opt.val}
                onPress={() => setPrevPest(opt.val)}
                style={{
                  flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
                  paddingVertical: 10, borderRadius: 12,
                  backgroundColor: prevPest === opt.val ? '#16A34A' : '#F0FFF4',
                  borderWidth: 1,
                  borderColor: prevPest === opt.val ? '#16A34A' : '#BBF7D0',
                }}>
                <Text style={{fontSize: 14}}>{opt.emoji}</Text>
                <Text style={{fontSize: 12, fontWeight: '600', color: prevPest === opt.val ? '#FFFFFF' : '#166534'}}>{opt.label}</Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* ── Alert Card ── */}
        <View style={{backgroundColor: cfg.bg, borderRadius: 20, overflow: 'hidden',
          shadowColor: '#000', shadowOpacity: 0.12, shadowRadius: 14, elevation: 5, marginBottom: 16}}>
          <View style={{width: 5, position: 'absolute', left: 0, top: 0, bottom: 0, backgroundColor: cfg.badge}} />
          <View style={{alignItems: 'center', paddingVertical: 30, paddingHorizontal: 24}}>
            <View style={{width: 80, height: 80, borderRadius: 40, backgroundColor: 'rgba(255,255,255,0.1)',
              alignItems: 'center', justifyContent: 'center', marginBottom: 12}}>
              <MaterialCommunityIcons name={PEST_ICONS[pestData.pestKey] ?? 'bug'} size={40} color={cfg.iconColor} />
            </View>
            <View style={{backgroundColor: cfg.badge, borderRadius: 20, paddingHorizontal: 16, paddingVertical: 5, marginBottom: 10}}>
              <Text style={{color: '#FFFFFF', fontSize: 11, fontWeight: '700', letterSpacing: 1.2}}>{cfg.label}</Text>
            </View>
            <Text style={{fontSize: 24, fontWeight: '800', color: '#FFFFFF', textAlign: 'center', textTransform: 'capitalize'}}>
              {pestData.pestName}
            </Text>
            <Text style={{fontSize: 14, color: 'rgba(255,255,255,0.65)', marginTop: 4}}>Detected for your {pestData.crop} field</Text>

            {/* Confidence + expected */}
            <View style={{flexDirection: 'row', gap: 10, marginTop: 14}}>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 6,
                backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 7}}>
                <MaterialCommunityIcons name="brain" size={14} color="rgba(255,255,255,0.8)" />
                <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.85)', fontWeight: '600'}}>{pestData.confidence}% confidence</Text>
              </View>
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 6,
                backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 7}}>
                <MaterialCommunityIcons name="clock-outline" size={14} color="rgba(255,255,255,0.8)" />
                <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.8)'}}>{pestData.expectedDays}</Text>
              </View>
            </View>
          </View>
        </View>

        {/* ── Info rows ── */}
        <View style={{backgroundColor: '#FFFFFF', borderRadius: 16, paddingVertical: 4, paddingHorizontal: 16,
          borderWidth: 1, borderColor: '#DCFCE7', shadowColor: '#000', shadowOpacity: 0.03, shadowRadius: 6, elevation: 1, marginBottom: 16}}>
          {[
            {icon: 'map-marker-outline', iconColor: '#16A34A', label: 'Location',         value: pestData.location},
            {icon: 'barley',             iconColor: '#D97706', label: 'Crop',              value: pestData.crop.charAt(0).toUpperCase() + pestData.crop.slice(1)},
            {icon: 'sprout-outline',     iconColor: '#16A34A', label: 'Growth Stage',      value: growthStage.charAt(0).toUpperCase() + growthStage.slice(1)},
            {icon: 'shield-alert-outline',iconColor: '#DC2626',label: 'Risk Level',        value: pestData.riskLabel},
            {icon: 'thermometer',        iconColor: '#F97316', label: 'Live Temperature',  value: pestData.weather ? `${pestData.weather.temperature != null ? parseFloat(pestData.weather.temperature.toFixed(1)) : '—'}°C · ${Math.round(pestData.weather.humidity ?? 0)}% humidity` : '—'},
          ].map((row, idx, arr) => (
            <View key={row.label}>
              <View style={{flexDirection: 'row', alignItems: 'center', paddingVertical: 13, gap: 12}}>
                <View style={{width: 36, height: 36, borderRadius: 10, backgroundColor: '#F0FFF4', alignItems: 'center', justifyContent: 'center'}}>
                  <MaterialCommunityIcons name={row.icon} size={18} color={row.iconColor} />
                </View>
                <View style={{flex: 1}}>
                  <Text style={{fontSize: 11, color: '#6B7280', fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4}}>{row.label}</Text>
                  <Text style={{fontSize: 14, color: '#14532D', fontWeight: '600', marginTop: 2}}>{row.value}</Text>
                </View>
              </View>
              {idx < arr.length - 1 ? <View style={{height: 1, backgroundColor: '#F0FFF4'}} /> : null}
            </View>
          ))}
        </View>

        {/* ── Live weather bar ── */}
        {pestData.weather ? (
          <View style={{backgroundColor: '#F0FFF4', borderRadius: 14, borderWidth: 1, borderColor: '#BBF7D0', padding: 14, marginBottom: 16}}>
            <View style={{flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 12}}>
              <MaterialCommunityIcons name="broadcast" size={14} color="#15803D" />
              <Text style={{fontSize: 13, fontWeight: '700', color: '#14532D'}}>Live Weather Conditions</Text>
            </View>
            <View style={{flexDirection: 'row', justifyContent: 'space-between'}}>
              {[
                {icon: 'thermometer',   color: '#F97316', label: 'Temp',     val: `${pestData.weather.temperature != null ? parseFloat(pestData.weather.temperature.toFixed(1)) : '—'}°C`},
                {icon: 'water-percent', color: '#3B82F6', label: 'Humidity', val: `${Math.round(pestData.weather.humidity ?? 0)}%`},
                {icon: 'weather-rainy', color: '#60A5FA', label: 'Rain',     val: `${Math.round(pestData.weather.rainfall_mm ?? 0)} mm`},
                {icon: 'weather-windy', color: '#A78BFA', label: 'Wind',     val: `${Math.round(pestData.weather.wind_speed ?? 0)} km/h`},
              ].map((m) => (
                <View key={m.label} style={{alignItems: 'center', flex: 1}}>
                  <MaterialCommunityIcons name={m.icon} size={20} color={m.color} />
                  <Text style={{fontSize: 11, color: '#6B7280', marginTop: 4}}>{m.label}</Text>
                  <Text style={{fontSize: 13, fontWeight: '700', color: '#14532D', marginTop: 2}}>{m.val}</Text>
                </View>
              ))}
            </View>
          </View>
        ) : null}

        {/* ── Recommendations ── */}
        <View style={{backgroundColor: '#F0FFF4', borderRadius: 16, borderWidth: 1.5, borderColor: '#16A34A', padding: 16, marginBottom: 16}}>
          <View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14}}>
            <MaterialCommunityIcons name="clipboard-check-outline" size={20} color="#15803D" />
            <Text style={{fontSize: 15, fontWeight: '700', color: '#15803D'}}>What to Do Now (क्या करें)</Text>
          </View>
          {pestData.recommendations.map((rec, idx) => (
            <View key={idx} style={{flexDirection: 'row', alignItems: 'flex-start', gap: 10,
              marginBottom: idx < pestData.recommendations.length - 1 ? 12 : 0}}>
              <View style={{width: 24, height: 24, borderRadius: 12, backgroundColor: '#16A34A',
                alignItems: 'center', justifyContent: 'center', marginTop: 1, flexShrink: 0}}>
                <Text style={{color: '#FFFFFF', fontSize: 11, fontWeight: '700'}}>{idx + 1}</Text>
              </View>
              <Text style={{fontSize: 14, color: '#1A3A2E', flex: 1, lineHeight: 22}}>{rec}</Text>
            </View>
          ))}
        </View>

        {/* ── Action buttons ── */}
        <TouchableOpacity
          onPress={() => navigation.navigate('VoiceAssistant')}
          style={{backgroundColor: '#15803D', borderRadius: 14, paddingVertical: 16,
            flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 10}}>
          <MaterialCommunityIcons name="microphone" size={20} color="#FFFFFF" />
          <Text style={{color: '#FFFFFF', fontSize: 15, fontWeight: '700'}}>Ask Voice Assistant</Text>
        </TouchableOpacity>

        <View style={{flexDirection: 'row', gap: 10}}>
          <TouchableOpacity style={{flex: 1, borderWidth: 2, borderColor: '#15803D', borderRadius: 14,
            paddingVertical: 13, alignItems: 'center', backgroundColor: '#FFFFFF'}}>
            <Text style={{color: '#15803D', fontWeight: '600', fontSize: 14}}>View More Details</Text>
          </TouchableOpacity>
          <TouchableOpacity style={{flex: 1, borderWidth: 1.5, borderColor: '#86EFAC', borderRadius: 14,
            paddingVertical: 13, alignItems: 'center', backgroundColor: '#F0FFF4'}}>
            <Text style={{color: '#15803D', fontWeight: '600', fontSize: 14}}>Mark as Resolved</Text>
          </TouchableOpacity>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
};

export default PestAlertScreen;
