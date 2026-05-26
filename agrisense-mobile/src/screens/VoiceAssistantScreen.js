import React, {useCallback, useEffect, useRef, useState} from 'react';
import {Animated, Platform, PermissionsAndroid, SafeAreaView, ScrollView, StatusBar, Text, TouchableOpacity, View} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {buildIrrigationPayloadFromInputs, buildPestPayloadFromInputs, getWeatherCurrent, getWeatherForecast, predictIrrigation, predictPest} from '../services/api';
import cropStore from '../services/cropStore';
import {getCurrentPosition, requestLocationPermission} from '../services/location';
import locationStore from '../services/locationStore';
import useTranslation from '../services/useTranslation';

const DEFAULT_LAT = 18.6650;
const DEFAULT_LON = 77.9046;

// ── Mic permission helper ──────────────────────────────────────────────────
const requestMicPermission = async () => {
  if (Platform.OS === 'android') {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: 'Microphone Permission',
          message: 'AgriSense AI needs the microphone for the Voice Assistant.',
          buttonPositive: 'Allow',
          buttonNegative: 'Deny',
        },
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    } catch {
      return false;
    }
  }
  // iOS: NSMicrophoneUsageDescription in Info.plist triggers the system prompt automatically
  // when audio access is first used — no JS call needed here.
  return true;
};

// ── Rule-based answer engine ───────────────────────────────────────────────
const buildAnswer = async ({question, weather, forecast, selectedCrop, language}) => {
  const crop = selectedCrop ?? 'rice';
  const temp = weather?.temperature != null ? parseFloat(weather.temperature.toFixed(1)) : '—';
  const humidity = Math.round(weather?.humidity ?? 0);
  const rain = Math.round(weather?.rainfall_mm ?? 0);
  const loc = weather?.location_name ?? 'your farm';

  if (question === 'irrigation') {
    const payload = buildIrrigationPayloadFromInputs({
      weather, cropType: crop, growthStage: 'vegetative', daysSinceIrrigation: 4, soilFeel: 'moist',
    });
    const res = await predictIrrigation(payload);
    const shouldWater = (res.irrigate ?? '').toLowerCase() === 'yes';
    const water = res.water_amount_mm ?? 0;
    const etc = res.etc_mm_per_day?.toFixed(1) ?? '—';

    const answers = {
      english: {
        decision: shouldWater ? 'Yes — Water Today' : 'No — Skip Today',
        decisionLocal: shouldWater ? '✅ Irrigate' : '❌ Skip',
        body: shouldWater
          ? `ML model recommends irrigation for your ${crop} crop. Apply ${water} mm. Best time: 6–8 AM. ETc demand: ${etc} mm/day.`
          : `No irrigation needed. Soil moisture is adequate. ETc demand: ${etc} mm/day. Recheck tomorrow.`,
        reasons: [`Temperature: ${temp}°C`, `Humidity: ${humidity}%`, `Rainfall today: ${rain} mm`, `Water demand: ${etc} mm/day`],
      },
      hindi: {
        decision: shouldWater ? 'हाँ — आज पानी दें' : 'नहीं — आज रुकें',
        decisionLocal: shouldWater ? '✅ पानी दें' : '❌ रुकें',
        body: shouldWater
          ? `ML मॉडल: ${crop} फसल को ${water} mm पानी दें। सुबह 6–8 बजे सबसे अच्छा समय।`
          : `अभी पानी की जरूरत नहीं। मिट्टी में पर्याप्त नमी है। ETc: ${etc} mm/day।`,
        reasons: [`तापमान: ${temp}°C`, `आर्द्रता: ${humidity}%`, `बारिश: ${rain} mm`, `पानी की जरूरत: ${etc} mm/day`],
      },
      telugu: {
        decision: shouldWater ? 'అవును — ఈరోజు నీరు పోయండి' : 'వద్దు — ఈరోజు వద్దు',
        decisionLocal: shouldWater ? '✅ నీరు పోయండి' : '❌ ఆగండి',
        body: shouldWater
          ? `ML మోడల్: ${crop} పంటకు ${water} mm నీరు పోయండి. ఉదయం 6–8 గంటలు అత్యుత్తమ.`
          : `ఇప్పుడు నీటిపోత అవసరం లేదు. నేల తేమ తగినంత ఉంది. ETc: ${etc} mm/day.`,
        reasons: [`ఉష్ణోగ్రత: ${temp}°C`, `ఆర్ద్రత: ${humidity}%`, `వర్షం: ${rain} mm`, `నీటి అవసరం: ${etc} mm/day`],
      },
      marathi: {
        decision: shouldWater ? 'होय — आज पाणी द्या' : 'नाही — आज थांबा',
        decisionLocal: shouldWater ? '✅ पाणी द्या' : '❌ थांबा',
        body: shouldWater
          ? `ML मॉडेल: ${crop} पिकाला ${water} mm पाणी द्या. सकाळी 6–8 वाजणे सर्वोत्तम.`
          : `आत्ता पाण्याची गरज नाही. मातीत पुरेशी ओलावा. ETc: ${etc} mm/day.`,
        reasons: [`तापमान: ${temp}°C`, `आर्द्रता: ${humidity}%`, `पाऊस: ${rain} mm`, `पाण्याची मागणी: ${etc} mm/day`],
      },
    };
    return answers[language] ?? answers.english;
  }

  if (question === 'weather') {
    const cond = weather?.condition ?? 'partly cloudy';
    const wind = Math.round(weather?.wind_speed ?? 0);
    const answers = {
      english: {
        decision: `${temp}°C — ${cond}`,
        decisionLocal: loc,
        body: `Current at ${loc}: ${temp}°C, ${humidity}% humidity, ${rain} mm rainfall, ${wind} km/h wind. Condition: ${cond}.`,
        reasons: [`Temperature: ${temp}°C`, `Humidity: ${humidity}%`, `Rainfall: ${rain} mm`, `Wind: ${wind} km/h`],
      },
      hindi: {
        decision: `${temp}°C — ${cond}`,
        decisionLocal: `${loc} में मौसम`,
        body: `${loc} में अभी: तापमान ${temp}°C, आर्द्रता ${humidity}%, बारिश ${rain} mm, हवा ${wind} km/h। स्थिति: ${cond}।`,
        reasons: [`तापमान: ${temp}°C`, `आर्द्रता: ${humidity}%`, `बारिश: ${rain} mm`, `हवा: ${wind} km/h`],
      },
      telugu: {
        decision: `${temp}°C — ${cond}`,
        decisionLocal: `${loc} వాతావరణం`,
        body: `${loc}లో ఇప్పుడు: ${temp}°C, ఆర్ద్రత ${humidity}%, వర్షం ${rain} mm, గాలి ${wind} km/h.`,
        reasons: [`ఉష్ణోగ్రత: ${temp}°C`, `ఆర్ద్రత: ${humidity}%`, `వర్షం: ${rain} mm`, `గాలి: ${wind} km/h`],
      },
      marathi: {
        decision: `${temp}°C — ${cond}`,
        decisionLocal: `${loc} हवामान`,
        body: `${loc} येथे सध्या: ${temp}°C, आर्द्रता ${humidity}%, पाऊस ${rain} mm, वारा ${wind} km/h.`,
        reasons: [`तापमान: ${temp}°C`, `आर्द्रता: ${humidity}%`, `पाऊस: ${rain} mm`, `वारा: ${wind} km/h`],
      },
    };
    return answers[language] ?? answers.english;
  }

  if (question === 'pest') {
    const payload = buildPestPayloadFromInputs({weather, cropType: crop, growthStage: 'vegetative', previousPestOccurrence: 1});
    const res = await predictPest(payload);
    const risk = String(res.risk_level || 'low');
    const pestName = String(res.pest_type ?? 'General Pest').replace(/_/g, ' ');
    const conf = Math.round(Number(res.confidence ?? 0) * 100);
    const answers = {
      english: {
        decision: `${risk.charAt(0).toUpperCase() + risk.slice(1)} Risk — ${pestName}`,
        decisionLocal: `${conf}% confidence`,
        body: `ML pest model: ${pestName} at ${risk} risk for your ${crop} crop. Confidence: ${conf}%. Monitor your field closely.`,
        reasons: [`Pest: ${pestName}`, `Risk: ${risk}`, `Confidence: ${conf}%`, `Temp: ${temp}°C · Humidity: ${humidity}%`],
      },
      hindi: {
        decision: `${risk === 'high' ? 'अधिक' : risk === 'medium' ? 'मध्यम' : 'कम'} जोखिम — ${pestName}`,
        decisionLocal: `${conf}% विश्वास`,
        body: `ML मॉडल: ${crop} फसल में ${pestName} — ${risk} जोखिम। विश्वास: ${conf}%। खेत की निगरानी करें।`,
        reasons: [`कीट: ${pestName}`, `जोखिम: ${risk}`, `विश्वास: ${conf}%`, `तापमान: ${temp}°C`],
      },
      telugu: {
        decision: `${risk === 'high' ? 'అధిక' : risk === 'medium' ? 'మధ్యమ' : 'తక్కువ'} రిస్క్ — ${pestName}`,
        decisionLocal: `${conf}% నమ్మకం`,
        body: `ML మోడల్: ${crop} పంటలో ${pestName} — ${risk} రిస్క్. నమ్మకం: ${conf}%.`,
        reasons: [`పురుగు: ${pestName}`, `రిస్క్: ${risk}`, `నమ్మకం: ${conf}%`, `ఉష్ణోగ్రత: ${temp}°C`],
      },
      marathi: {
        decision: `${risk === 'high' ? 'उच्च' : risk === 'medium' ? 'मध्यम' : 'कमी'} धोका — ${pestName}`,
        decisionLocal: `${conf}% विश्वास`,
        body: `ML मॉडेल: ${crop} पिकात ${pestName} — ${risk} धोका. विश्वास: ${conf}%.`,
        reasons: [`कीड: ${pestName}`, `धोका: ${risk}`, `विश्वास: ${conf}%`, `तापमान: ${temp}°C`],
      },
    };
    return answers[language] ?? answers.english;
  }

  if (question === 'next_water') {
    const dayPayloads = (forecast ?? []).slice(0, 3).map((day, idx) =>
      buildIrrigationPayloadFromInputs({
        weather: {temperature: day.temperature, humidity: day.humidity, rainfall_mm: day.rainfall_mm, wind_speed: day.wind_speed},
        cropType: crop, growthStage: 'vegetative', daysSinceIrrigation: 4 + idx,
        soilFeel: day.rainfall_mm > 5 ? 'wet' : 'moist',
      }),
    );
    const results = await Promise.all(dayPayloads.map(predictIrrigation));
    const nextIdx = results.findIndex((r) => (r.irrigate ?? '').toLowerCase() === 'yes');
    const dayLabels = {
      english: ['Today', 'Tomorrow', 'Day after tomorrow'],
      hindi: ['आज', 'कल', 'परसों'],
      telugu: ['ఈరోజు', 'రేపు', 'ఎల్లుండి'],
      marathi: ['आज', 'उद्या', 'परवा'],
    };
    const labels = dayLabels[language] ?? dayLabels.english;
    const noWaterLabel = {english: 'Not this week', hindi: 'इस हफ्ते नहीं', telugu: 'ఈ వారంలో లేదు', marathi: 'या आठवड्यात नाही'}[language] ?? 'Not this week';
    const nextLabel = nextIdx >= 0 ? labels[nextIdx] : noWaterLabel;
    const answers = {
      english: {
        decision: nextLabel,
        decisionLocal: 'Next Irrigation',
        body: nextIdx >= 0
          ? `Next watering: ${nextLabel}. Apply ${results[nextIdx]?.water_amount_mm ?? 0} mm. Best time: 6–8 AM.`
          : 'No irrigation needed for the next 3 days based on current forecast.',
        reasons: results.slice(0, 3).map((r, i) => `${dayLabels.english[i]}: ${(r.irrigate ?? '').toLowerCase() === 'yes' ? `Irrigate (${r.water_amount_mm ?? 0} mm)` : 'Skip'}`),
      },
      hindi: {
        decision: nextLabel,
        decisionLocal: 'अगली सिंचाई',
        body: nextIdx >= 0
          ? `${nextLabel} को सिंचाई करें। ${results[nextIdx]?.water_amount_mm ?? 0} mm पानी। सुबह 6–8 बजे।`
          : 'अगले 3 दिनों में सिंचाई की जरूरत नहीं।',
        reasons: results.slice(0, 3).map((r, i) => `${dayLabels.hindi[i]}: ${(r.irrigate ?? '').toLowerCase() === 'yes' ? `पानी दें (${r.water_amount_mm ?? 0} mm)` : 'रुकें'}`),
      },
      telugu: {
        decision: nextLabel,
        decisionLocal: 'తదుపరి నీటిపోత',
        body: nextIdx >= 0
          ? `${nextLabel} నీటిపోత చేయండి. ${results[nextIdx]?.water_amount_mm ?? 0} mm నీరు.`
          : 'వచ్చే 3 రోజులలో నీటిపోత అవసరం లేదు.',
        reasons: results.slice(0, 3).map((r, i) => `${dayLabels.telugu[i]}: ${(r.irrigate ?? '').toLowerCase() === 'yes' ? `నీరు (${r.water_amount_mm ?? 0} mm)` : 'ఆగండి'}`),
      },
      marathi: {
        decision: nextLabel,
        decisionLocal: 'पुढचे पाणी',
        body: nextIdx >= 0
          ? `${nextLabel} पाणी द्या. ${results[nextIdx]?.water_amount_mm ?? 0} mm. सकाळी 6–8.`
          : 'पुढील 3 दिवस पाण्याची गरज नाही.',
        reasons: results.slice(0, 3).map((r, i) => `${dayLabels.marathi[i]}: ${(r.irrigate ?? '').toLowerCase() === 'yes' ? `पाणी (${r.water_amount_mm ?? 0} mm)` : 'थांबा'}`),
      },
    };
    return answers[language] ?? answers.english;
  }

  return {decision: '—', decisionLocal: '—', body: 'Select a question below.', reasons: []};
};

// ── Language-aware question labels ─────────────────────────────────────────
const QUICK_QUESTIONS = [
  {
    key: 'irrigation', icon: 'water', color: '#2563EB', border: '#60A5FA',
    labels: {english: 'Water today?', hindi: 'आज पानी दूं?', telugu: 'ఈరోజు నీరు?', marathi: 'आज पाणी?'},
    sub:    {english: 'Irrigate now?', hindi: 'सिंचाई करें?', telugu: 'నీటిపోత?',    marathi: 'सिंचन करावे?'},
  },
  {
    key: 'weather', icon: 'weather-partly-cloudy', color: '#D97706', border: '#FCD34D',
    labels: {english: 'Weather now?', hindi: 'मौसम कैसा है?', telugu: 'వాతావరణం?', marathi: 'हवामान?'},
    sub:    {english: 'Live forecast', hindi: 'लाइव मौसम',     telugu: 'ప్రత్యక్ష',  marathi: 'थेट हवामान'},
  },
  {
    key: 'pest', icon: 'bug-outline', color: '#15803D', border: '#4ADE80',
    labels: {english: 'Pest alert?', hindi: 'कीट समस्या?',  telugu: 'పురుగు హెచ్చరిక?', marathi: 'कीड सतर्कता?'},
    sub:    {english: 'ML risk scan', hindi: 'ML जाँच',       telugu: 'ML స్కాన్',         marathi: 'ML तपासणी'},
  },
  {
    key: 'next_water', icon: 'calendar-clock', color: '#7C3AED', border: '#C4B5FD',
    labels: {english: 'Next watering?', hindi: 'अगला पानी कब?', telugu: 'తదుపరి నీరు?', marathi: 'पुढचे पाणी?'},
    sub:    {english: '3-day schedule', hindi: '3 दिन की योजना', telugu: '3 రోజుల షెడ్యూల్', marathi: '3 दिवसांचे वेळापत्रक'},
  },
];

const ALL_LANGS = [
  {key: 'english', label: 'English'},
  {key: 'hindi',   label: 'हिंदी'},
  {key: 'telugu',  label: 'తెలుగు'},
  {key: 'marathi', label: 'मराठी'},
];

const REASON_LABEL = {english: 'Reason', hindi: 'कारण', telugu: 'కారణం', marathi: 'कारण'};

const VoiceAssistantScreen = ({navigation}) => {
  const [selectedCrop, setSelectedCrop]           = useState('rice');
  // ✅ Default is 'english' — was wrongly 'hindi' before
  const [selectedLanguage, setSelectedLanguage]   = useState('english');
  const [activeQuestion, setActiveQuestion]       = useState('irrigation');
  const [answer, setAnswer]                       = useState(null);
  const [answerLoading, setAnswerLoading]         = useState(false);
  const [weather, setWeather]                     = useState(null);
  const [forecast, setForecast]                   = useState([]);
  const [coords, setCoords]                       = useState(null);
  const [micPermission, setMicPermission]         = useState(null); // null=unknown true=granted false=denied

  const pulseScale   = useRef(new Animated.Value(1)).current;
  const pulseOpacity = useRef(new Animated.Value(0.18)).current;
  const {t} = useTranslation();

  // ── Pulse animation ──────────────────────────────────────────
  useEffect(() => {
    const anim = Animated.loop(Animated.parallel([
      Animated.sequence([
        Animated.timing(pulseScale,   {toValue: 1.35, duration: 900, useNativeDriver: true}),
        Animated.timing(pulseScale,   {toValue: 1,    duration: 900, useNativeDriver: true}),
      ]),
      Animated.sequence([
        Animated.timing(pulseOpacity, {toValue: 0,    duration: 900, useNativeDriver: true}),
        Animated.timing(pulseOpacity, {toValue: 0.18, duration: 900, useNativeDriver: true}),
      ]),
    ]));
    anim.start();
    return () => anim.stop();
  }, [pulseOpacity, pulseScale]);

  // ── Mic permission on mount ──────────────────────────────────
  useEffect(() => {
    requestMicPermission().then(setMicPermission);
  }, []);

  // ── Load crop + language from store ─────────────────────────
  useEffect(() => {
    (async () => {
      const [crop, lang] = await Promise.all([cropStore.getCrop(), cropStore.getLanguage()]);
      if (crop) setSelectedCrop(crop);
      // ✅ default to 'english', never 'hindi'
      if (lang) setSelectedLanguage(lang);
    })();
    const unsubCrop = cropStore.subscribe((v) => { if (v) setSelectedCrop(v); });
    // ✅ subscription fallback is 'english', not 'hindi'
    const unsubLang = cropStore.subscribeLanguage((v) => setSelectedLanguage(v || 'english'));
    return () => { unsubCrop(); unsubLang(); };
  }, []);

  // ── Resolve location ─────────────────────────────────────────
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

  // ── Load weather ─────────────────────────────────────────────
  useEffect(() => {
    if (!coords) return;
    (async () => {
      try {
        const [w, f] = await Promise.all([
          getWeatherCurrent(coords.lat, coords.lon),
          getWeatherForecast(coords.lat, coords.lon),
        ]);
        setWeather(w);
        setForecast(f?.forecast ?? []);
      } catch (e) {
        console.error('VoiceAssistant weather load', e);
      }
    })();
  }, [coords]);

  // ── Build answer when inputs change ──────────────────────────
  const loadAnswer = useCallback(async () => {
    if (!weather) return;
    setAnswerLoading(true);
    try {
      const ans = await buildAnswer({
        question: activeQuestion,
        weather,
        forecast,
        selectedCrop,
        language: selectedLanguage,
      });
      setAnswer(ans);
    } catch (e) {
      console.error('buildAnswer error', e);
      setAnswer({decision: 'Error', decisionLocal: '—', body: 'Could not load answer. Check backend.', reasons: []});
    } finally {
      setAnswerLoading(false);
    }
  }, [activeQuestion, weather, forecast, selectedCrop, selectedLanguage]);

  useEffect(() => { loadAnswer(); }, [loadAnswer]);

  const activeQ = QUICK_QUESTIONS.find((q) => q.key === activeQuestion);
  const reasonLabel = REASON_LABEL[selectedLanguage] ?? 'Reason';

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#0B3820'}}>
      <StatusBar barStyle="light-content" backgroundColor="#0B3820" />

      {/* ── Top bar ── */}
      <View style={{flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 8}}>
        <View style={{flexDirection: 'row', alignItems: 'center', gap: 8}}>
          <MaterialCommunityIcons name="translate" size={18} color="#B6F2D0" />
          <Text style={{fontSize: 13, color: '#E7F7ED', fontWeight: '600'}}>
            {ALL_LANGS.find((l) => l.key === selectedLanguage)?.label ?? selectedLanguage}
          </Text>
          {micPermission === false ? (
            <View style={{flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: 'rgba(239,68,68,0.2)', borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3}}>
              <MaterialCommunityIcons name="microphone-off" size={12} color="#FCA5A5" />
              <Text style={{fontSize: 11, color: '#FCA5A5', fontWeight: '600'}}>Mic denied</Text>
            </View>
          ) : micPermission === true ? (
            <View style={{flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: 'rgba(34,197,94,0.15)', borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3}}>
              <MaterialCommunityIcons name="microphone-outline" size={12} color="#86EFAC" />
              <Text style={{fontSize: 11, color: '#86EFAC', fontWeight: '600'}}>Mic ready</Text>
            </View>
          ) : null}
        </View>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}
          style={{width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.08)'}}>
          <MaterialCommunityIcons name="close" size={20} color="#FFFFFF" />
        </TouchableOpacity>
      </View>

      {/* ── Mic orb ── */}
      <View style={{alignItems: 'center', paddingTop: 18, paddingBottom: 8}}>
        <View style={{width: 148, height: 148, alignItems: 'center', justifyContent: 'center'}}>
          <Animated.View style={{
            position: 'absolute', width: 148, height: 148, borderRadius: 74,
            backgroundColor: micPermission === false ? '#EF4444' : '#34D399',
            transform: [{scale: pulseScale}], opacity: pulseOpacity,
          }} />
          <TouchableOpacity
            onPress={micPermission === false ? () => requestMicPermission().then(setMicPermission) : loadAnswer}
            style={{
              width: 120, height: 120, borderRadius: 60,
              backgroundColor: micPermission === false ? '#DC2626' : '#34D399',
              borderWidth: 4, borderColor: '#FFFFFF',
              alignItems: 'center', justifyContent: 'center',
              shadowColor: micPermission === false ? '#EF4444' : '#4ADE80',
              shadowOpacity: 0.6, shadowRadius: 20, elevation: 10,
            }}>
            <MaterialCommunityIcons
              name={micPermission === false ? 'microphone-off' : 'microphone'}
              size={52} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
        <Text style={{fontSize: 18, fontWeight: '600', color: '#FFFFFF', marginTop: 12}}>
          {micPermission === false ? 'Tap to enable mic' : t('voice', 'listening')}
        </Text>
        <View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8}}>
          <MaterialCommunityIcons name={activeQ?.icon ?? 'help-circle'} size={16} color="#34D399" />
          {/* ✅ Language-aware question label */}
          <Text style={{fontSize: 13, color: '#B6F2D0', fontWeight: '600'}}>
            {activeQ?.labels?.[selectedLanguage] ?? activeQ?.labels?.english ?? '—'}
          </Text>
        </View>
      </View>

      {/* ── Response panel ── */}
      <View style={{flex: 1, backgroundColor: '#FFFFFF', borderTopLeftRadius: 28, borderTopRightRadius: 28, marginTop: 8}}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 18, paddingBottom: 32, paddingTop: 20, gap: 16}}>

          {/* AI header */}
          <View style={{flexDirection: 'row', gap: 12, alignItems: 'flex-start'}}>
            <View style={{width: 48, height: 48, borderRadius: 24, backgroundColor: '#10B981', alignItems: 'center', justifyContent: 'center'}}>
              <MaterialCommunityIcons name="robot" size={26} color="#FFFFFF" />
            </View>
            <View style={{flex: 1}}>
              <Text style={{fontSize: 15, fontWeight: '700', color: '#14532D'}}>AgriSense AI</Text>
              {weather ? (
                <View style={{flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2}}>
                  <MaterialCommunityIcons name="map-marker" size={12} color="#16A34A" />
                  <Text style={{fontSize: 12, color: '#6B7280'}}>{weather.location_name ?? 'Farm'} · Live data</Text>
                </View>
              ) : null}
            </View>
            <TouchableOpacity onPress={loadAnswer}
              style={{width: 36, height: 36, borderRadius: 18, backgroundColor: '#F0FFF4', alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#BBF7D0'}}>
              <MaterialCommunityIcons name="refresh" size={18} color="#16A34A" />
            </TouchableOpacity>
          </View>

          {/* Decision card */}
          {answerLoading ? (
            <View style={{backgroundColor: '#F0FFF4', borderRadius: 16, padding: 20, alignItems: 'center', gap: 10}}>
              <MaterialCommunityIcons name="loading" size={28} color="#16A34A" />
              <Text style={{fontSize: 14, color: '#14532D', fontWeight: '600'}}>Analysing live data…</Text>
            </View>
          ) : answer ? (
            <>
              <View style={{backgroundColor: '#FFF5F4', borderWidth: 2, borderColor: '#F87171', borderRadius: 16, padding: 16}}>
                <Text style={{fontSize: 22, fontWeight: '800', color: '#DC2626'}}>{answer.decision}</Text>
                <Text style={{fontSize: 15, fontWeight: '600', color: '#EF4444', marginTop: 4}}>{answer.decisionLocal}</Text>
              </View>

              <View style={{backgroundColor: '#F0FFF4', borderRadius: 14, borderWidth: 1, borderColor: '#BBF7D0', padding: 14}}>
                <Text style={{fontSize: 13, fontWeight: '700', color: '#14532D', marginBottom: 8}}>{reasonLabel}:</Text>
                <Text style={{fontSize: 13, color: '#166534', lineHeight: 20}}>{answer.body}</Text>
              </View>

              {answer.reasons?.length > 0 ? (
                <View style={{gap: 8}}>
                  {answer.reasons.map((reason, idx) => (
                    <View key={idx} style={{flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#F8FAFC', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10}}>
                      <MaterialCommunityIcons name="check-circle-outline" size={16} color="#16A34A" />
                      <Text style={{fontSize: 13, color: '#374151', flex: 1}}>{reason}</Text>
                    </View>
                  ))}
                </View>
              ) : null}
            </>
          ) : (
            <View style={{backgroundColor: '#F0FFF4', borderRadius: 14, padding: 16, alignItems: 'center'}}>
              <Text style={{color: '#14532D', fontSize: 14}}>Waiting for weather data…</Text>
            </View>
          )}

          {/* ✅ Language-aware quick questions */}
          <View>
            <Text style={{fontSize: 13, fontWeight: '700', color: '#14532D', marginBottom: 10}}>
              {t('voice', 'askMeAbout')}
            </Text>
            <View style={{flexDirection: 'row', flexWrap: 'wrap', gap: 10}}>
              {QUICK_QUESTIONS.map((q) => {
                const isActive = activeQuestion === q.key;
                return (
                  <TouchableOpacity
                    key={q.key}
                    onPress={() => setActiveQuestion(q.key)}
                    style={{
                      width: '47%', borderRadius: 14, paddingVertical: 14, paddingHorizontal: 12,
                      backgroundColor: isActive ? q.color : q.color + '18',
                      borderWidth: 2, borderColor: isActive ? q.border : q.color + '44',
                    }}>
                    <MaterialCommunityIcons name={q.icon} size={20} color={isActive ? '#FFFFFF' : q.color} />
                    {/* ✅ Shows label in the currently selected language */}
                    <Text style={{fontSize: 14, fontWeight: '700', color: isActive ? '#FFFFFF' : q.color, marginTop: 6}}>
                      {q.labels[selectedLanguage] ?? q.labels.english}
                    </Text>
                    <Text style={{fontSize: 11, color: isActive ? 'rgba(255,255,255,0.8)' : '#6B7280', marginTop: 2}}>
                      {q.sub[selectedLanguage] ?? q.sub.english}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* ✅ Language selector — full list, real-time update */}
          <View style={{borderTopWidth: 1, borderTopColor: '#E2E8F0', paddingTop: 14}}>
            <Text style={{fontSize: 12, color: '#9CA3AF', fontWeight: '600', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.5}}>
              Language / भाषा / భాష
            </Text>
            <View style={{flexDirection: 'row', flexWrap: 'wrap', gap: 8}}>
              {ALL_LANGS.map((lang) => (
                <TouchableOpacity
                  key={lang.key}
                  onPress={async () => {
                    setSelectedLanguage(lang.key);
                    await cropStore.setLanguage(lang.key);
                    // ✅ Answer re-builds immediately because selectedLanguage is in loadAnswer deps
                  }}
                  style={{
                    paddingHorizontal: 16, paddingVertical: 9, borderRadius: 20,
                    backgroundColor: selectedLanguage === lang.key ? '#15803D' : '#F0FFF4',
                    borderWidth: 1.5, borderColor: selectedLanguage === lang.key ? '#15803D' : '#BBF7D0',
                  }}>
                  <Text style={{fontSize: 13, fontWeight: '700', color: selectedLanguage === lang.key ? '#FFFFFF' : '#166534'}}>
                    {lang.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

        </ScrollView>
      </View>
    </SafeAreaView>
  );
};

export default VoiceAssistantScreen;
