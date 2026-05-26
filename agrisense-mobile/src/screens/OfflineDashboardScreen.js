/**
 * OfflineDashboardScreen
 *
 * Saves real JSON files to the app's Documents directory.
 * Because Info.plist has UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace,
 * these files are visible in the iOS Files app under:
 *   On My iPhone → AgriSense AI
 *
 * Files written:
 *   agrisense_irrigation.json
 *   agrisense_weather.json
 *   agrisense_pest.json
 *   agrisense_tips.json
 */
import React, {useEffect, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  SafeAreaView,
  ScrollView,
  Share,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import RNFS from 'react-native-fs';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {
  getDashboardWeather,
  getWeatherForecast,
  predictIrrigation,
  buildIrrigationPayloadFromInputs,
  predictPest,
  buildPestPayloadFromInputs,
} from '../services/api';
import locationStore from '../services/locationStore';
import cropStore from '../services/cropStore';

const DEFAULT_LAT = 18.6650;
const DEFAULT_LON = 77.9046;
const DOCS = RNFS.DocumentDirectoryPath;

const FILE_NAMES = {
  irrigation: 'agrisense_irrigation.json',
  weather:    'agrisense_weather.json',
  pest:       'agrisense_pest.json',
  tips:       'agrisense_tips.json',
};

const DOWNLOAD_CONFIG = [
  {id: 'irrigation', icon: 'calendar-check',       iconBg: '#3B82F6', cardBg: 'rgba(59,130,246,0.14)',  borderColor: 'rgba(96,165,250,0.35)',  title: 'Irrigation Schedule', subtitle: '7-day ML watering plan',   subtitleHindi: 'पानी देने की योजना'},
  {id: 'weather',    icon: 'weather-partly-cloudy', iconBg: '#F97316', cardBg: 'rgba(249,115,22,0.14)',  borderColor: 'rgba(251,146,60,0.35)',  title: 'Weather Data',        subtitle: '7-day forecast',           subtitleHindi: 'मौसम का पूर्वानुमान'},
  {id: 'pest',       icon: 'bug-outline',           iconBg: '#EF4444', cardBg: 'rgba(239,68,68,0.14)',   borderColor: 'rgba(248,113,113,0.35)', title: 'Pest Alerts',         subtitle: 'ML risk prediction',       subtitleHindi: 'कीट अलर्ट'},
  {id: 'tips',       icon: 'sprout-outline',        iconBg: '#22C55E', cardBg: 'rgba(34,197,94,0.14)',   borderColor: 'rgba(74,222,128,0.35)',  title: 'Farming Tips',        subtitle: 'Seasonal guidance',        subtitleHindi: 'खेती के सुझाव'},
];

// ── Data fetchers ───────────────────────────────────────────────────────────

const fetchIrrigation = async (lat, lon, crop) => {
  const weather      = await getDashboardWeather(lat, lon);
  const forecastResp = await getWeatherForecast(lat, lon);
  const forecastList = forecastResp?.forecast ?? [];

  const schedule = await Promise.all(
    forecastList.slice(0, 7).map(async (day, idx) => {
      const payload = buildIrrigationPayloadFromInputs({
        weather: {temperature: day.temperature, humidity: day.humidity, rainfall_mm: day.rainfall_mm, wind_speed: day.wind_speed},
        cropType: crop, growthStage: 'vegetative',
        daysSinceIrrigation: idx + 1,
        soilFeel: day.rainfall_mm > 5 ? 'wet' : day.rainfall_mm > 1 ? 'moist' : 'dry',
      });
      const res = await predictIrrigation(payload);
      return {
        date: day.date,
        irrigate: res.irrigate,
        water_mm: res.water_amount_mm ?? 0,
        etc_mm_per_day: res.etc_mm_per_day ?? 0,
        condition: day.condition ?? '',
      };
    }),
  );
  return {savedAt: new Date().toISOString(), crop, location: weather.location_name ?? 'Farm', schedule};
};

const fetchWeather = async (lat, lon) => {
  const [current, forecastResp] = await Promise.all([
    getDashboardWeather(lat, lon),
    getWeatherForecast(lat, lon),
  ]);
  return {savedAt: new Date().toISOString(), current, forecast: forecastResp?.forecast ?? []};
};

const fetchPest = async (lat, lon, crop) => {
  const weather = await getDashboardWeather(lat, lon);
  const payload = buildPestPayloadFromInputs({weather, cropType: crop, growthStage: 'vegetative', previousPestOccurrence: 1});
  const result  = await predictPest(payload);
  return {savedAt: new Date().toISOString(), crop, location: weather.location_name ?? 'Farm', weather, result};
};

const fetchTips = async (_lat, _lon, crop) => {
  return {
    savedAt: new Date().toISOString(),
    crop,
    tips: [
      {title: 'Irrigation timing',   body: 'Water early morning (6–8 AM) to cut evaporation by up to 30%.'},
      {title: 'Soil health',         body: 'Test soil pH every season. Most crops thrive at pH 6.0–7.0.'},
      {title: 'Pest scouting',       body: 'Walk your field in a W-pattern, check 5 spots for early detection.'},
      {title: 'Fertiliser splits',   body: `Split fertiliser into 3 applications: basal, tillering, panicle initiation.`},
      {title: 'Water conservation',  body: 'Mulching reduces soil moisture loss by 25% and keeps roots cooler.'},
      {title: 'Harvest timing',      body: 'Harvest at 20–25% grain moisture for better storage and less breakage.'},
    ],
  };
};

const FETCHERS = {irrigation: fetchIrrigation, weather: fetchWeather, pest: fetchPest, tips: fetchTips};

// ── Helpers ─────────────────────────────────────────────────────────────────

const filePath = (id) => `${DOCS}/${FILE_NAMES[id]}`;

const readFileMeta = async (id) => {
  try {
    const path   = filePath(id);
    const exists = await RNFS.exists(path);
    if (!exists) return null;
    const stat   = await RNFS.stat(path);
    const raw    = await RNFS.readFile(path, 'utf8');
    const parsed = JSON.parse(raw);
    return {savedAt: parsed.savedAt, sizeKB: Math.round(stat.size / 1024)};
  } catch {
    return null;
  }
};

const fmtDate = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return (
    d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'}) +
    ' ' +
    d.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'})
  );
};

// ── Component ────────────────────────────────────────────────────────────────

const OfflineDashboardScreen = ({navigation}) => {
  const [fileMeta,    setFileMeta]    = useState({});   // {id: {savedAt, sizeKB} | null}
  const [downloading, setDownloading] = useState({});   // {id: bool}
  const [totalKB,     setTotalKB]     = useState(0);
  const [coords,      setCoords]      = useState(null);
  const [crop,        setCrop]        = useState('rice');

  // ── Resolve location + crop ───────────────────────────────────
  useEffect(() => {
    locationStore.getLocation()
      .then((loc) => setCoords(loc ?? {lat: DEFAULT_LAT, lon: DEFAULT_LON}))
      .catch(()    => setCoords({lat: DEFAULT_LAT, lon: DEFAULT_LON}));

    cropStore.getCrop().then((v) => { if (v) setCrop(v); });
    const unsub = cropStore.subscribe((v) => { if (v) setCrop(v); });
    return unsub;
  }, []);

  // ── Load metadata for all files on mount ──────────────────────
  useEffect(() => {
    (async () => {
      const meta = {};
      let kb = 0;
      for (const cfg of DOWNLOAD_CONFIG) {
        const m = await readFileMeta(cfg.id);
        meta[cfg.id] = m;
        if (m) kb += m.sizeKB;
      }
      setFileMeta(meta);
      setTotalKB(kb);
    })();
  }, []);

  // ── Download one item ─────────────────────────────────────────
  const handleDownload = async (id) => {
    const loc = coords ?? {lat: DEFAULT_LAT, lon: DEFAULT_LON};
    setDownloading((p) => ({...p, [id]: true}));
    try {
      const data = await FETCHERS[id](loc.lat, loc.lon, crop);
      const json = JSON.stringify(data, null, 2);
      await RNFS.writeFile(filePath(id), json, 'utf8');
      const stat = await RNFS.stat(filePath(id));
      const kb   = Math.round(stat.size / 1024);
      const old  = fileMeta[id]?.sizeKB ?? 0;
      setFileMeta((p) => ({...p, [id]: {savedAt: data.savedAt, sizeKB: kb}}));
      setTotalKB((p)  => p - old + kb);
    } catch (e) {
      Alert.alert('Download failed', `Could not save ${id} data.\n${e.message}`);
    } finally {
      setDownloading((p) => ({...p, [id]: false}));
    }
  };

  const handleDownloadAll = async () => {
    for (const cfg of DOWNLOAD_CONFIG) {
      await handleDownload(cfg.id);
    }
  };

  // ── Delete one file ───────────────────────────────────────────
  const handleDelete = (id) => {
    Alert.alert('Delete file', `Remove ${FILE_NAMES[id]} from Files?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => {
          try { await RNFS.unlink(filePath(id)); } catch {}
          const kb = fileMeta[id]?.sizeKB ?? 0;
          setFileMeta((p) => ({...p, [id]: null}));
          setTotalKB((p) => Math.max(0, p - kb));
        },
      },
    ]);
  };

  // ── Share file via iOS share sheet ───────────────────────────
  const handleShare = async (id) => {
    try {
      await Share.share({
        title: FILE_NAMES[id],
        url:   `file://${filePath(id)}`,
        message: `AgriSense AI — ${FILE_NAMES[id]}`,
      });
    } catch {}
  };

  const usedMB      = (totalKB / 1024).toFixed(2);
  const TOTAL_MB    = 50;
  const progressPct = Math.min(Math.round((totalKB / (TOTAL_MB * 1024)) * 100), 100);

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: '#3B0F63'}}>
      <StatusBar barStyle="light-content" backgroundColor="#3B0F63" />

      {/* ── Header ── */}
      <View style={{flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingTop: 12, paddingBottom: 14}}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
          style={{width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.1)'}}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#FFFFFF" />
        </TouchableOpacity>
        <View style={{flex: 1}}>
          <Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>Offline Dashboard</Text>
          <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.6)', marginTop: 1}}>Saved to iOS Files app</Text>
        </View>
        {/* Files app hint */}
        <View style={{backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 6, flexDirection: 'row', alignItems: 'center', gap: 5}}>
          <MaterialCommunityIcons name="folder-outline" size={14} color="#C4B5FD" />
          <Text style={{fontSize: 11, fontWeight: '600', color: '#C4B5FD'}}>Files</Text>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 36}}>

        {/* ── Files app banner ── */}
        <View style={{backgroundColor: '#7C3AED', borderRadius: 20, padding: 16, borderWidth: 1.5, borderColor: 'rgba(196,181,253,0.5)', marginBottom: 20}}>
          <View style={{flexDirection: 'row', gap: 14, alignItems: 'flex-start'}}>
            <View style={{width: 48, height: 48, borderRadius: 24, backgroundColor: 'rgba(255,255,255,0.15)', alignItems: 'center', justifyContent: 'center', flexShrink: 0}}>
              <MaterialCommunityIcons name="folder-open-outline" size={24} color="#FFFFFF" />
            </View>
            <View style={{flex: 1}}>
              <Text style={{fontSize: 15, fontWeight: '700', color: '#FFFFFF', marginBottom: 4}}>Visible in iOS Files App</Text>
              <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.85)', lineHeight: 19}}>
                Files → On My iPhone → AgriSense AI
              </Text>
              <Text style={{fontSize: 12, color: 'rgba(255,255,255,0.55)', marginTop: 3}}>
                4 JSON files · Share, open, or back up from Files
              </Text>
            </View>
          </View>
          <View style={{marginTop: 12, backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8}}>
            <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.7)', fontFamily: 'Menlo'}}>
              📂 On My iPhone / AgriSense AI/{'\n'}
              {'   '}agrisense_irrigation.json{'\n'}
              {'   '}agrisense_weather.json{'\n'}
              {'   '}agrisense_pest.json{'\n'}
              {'   '}agrisense_tips.json
            </Text>
          </View>
        </View>

        {/* ── Download cards ── */}
        <Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF', marginBottom: 12}}>Available Downloads</Text>

        {DOWNLOAD_CONFIG.map((cfg) => {
          const meta         = fileMeta[cfg.id];
          const isDownloaded = !!meta;
          const isLoading    = !!downloading[cfg.id];

          return (
            <View key={cfg.id}
              style={{backgroundColor: cfg.cardBg, borderRadius: 16, borderWidth: 1.5, borderColor: cfg.borderColor, padding: 16, marginBottom: 12}}>
              {/* Top row */}
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 12}}>
                <View style={{width: 52, height: 52, borderRadius: 14, backgroundColor: cfg.iconBg, alignItems: 'center', justifyContent: 'center'}}>
                  <MaterialCommunityIcons name={cfg.icon} size={26} color="#FFFFFF" />
                </View>
                <View style={{flex: 1}}>
                  <Text style={{fontSize: 16, fontWeight: '700', color: '#FFFFFF'}}>{cfg.title}</Text>
                  <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.8)', marginTop: 1}}>{cfg.subtitle}</Text>
                  <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.45)', marginTop: 1, fontFamily: 'Menlo'}}>{FILE_NAMES[cfg.id]}</Text>
                </View>
              </View>

              {/* Status + buttons */}
              <View style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 14}}>
                {/* Status */}
                <View style={{flexDirection: 'row', alignItems: 'center', gap: 5, flex: 1}}>
                  {isDownloaded ? (
                    <>
                      <MaterialCommunityIcons name="check-circle-outline" size={14} color="#22C55E" />
                      <Text style={{fontSize: 11, fontWeight: '600', color: '#22C55E'}}>{fmtDate(meta.savedAt)}</Text>
                      {meta.sizeKB > 0 ? (
                        <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.35)'}}> · {meta.sizeKB} KB</Text>
                      ) : null}
                    </>
                  ) : (
                    <Text style={{fontSize: 12, color: '#9CA3AF'}}>Not saved yet</Text>
                  )}
                </View>

                {/* Action buttons */}
                <View style={{flexDirection: 'row', gap: 8}}>
                  {/* Share */}
                  {isDownloaded ? (
                    <TouchableOpacity
                      onPress={() => handleShare(cfg.id)}
                      style={{width: 36, height: 36, borderRadius: 10, backgroundColor: 'rgba(255,255,255,0.12)', alignItems: 'center', justifyContent: 'center'}}>
                      <MaterialCommunityIcons name="share-outline" size={17} color="rgba(255,255,255,0.8)" />
                    </TouchableOpacity>
                  ) : null}

                  {/* Delete */}
                  {isDownloaded ? (
                    <TouchableOpacity
                      onPress={() => handleDelete(cfg.id)}
                      style={{width: 36, height: 36, borderRadius: 10, backgroundColor: 'rgba(239,68,68,0.15)', alignItems: 'center', justifyContent: 'center'}}>
                      <MaterialCommunityIcons name="delete-outline" size={17} color="#FCA5A5" />
                    </TouchableOpacity>
                  ) : null}

                  {/* Download / Update */}
                  <TouchableOpacity
                    onPress={() => handleDownload(cfg.id)}
                    disabled={isLoading}
                    style={{
                      flexDirection: 'row', alignItems: 'center', gap: 6,
                      backgroundColor: cfg.iconBg, borderRadius: 10,
                      paddingHorizontal: 14, paddingVertical: 9,
                      opacity: isLoading ? 0.6 : 1,
                    }}>
                    {isLoading
                      ? <ActivityIndicator size="small" color="#FFFFFF" />
                      : <MaterialCommunityIcons name={isDownloaded ? 'refresh' : 'download'} size={15} color="#FFFFFF" />}
                    <Text style={{fontSize: 13, fontWeight: '700', color: '#FFFFFF'}}>
                      {isLoading ? 'Saving…' : isDownloaded ? 'Update' : 'Download'}
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          );
        })}

        {/* ── Storage ── */}
        <Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF', marginTop: 8, marginBottom: 12}}>Storage</Text>
        <View style={{backgroundColor: 'rgba(147,51,234,0.22)', borderRadius: 16, padding: 16, borderWidth: 1.5, borderColor: 'rgba(196,181,253,0.4)', marginBottom: 24}}>
          <View style={{flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10}}>
            <Text style={{fontSize: 14, fontWeight: '600', color: 'rgba(255,255,255,0.85)'}}>Used Space</Text>
            <Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{usedMB} MB</Text>
          </View>
          <View style={{height: 8, borderRadius: 4, backgroundColor: 'rgba(88,28,135,0.5)', overflow: 'hidden'}}>
            <View style={{width: `${progressPct || 1}%`, height: '100%', borderRadius: 4, backgroundColor: '#C084FC'}} />
          </View>
          <Text style={{fontSize: 12, color: 'rgba(255,255,255,0.5)', marginTop: 8}}>
            {progressPct}% of {TOTAL_MB} MB · Files → On My iPhone → AgriSense AI
          </Text>
        </View>

        {/* ── How It Works ── */}
        <Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF', marginBottom: 12}}>
          How It Works
        </Text>
        <View style={{backgroundColor: 'rgba(109,40,217,0.26)', borderRadius: 16, padding: 16, borderWidth: 1.5, borderColor: 'rgba(168,85,247,0.4)', marginBottom: 20}}>
          {[
            {step: '1', icon: 'download',         title: 'Download',        desc: 'Tap Download to fetch live ML predictions and save them as JSON to your device.'},
            {step: '2', icon: 'folder-open',       title: 'Access in Files', desc: 'Open the iOS Files app → On My iPhone → AgriSense AI to view, copy, or share files.'},
            {step: '3', icon: 'share-variant',     title: 'Share',           desc: 'Tap the share button to send the JSON file via WhatsApp, email, or AirDrop.'},
            {step: '4', icon: 'refresh',           title: 'Keep Updated',    desc: 'Tap Update weekly to refresh predictions with the latest weather data.'},
          ].map((item, idx, arr) => (
            <View key={item.step} style={{flexDirection: 'row', gap: 12, alignItems: 'flex-start', marginBottom: idx < arr.length - 1 ? 16 : 0}}>
              <View style={{width: 32, height: 32, borderRadius: 16, backgroundColor: '#C084FC', alignItems: 'center', justifyContent: 'center', flexShrink: 0}}>
                <MaterialCommunityIcons name={item.icon} size={16} color="#FFFFFF" />
              </View>
              <View style={{flex: 1}}>
                <Text style={{fontSize: 14, fontWeight: '700', color: '#FFFFFF', marginBottom: 2}}>{item.title}</Text>
                <Text style={{fontSize: 13, color: 'rgba(255,255,255,0.7)', lineHeight: 19}}>{item.desc}</Text>
              </View>
            </View>
          ))}
        </View>

        {/* ── Download All ── */}
        <TouchableOpacity
          onPress={handleDownloadAll}
          style={{borderRadius: 16, paddingVertical: 16, alignItems: 'center',
            backgroundColor: '#8B5CF6', borderWidth: 1.5, borderColor: 'rgba(196,181,253,0.6)',
            flexDirection: 'row', justifyContent: 'center', gap: 8}}>
          <MaterialCommunityIcons name="download-multiple" size={20} color="#FFFFFF" />
          <Text style={{fontSize: 16, fontWeight: '700', color: '#FFFFFF'}}>
            Download All (सब डाउनलोड करें)
          </Text>
        </TouchableOpacity>

      </ScrollView>
    </SafeAreaView>
  );
};

export default OfflineDashboardScreen;
