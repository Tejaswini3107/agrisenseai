import React, {useEffect, useRef, useState} from 'react';
import {
	ActivityIndicator,
	Modal,
	Pressable,
	SafeAreaView,
	ScrollView,
	StatusBar,
	Text,
	View,
	TextInput,
} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {buildIrrigationPayloadFromInputs, getCropCatalog, getDashboardWeather, predictIrrigation} from '../services/api';
import locationStore from '../services/locationStore';
import {getCurrentPosition, requestLocationPermission} from '../services/location';
import BottomNav from '../components/BottomNav';
import cropStore from '../services/cropStore';
import useTranslation from '../services/useTranslation';

const APP_TITLE = 'AgriSense AI';
const FALLBACK_CROPS = ['rice', 'wheat', 'cotton', 'maize', 'turmeric', 'sugarcane', 'soybean', 'potato', 'onion', 'garlic'];
const LANGUAGE_OPTIONS = [
	{key: 'english', label: 'English'},
	{key: 'hindi',   label: 'हिंदी'},
	{key: 'telugu',  label: 'తెలుగు'},
	{key: 'marathi', label: 'मराठी'},
	{key: 'assamese', label: 'অসমীয়া'},
	{key: 'bengali',  label: 'বাংলা'},
	{key: 'gujarati', label: 'ગુજરાતી'},
	{key: 'kannada',  label: 'ಕನ್ನಡ'},
	{key: 'malayalam',label: 'മലയാളം'},
	{key: 'punjabi',  label: 'ਪੰਜਾਬੀ'},
	{key: 'tamil',    label: 'தமிழ்'},
	{key: 'odia',     label: 'ଓଡ଼ିଆ'},
	{key: 'urdu',     label: 'اردو'},
	{key: 'arabic',   label: 'العربية'},
	{key: 'french',   label: 'Français'},
];

const HomeScreen = ({navigation}) => {
	const [selectedCrop, setSelectedCrop] = useState('rice');
	const [selectedLanguage, setSelectedLanguage] = useState('english');
	const [dashboardWeather, setDashboardWeather] = useState(null);
	const [irrigationDecision, setIrrigationDecision] = useState(null); // {shouldWater, waterMm, etcPerDay}
	const [locationModalVisible, setLocationModalVisible] = useState(false);
	const [locationLatInput, setLocationLatInput] = useState('');
	const [locationLonInput, setLocationLonInput] = useState('');
	const [currentLocation, setCurrentLocation] = useState(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);
	const [showCropPicker, setShowCropPicker] = useState(false);
	const [activeDropdown, setActiveDropdown] = useState(null);
	const [cropOptions, setCropOptions] = useState(FALLBACK_CROPS);

	const {t, language} = useTranslation();

	// Keep a ref to latest crop so the weather loader always uses the freshest value
	const cropRef = useRef(selectedCrop);
	useEffect(() => { cropRef.current = selectedCrop; }, [selectedCrop]);

	// ── Load crop catalog ─────────────────────────────────────────
	useEffect(() => {
		getCropCatalog()
			.then((res) => {
				const list = Array.isArray(res?.crops) ? res.crops.map((c) => c.key).filter(Boolean) : [];
				if (list.length > 0) setCropOptions(list);
			})
			.catch(() => {});
	}, []);

	// ── Load & sync crop + language from store ────────────────────
	useEffect(() => {
		(async () => {
			const [crop, lang] = await Promise.all([cropStore.getCrop(), cropStore.getLanguage()]);
			if (crop) { setSelectedCrop(crop); cropRef.current = crop; }
			if (lang) setSelectedLanguage(lang);
			if (!crop || !lang) {
				setActiveDropdown(!crop ? 'crop' : 'language');
				setShowCropPicker(true);
			}
		})();
		const unsubCrop = cropStore.subscribe((v) => { if (v) { setSelectedCrop(v); cropRef.current = v; } });
		const unsubLang = cropStore.subscribeLanguage((v) => setSelectedLanguage(v || 'english'));
		return () => { unsubCrop(); unsubLang(); };
	}, []);

	// ── Fetch weather + irrigation decision ──────────────────────
	const loadAll = async (coords, crop) => {
		const lat = coords?.lat;
		const lon = coords?.lon;
		try {
			setLoading(true);
			setError(null);
			const weather = await getDashboardWeather(lat, lon);
			setDashboardWeather(weather);

			// Irrigation decision for the selected crop
			const irriPayload = buildIrrigationPayloadFromInputs({
				weather,
				cropType: crop ?? cropRef.current ?? 'rice',
				growthStage: 'vegetative',
				daysSinceIrrigation: 4,
				soilFeel: 'moist',
			});
			const irriResult = await predictIrrigation(irriPayload);
			setIrrigationDecision({
				shouldWater: (irriResult.irrigate ?? '').toLowerCase() === 'yes',
				waterMm: irriResult.water_amount_mm ?? 0,
				etcPerDay: irriResult.etc_mm_per_day ?? 0,
			});
		} catch (e) {
			console.error('Home load error', e);
			setError('Unable to load live data');
		} finally {
			setLoading(false);
		}
	};

	// Initial load from saved location
	useEffect(() => {
		let mounted = true;
		locationStore.getLocation().then((loc) => {
			if (mounted) { setCurrentLocation(loc); loadAll(loc, cropRef.current); }
		}).catch(() => { if (mounted) loadAll(null, cropRef.current); });

		const unsub = locationStore.subscribe((loc) => {
			setCurrentLocation(loc);
			loadAll(loc, cropRef.current);
		});
		return () => { mounted = false; if (typeof unsub === 'function') unsub(); };
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	// Refetch when crop changes (crop affects irrigation + ML predictions)
	useEffect(() => {
		if (!loading) {
			locationStore.getLocation().then((loc) => loadAll(loc, selectedCrop)).catch(() => loadAll(null, selectedCrop));
		}
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [selectedCrop]);

	const todayWeather       = dashboardWeather?.today ?? {};
	const diseaseRisk        = dashboardWeather?.disease_risk ?? '—';
	const diseaseConf        = dashboardWeather?.disease_confidence != null ? Math.round(dashboardWeather.disease_confidence * 100) : null;
	const irrigationL        = dashboardWeather?.irrigation_need_litres ?? '—';
	const climateRisk        = dashboardWeather?.climate_risk ?? '—';
	const climateConf        = dashboardWeather?.climate_confidence != null ? Math.round(dashboardWeather.climate_confidence * 100) : null;
	const locationName       = dashboardWeather?.location_name ?? '';
	const recommendedCrop    = dashboardWeather?.recommended_crop ?? null;
	const cropConfidence     = dashboardWeather?.crop_confidence != null ? Math.round(dashboardWeather.crop_confidence * 100) : null;
	const plantStress        = dashboardWeather?.plant_stress ?? null;
	const stressConfidence   = dashboardWeather?.stress_confidence != null ? Math.round(dashboardWeather.stress_confidence * 100) : null;
	const expectedYield      = dashboardWeather?.expected_yield_tons ?? null;

	if (loading) {
		return (
			<SafeAreaView style={{flex: 1, backgroundColor: '#045E3A'}}>
				<StatusBar barStyle="light-content" backgroundColor="#045E3A" />
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12}}>
					<ActivityIndicator size="large" color="#0ABF71" />
					<Text style={{fontSize: 14, color: '#E7F7ED'}}>{t('home', 'loading')}</Text>
				</View>
			</SafeAreaView>
		);
	}

	if (error) {
		return (
			<SafeAreaView style={{flex: 1, backgroundColor: '#045E3A'}}>
				<StatusBar barStyle="light-content" backgroundColor="#045E3A" />
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24, gap: 10}}>
					<MaterialCommunityIcons name="wifi-off" size={40} color="#86EFAC" />
					<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{t('home', 'liveDataUnavailable')}</Text>
					<Text style={{fontSize: 13, color: '#E7F7ED', textAlign: 'center'}}>{error}</Text>
				</View>
			</SafeAreaView>
		);
	}

	const shouldWater = irrigationDecision?.shouldWater ?? false;
	const waterMm     = irrigationDecision?.waterMm ?? 0;

	return (
		<SafeAreaView style={{flex: 1, backgroundColor: '#045E3A'}}>
			<StatusBar barStyle="light-content" backgroundColor="#045E3A" />

			<ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 16, paddingTop: 10, paddingBottom: 24}}>

				{/* ── Header ── */}
				<View style={{flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingTop: 4, paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: '#1A7A55'}}>
					<View style={{flexDirection: 'row', alignItems: 'center', gap: 8}}>
						<View style={{width: 32, height: 32, borderRadius: 8, backgroundColor: '#0ABF71', alignItems: 'center', justifyContent: 'center'}}>
							<MaterialCommunityIcons name="sprout" size={18} color="#FFFFFF" />
						</View>
						<Text style={{fontSize: 20, fontWeight: '700', color: '#EAFBF2'}}>{APP_TITLE}</Text>
					</View>

					<View style={{flexDirection: 'row', alignItems: 'center', gap: 8}}>
						<Pressable
							onPress={() => { setLocationModalVisible(true); setLocationLatInput(currentLocation?.lat ? String(currentLocation.lat) : ''); setLocationLonInput(currentLocation?.lon ? String(currentLocation.lon) : ''); }}
							style={{width: 34, height: 34, borderRadius: 17, backgroundColor: '#0B8A55', alignItems: 'center', justifyContent: 'center'}}>
							<MaterialCommunityIcons name="map-marker" size={16} color={currentLocation ? '#6EE7B7' : '#9CA3AF'} />
						</Pressable>

						<Pressable
							onPress={() => { setActiveDropdown('language'); setShowCropPicker(true); }}
							style={{flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 20, paddingHorizontal: 10, paddingVertical: 6, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)'}}>
							<MaterialCommunityIcons name="web" size={14} color="#A7F3D0" />
							<Text style={{fontSize: 12, fontWeight: '600', color: '#D9FCE9'}}>
								{LANGUAGE_OPTIONS.find((l) => l.key === selectedLanguage)?.label ?? 'EN'}
							</Text>
							<MaterialCommunityIcons name="chevron-down" size={13} color="#A7F3D0" />
						</Pressable>

						<Pressable
							onPress={() => { setActiveDropdown('crop'); setShowCropPicker(true); }}
							style={{width: 34, height: 34, borderRadius: 17, backgroundColor: '#16CF79', alignItems: 'center', justifyContent: 'center'}}>
							<Text style={{color: '#FFFFFF', fontWeight: '700', fontSize: 14}}>
								{(selectedCrop ?? 'A')[0].toUpperCase()}
							</Text>
						</Pressable>
					</View>
				</View>

				{/* ── Greeting ── */}
				<Text style={{marginTop: 14, fontSize: 32, color: '#E9FAF0', fontWeight: '700'}}>{t('home', 'namaste')} 🌾</Text>
				{locationName ? (
					<View style={{flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 4}}>
						<MaterialCommunityIcons name="map-marker-outline" size={13} color="#6EE7B7" />
						<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.6)'}}>{locationName}</Text>
					</View>
				) : null}

				{/* ── Irrigation Decision Card (full-width row) ── */}
				<Pressable
					onPress={() => navigation.navigate('SmartIrrigation')}
					style={{
						marginTop: 18,
						borderRadius: 18,
						padding: 18,
						backgroundColor: shouldWater ? '#15803D' : '#1E3A5F',
						borderWidth: 1.5,
						borderColor: shouldWater ? '#4ADE80' : '#3B82F6',
						flexDirection: 'row',
						alignItems: 'center',
						gap: 14,
					}}>
					<View style={{width: 52, height: 52, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.15)', alignItems: 'center', justifyContent: 'center'}}>
						<MaterialCommunityIcons name={shouldWater ? 'water' : 'hand-back-right-off'} size={28} color="#FFFFFF" />
					</View>
					<View style={{flex: 1}}>
						<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.6)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.8}}>
							Today's Irrigation · {(selectedCrop ?? 'rice').charAt(0).toUpperCase() + (selectedCrop ?? 'rice').slice(1)}
						</Text>
						<Text style={{fontSize: 22, fontWeight: '800', color: '#FFFFFF', marginTop: 2}}>
							{shouldWater ? t('irrigation', 'waterNow') : t('irrigation', 'dontWater')}
						</Text>
						{shouldWater && waterMm > 0 ? (
							<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.75)', marginTop: 2}}>Apply {waterMm} mm · tap for full schedule</Text>
						) : (
							<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.65)', marginTop: 2}}>Soil moisture adequate · tap for details</Text>
						)}
					</View>
					<MaterialCommunityIcons name="chevron-right" size={22} color="rgba(255,255,255,0.5)" />
				</Pressable>

				{/* ── Feature tiles ── */}
				<View style={{marginTop: 16, flexDirection: 'row', flexWrap: 'wrap', gap: 12}}>

					<Pressable onPress={() => navigation.navigate('PestAlert')}
						style={{width: '47.5%', minHeight: 160, borderRadius: 18, padding: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)', justifyContent: 'flex-end', backgroundColor: '#E15D2A'}}>
						<MaterialCommunityIcons name="bug-outline" size={38} color="#FFFFFF" style={{marginBottom: 10}} />
						<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{t('pest', 'title')}</Text>
						<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.85)', marginTop: 3}}>{diseaseRisk} risk</Text>
						{diseaseConf != null ? <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.65)', marginTop: 2}}>{diseaseConf}% confidence</Text> : null}
					</Pressable>

					<Pressable onPress={() => navigation.navigate('WeatherForecast')}
						style={{width: '47.5%', minHeight: 160, borderRadius: 18, padding: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)', justifyContent: 'flex-end', backgroundColor: '#EAA11E'}}>
						<MaterialCommunityIcons name="weather-partly-cloudy" size={38} color="#FFFFFF" style={{marginBottom: 10}} />
						<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{t('weather', 'title')}</Text>
						<Text style={{fontSize: 20, fontWeight: '800', color: '#FFFFFF', marginTop: 2}}>
							{todayWeather.temperatureC ?? '—'}°C
						</Text>
						<Text style={{fontSize: 12, color: 'rgba(255,255,255,0.85)', marginTop: 2}} numberOfLines={1}>{todayWeather.condition ?? '—'}</Text>
					</Pressable>

					<Pressable onPress={() => navigation.navigate('SmartIrrigation')}
						style={{width: '47.5%', minHeight: 160, borderRadius: 18, padding: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)', justifyContent: 'flex-end', backgroundColor: '#1D8FE3'}}>
						<MaterialCommunityIcons name="water-percent" size={38} color="#FFFFFF" style={{marginBottom: 10}} />
						<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{t('irrigation', 'title')}</Text>
						<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.85)', marginTop: 3}}>{irrigationL} L needed</Text>
						<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.7)', marginTop: 2}}>ML Powered</Text>
					</Pressable>

					<Pressable onPress={() => navigation.navigate('OfflineDashboard')}
						style={{width: '47.5%', minHeight: 160, borderRadius: 18, padding: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)', justifyContent: 'flex-end', backgroundColor: '#7C3AED'}}>
						<MaterialCommunityIcons name="download-outline" size={38} color="#FFFFFF" style={{marginBottom: 10}} />
						<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{t('offline', 'title')}</Text>
						<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.85)', marginTop: 3}}>{climateRisk} climate</Text>
						{climateConf != null ? <Text style={{fontSize: 11, color: 'rgba(255,255,255,0.65)', marginTop: 2}}>{climateConf}% confidence</Text> : null}
					</Pressable>
				</View>

				{/* ── Farm Intelligence Card ── */}
				{(recommendedCrop || plantStress || expectedYield != null) ? (
					<View style={{marginTop: 16, borderRadius: 18, backgroundColor: 'rgba(10,191,113,0.12)', borderWidth: 1.5, borderColor: 'rgba(10,191,113,0.3)', padding: 16}}>
						<View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14}}>
							<MaterialCommunityIcons name="brain" size={18} color="#0ABF71" />
							<Text style={{fontSize: 15, fontWeight: '700', color: '#EAFBF2'}}>Farm Intelligence</Text>
							<View style={{flex: 1}} />
							<View style={{backgroundColor: 'rgba(10,191,113,0.2)', borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3}}>
								<Text style={{fontSize: 10, fontWeight: '700', color: '#0ABF71'}}>ML LIVE</Text>
							</View>
						</View>

						<View style={{gap: 10}}>

							{/* ML Recommended Crop */}
							{recommendedCrop ? (
								<View style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.07)', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12}}>
									<View style={{flexDirection: 'row', alignItems: 'center', gap: 10}}>
										<View style={{width: 34, height: 34, borderRadius: 9, backgroundColor: 'rgba(234,161,30,0.2)', alignItems: 'center', justifyContent: 'center'}}>
											<MaterialCommunityIcons name="barley" size={18} color="#EAA11E" />
										</View>
										<View>
											<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.5)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4}}>ML Crop Suggestion</Text>
											<Text style={{fontSize: 15, fontWeight: '700', color: '#FFFFFF', marginTop: 2, textTransform: 'capitalize'}}>{recommendedCrop}</Text>
										</View>
									</View>
									{cropConfidence != null ? (
										<View style={{alignItems: 'flex-end'}}>
											<Text style={{fontSize: 18, fontWeight: '800', color: '#0ABF71'}}>{cropConfidence}%</Text>
											<Text style={{fontSize: 10, color: 'rgba(255,255,255,0.4)', marginTop: 1}}>confidence</Text>
										</View>
									) : null}
								</View>
							) : null}

							{/* Plant Stress */}
							{plantStress ? (
								<View style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.07)', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12}}>
									<View style={{flexDirection: 'row', alignItems: 'center', gap: 10}}>
										<View style={{width: 34, height: 34, borderRadius: 9,
											backgroundColor: plantStress.toLowerCase() === 'high' ? 'rgba(239,68,68,0.2)' : plantStress.toLowerCase() === 'medium' ? 'rgba(249,115,22,0.2)' : 'rgba(34,197,94,0.2)',
											alignItems: 'center', justifyContent: 'center'}}>
											<MaterialCommunityIcons name="leaf" size={18}
												color={plantStress.toLowerCase() === 'high' ? '#EF4444' : plantStress.toLowerCase() === 'medium' ? '#F97316' : '#22C55E'} />
										</View>
										<View>
											<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.5)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4}}>Plant Stress</Text>
											<Text style={{fontSize: 15, fontWeight: '700', marginTop: 2,
												color: plantStress.toLowerCase() === 'high' ? '#FCA5A5' : plantStress.toLowerCase() === 'medium' ? '#FDBA74' : '#86EFAC'}}>
												{plantStress}
											</Text>
										</View>
									</View>
									{stressConfidence != null ? (
										<View style={{alignItems: 'flex-end'}}>
											<Text style={{fontSize: 18, fontWeight: '800', color: '#A78BFA'}}>{stressConfidence}%</Text>
											<Text style={{fontSize: 10, color: 'rgba(255,255,255,0.4)', marginTop: 1}}>confidence</Text>
										</View>
									) : null}
								</View>
							) : null}

							{/* Expected Yield */}
							{expectedYield != null ? (
								<View style={{flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.07)', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12}}>
									<View style={{flexDirection: 'row', alignItems: 'center', gap: 10}}>
										<View style={{width: 34, height: 34, borderRadius: 9, backgroundColor: 'rgba(29,143,227,0.2)', alignItems: 'center', justifyContent: 'center'}}>
											<MaterialCommunityIcons name="chart-line" size={18} color="#1D8FE3" />
										</View>
										<View>
											<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.5)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4}}>Expected Yield</Text>
											<Text style={{fontSize: 15, fontWeight: '700', color: '#FFFFFF', marginTop: 2}}>{expectedYield} tons/ha</Text>
										</View>
									</View>
									<View style={{alignItems: 'flex-end'}}>
										<Text style={{fontSize: 18, fontWeight: '800', color: '#60A5FA'}}>{expectedYield}t</Text>
										<Text style={{fontSize: 10, color: 'rgba(255,255,255,0.4)', marginTop: 1}}>per hectare</Text>
									</View>
								</View>
							) : null}

						</View>
					</View>
				) : null}

				{/* ── Voice button ── */}
				<Pressable onPress={() => navigation.navigate('VoiceAssistant')}
					style={{alignSelf: 'center', marginTop: 22, marginBottom: 4, width: 100, height: 100, alignItems: 'center', justifyContent: 'center'}}>
					<View style={{position: 'absolute', width: 100, height: 100, borderRadius: 50, backgroundColor: '#40D37D', opacity: 0.2}} />
					<View style={{width: 88, height: 88, borderRadius: 44, borderWidth: 2, borderColor: '#14D17C', backgroundColor: '#0EC26F', alignItems: 'center', justifyContent: 'center'}}>
						<MaterialCommunityIcons name="microphone" size={30} color="#FFFFFF" />
						<Text style={{marginTop: 2, fontSize: 11, color: '#F2FFF8', fontWeight: '700'}}>{t('home', 'tapToAsk')}</Text>
					</View>
				</Pressable>

			</ScrollView>

			{/* ── Crop / Language picker modal ── */}
			<Modal visible={showCropPicker} transparent animationType="fade" onRequestClose={() => setShowCropPicker(false)}>
				<View style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center', padding: 20}}>
					<View style={{width: '100%', borderRadius: 20, padding: 18, backgroundColor: '#F7FFF9'}}>
						<Text style={{fontSize: 20, fontWeight: '700', color: '#063B24'}}>{t('home', 'chooseCropAndLanguage')}</Text>
						<Text style={{marginTop: 6, fontSize: 13, color: '#4B6A57'}}>{t('home', 'modalSubtitle')}</Text>

						{/* Crop selector */}
						<Pressable
							onPress={() => setActiveDropdown((d) => (d === 'crop' ? null : 'crop'))}
							style={{marginTop: 14, paddingHorizontal: 14, paddingVertical: 12, borderRadius: 12, borderWidth: 1, borderColor: '#B6DCC6', backgroundColor: '#FFFFFF', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'}}>
							<Text style={{fontSize: 13, fontWeight: '700', color: '#4B6A57'}}>{t('home', 'crop')}</Text>
							<Text style={{flex: 1, fontSize: 14, color: '#0B3B24', fontWeight: '700', textAlign: 'right', marginLeft: 8, textTransform: 'capitalize'}}>{selectedCrop ?? t('home', 'selectCrop')}</Text>
							<MaterialCommunityIcons name={activeDropdown === 'crop' ? 'chevron-up' : 'chevron-down'} size={18} color="#0B8A55" style={{marginLeft: 6}} />
						</Pressable>
						{activeDropdown === 'crop' ? (
							<ScrollView style={{maxHeight: 200, marginTop: 6, borderRadius: 12, borderWidth: 1, borderColor: '#D6E9DC', backgroundColor: '#F8FFFA'}}>
								{cropOptions.map((crop) => (
									<Pressable key={crop} onPress={async () => { await cropStore.setCrop(crop); setActiveDropdown(null); }}
										style={{paddingHorizontal: 14, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#E5F2E9', backgroundColor: selectedCrop === crop ? '#0ABF71' : 'transparent'}}>
										<Text style={{fontSize: 14, fontWeight: '600', textTransform: 'capitalize', color: selectedCrop === crop ? '#FFFFFF' : '#0B3B24'}}>{crop}</Text>
									</Pressable>
								))}
							</ScrollView>
						) : null}

						{/* Language selector */}
						<Pressable
							onPress={() => setActiveDropdown((d) => (d === 'language' ? null : 'language'))}
							style={{marginTop: 10, paddingHorizontal: 14, paddingVertical: 12, borderRadius: 12, borderWidth: 1, borderColor: '#B6DCC6', backgroundColor: '#FFFFFF', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'}}>
							<Text style={{fontSize: 13, fontWeight: '700', color: '#4B6A57'}}>{t('home', 'language')}</Text>
							<Text style={{flex: 1, fontSize: 14, color: '#0B3B24', fontWeight: '700', textAlign: 'right', marginLeft: 8}}>
								{LANGUAGE_OPTIONS.find((l) => l.key === selectedLanguage)?.label ?? t('home', 'selectLanguage')}
							</Text>
							<MaterialCommunityIcons name={activeDropdown === 'language' ? 'chevron-up' : 'chevron-down'} size={18} color="#0B8A55" style={{marginLeft: 6}} />
						</Pressable>
						{activeDropdown === 'language' ? (
							<ScrollView style={{maxHeight: 200, marginTop: 6, borderRadius: 12, borderWidth: 1, borderColor: '#D6E9DC', backgroundColor: '#F8FFFA'}}>
								{LANGUAGE_OPTIONS.map((lang) => (
									<Pressable key={lang.key} onPress={async () => { await cropStore.setLanguage(lang.key); setActiveDropdown(null); }}
										style={{paddingHorizontal: 14, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#E5F2E9', backgroundColor: selectedLanguage === lang.key ? '#0ABF71' : 'transparent'}}>
										<Text style={{fontSize: 14, fontWeight: '600', color: selectedLanguage === lang.key ? '#FFFFFF' : '#0B3B24'}}>{lang.label}</Text>
									</Pressable>
								))}
							</ScrollView>
						) : null}

						<Pressable onPress={() => setShowCropPicker(false)}
							style={{marginTop: 18, backgroundColor: '#0ABF71', borderRadius: 12, paddingVertical: 14, alignItems: 'center'}}>
							<Text style={{fontSize: 15, color: '#FFFFFF', fontWeight: '700'}}>{t('home', 'done')}</Text>
						</Pressable>
					</View>
				</View>
			</Modal>

			{/* ── Location modal ── */}
			<Modal visible={locationModalVisible} transparent animationType="fade" onRequestClose={() => setLocationModalVisible(false)}>
				<View style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center', padding: 20}}>
					<View style={{width: '100%', borderRadius: 20, padding: 18, backgroundColor: '#F7FFF9'}}>
						<Text style={{fontSize: 18, fontWeight: '700', color: '#063B24', marginBottom: 6}}>Set Location</Text>
						<Text style={{fontSize: 13, color: '#4B6A57', marginBottom: 14}}>Enter coordinates or use device GPS</Text>
						<TextInput placeholder="Latitude" value={locationLatInput} onChangeText={setLocationLatInput} keyboardType="numeric"
							style={{borderWidth: 1, borderColor: '#D6E9DC', borderRadius: 10, padding: 12, marginBottom: 10, fontSize: 14}} />
						<TextInput placeholder="Longitude" value={locationLonInput} onChangeText={setLocationLonInput} keyboardType="numeric"
							style={{borderWidth: 1, borderColor: '#D6E9DC', borderRadius: 10, padding: 12, fontSize: 14}} />
						<View style={{flexDirection: 'row', gap: 10, marginTop: 16}}>
							<Pressable onPress={async () => {
								const lat = parseFloat(locationLatInput);
								const lon = parseFloat(locationLonInput);
								if (Number.isFinite(lat) && Number.isFinite(lon)) await locationStore.setLocation({lat, lon});
								setLocationModalVisible(false);
							}} style={{flex: 1, backgroundColor: '#0ABF71', borderRadius: 10, paddingVertical: 12, alignItems: 'center'}}>
								<Text style={{color: '#FFFFFF', fontWeight: '700'}}>Save</Text>
							</Pressable>
							<Pressable onPress={async () => {
								try {
									await requestLocationPermission();
									const pos = await getCurrentPosition({enableHighAccuracy: true, timeout: 10000});
									if (pos?.coords) await locationStore.setLocation({lat: pos.coords.latitude, lon: pos.coords.longitude});
								} catch (e) { console.warn('GPS error', e); }
								setLocationModalVisible(false);
							}} style={{flex: 1, backgroundColor: '#1D8FE3', borderRadius: 10, paddingVertical: 12, alignItems: 'center'}}>
								<Text style={{color: '#FFFFFF', fontWeight: '700'}}>Use GPS</Text>
							</Pressable>
							<Pressable onPress={() => setLocationModalVisible(false)}
								style={{paddingVertical: 12, paddingHorizontal: 14, alignItems: 'center', justifyContent: 'center'}}>
								<Text style={{color: '#0B8A55', fontWeight: '700'}}>{t('home', 'cancel')}</Text>
							</Pressable>
						</View>
					</View>
				</View>
			</Modal>

			{/* <BottomNav /> */}
		</SafeAreaView>
	);
};

export default HomeScreen;
