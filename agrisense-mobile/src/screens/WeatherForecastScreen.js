import React, {useEffect, useState} from 'react';
import {
	ActivityIndicator,
	Pressable,
	SafeAreaView,
	ScrollView,
	StatusBar,
	Text,
	View,
} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {getWeatherCurrent, getWeatherForecast} from '../services/api';
import {getCurrentPosition, requestLocationPermission} from '../services/location';
import locationStore from '../services/locationStore';
import useTranslation from '../services/useTranslation';

const DEFAULT_LAT = 18.6650; // Bodhan, Nizamabad
const DEFAULT_LON = 77.9046;

// Map condition string → MaterialCommunityIcons name
const weatherIcon = (condition = '') => {
	const c = condition.toLowerCase();
	if (c.includes('thunder') || c.includes('storm')) return 'weather-lightning-rainy';
	if (c.includes('drizzle')) return 'weather-drizzle';
	if (c.includes('heavy rain')) return 'weather-pouring';
	if (c.includes('rain') || c.includes('shower')) return 'weather-rainy';
	if (c.includes('snow')) return 'weather-snowy';
	if (c.includes('mist') || c.includes('fog') || c.includes('haze')) return 'weather-fog';
	if (c.includes('overcast')) return 'weather-cloudy';
	if (c.includes('broken cloud') || c.includes('cloud')) return 'weather-partly-cloudy';
	if (c.includes('clear') || c.includes('sunny')) return 'weather-sunny';
	return 'weather-partly-cloudy';
};

// Parse a YYYY-MM-DD string correctly in local time (avoids UTC-midnight off-by-one)
const parseLocalDate = (dateStr) => {
	if (!dateStr) return new Date();
	const [y, m, d] = dateStr.split('-').map(Number);
	return new Date(y, m - 1, d);
};

// Build dynamic farming tips from live weather data
const buildFarmingTip = (current, forecastDays) => {
	const rainToday = (current?.rainfall_mm ?? 0) > 2;
	const rainTomorrow = forecastDays.length > 0 && (forecastDays[0]?.rainfall_mm ?? 0) > 5;
	const highTemp = (current?.temperature ?? 0) > 38;
	const highHumidity = (current?.humidity ?? 0) > 75;
	const tips = [];
	if (rainToday) tips.push('Rain detected today — skip irrigation to save water.');
	if (rainTomorrow) tips.push(`Rain of ${Math.round(forecastDays[0].rainfall_mm)} mm expected tomorrow — good time for transplanting.`);
	if (highTemp) tips.push(`High temperature (${current.temperature.toFixed(1)}°C) — water crops early morning to reduce stress.`);
	if (highHumidity) tips.push('High humidity detected — watch for fungal diseases on leaves.');
	if (!rainToday && !rainTomorrow && !highTemp && !highHumidity) {
		tips.push('Dry conditions expected — ensure adequate irrigation.');
		tips.push('Good time for field preparation and fertiliser application.');
	}
	return tips.slice(0, 3);
};

const WeatherForecastScreen = ({navigation}) => {
	const [current, setCurrent] = useState(null);
	const [forecastDays, setForecastDays] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);
	const [coords, setCoords] = useState(null);

	const {t} = useTranslation();

	// Resolve location
	useEffect(() => {
		const resolve = async () => {
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
		};
		resolve();
	}, []);

	useEffect(() => {
		if (!coords) return;
		const {lat, lon} = coords;
		Promise.all([getWeatherCurrent(lat, lon), getWeatherForecast(lat, lon)])
			.then(([cur, forecast]) => {
				setCurrent(cur);
				setForecastDays(forecast?.forecast ?? []);
			})
			.catch((e) => setError(e.message))
			.finally(() => setLoading(false));
	}, [coords]);

	const now = new Date();
	const todayLabel = now.toLocaleDateString('en-US', {month: 'long', day: 'numeric'});
	const temp = current?.temperature != null ? parseFloat(current.temperature.toFixed(1)) : 0;
	const condition = current?.condition ?? '';
	const wind = Math.round(current?.wind_speed ?? 0);
	const humidity = Math.round(current?.humidity ?? 0);
	const rain = Math.round(current?.rainfall_mm ?? 0);
	const locationName = current?.location_name ?? 'Bodhan, Nizamabad';

	// Hourly slots derived from today's weather (real current + trend from first forecast)
	const nextTemp = forecastDays.length > 0 ? parseFloat((forecastDays[0].temperature ?? temp).toFixed(1)) : temp;
	const hourlySlots = [
		{label: 'Now', temp, condition},
		{label: '2 PM', temp: parseFloat((temp + (nextTemp - temp) * 0.25).toFixed(1)), condition},
		{label: '4 PM', temp: parseFloat((temp + (nextTemp - temp) * 0.5).toFixed(1)), condition},
		{label: '6 PM', temp: parseFloat((temp + (nextTemp - temp) * 0.75).toFixed(1)), condition},
	];

	const farmingTips = buildFarmingTip(current, forecastDays);

	if (loading) {
		return (
			<SafeAreaView style={{flex: 1, backgroundColor: '#3B0F00'}}>
				<StatusBar barStyle="light-content" backgroundColor="#3B0F00" />
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12}}>
					<ActivityIndicator color="#E07000" size="large" />
					<Text style={{color: 'rgba(255,255,255,0.6)', fontSize: 14}}>Loading forecast…</Text>
				</View>
			</SafeAreaView>
		);
	}

	if (error) {
		return (
			<SafeAreaView style={{flex: 1, backgroundColor: '#3B0F00'}}>
				<StatusBar barStyle="light-content" backgroundColor="#3B0F00" />
				<View style={{flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 12}}>
					<Pressable onPress={() => navigation.goBack()} hitSlop={10} style={{width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 22}}>
						<MaterialCommunityIcons name="chevron-left" size={26} color="#FFFFFF" />
					</Pressable>
					<Text style={{color: '#FFFFFF', fontSize: 18, fontWeight: '700', marginLeft: 8}}>Weather Forecast</Text>
				</View>
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 32}}>
					<MaterialCommunityIcons name="wifi-off" size={40} color="#E07000" />
					<Text style={{color: 'rgba(255,255,255,0.7)', fontSize: 15, textAlign: 'center', marginTop: 12}}>Could not load forecast — check your connection.</Text>
				</View>
			</SafeAreaView>
		);
	}

	return (
		<SafeAreaView style={{flex: 1, backgroundColor: '#3B0F00'}}>
			<StatusBar barStyle="light-content" backgroundColor="#3B0F00" />

			{/* Header */}
			<View style={{flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingTop: 12, paddingBottom: 12, gap: 12}}>
				<Pressable onPress={() => navigation.goBack()} hitSlop={10} style={{width: 44, height: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.08)'}}>
					<MaterialCommunityIcons name="chevron-left" size={26} color="#FFFFFF" />
				</Pressable>
				<View style={{flex: 1}}>
					<Text style={{color: '#FFFFFF', fontSize: 18, fontWeight: '700'}}>Weather Forecast</Text>
					<View style={{flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2}}>
						<MaterialCommunityIcons name="map-marker" size={12} color="#FCD34D" />
						<Text style={{color: 'rgba(255,255,255,0.55)', fontSize: 12}}>{locationName}</Text>
					</View>
				</View>
			</View>

			<ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingHorizontal: 16, paddingBottom: 32, gap: 20}}>

				{/* Hero card */}
				<View style={{backgroundColor: '#E07000', borderRadius: 22, padding: 22, alignItems: 'center'}}>
					<Text style={{fontSize: 14, color: 'rgba(255,255,255,0.85)', fontWeight: '500'}}>
						{t('weather', 'today') || 'Today'}, {todayLabel}
					</Text>
					<MaterialCommunityIcons name={weatherIcon(condition)} size={60} color="#FFFFFF" style={{marginVertical: 10}} />
					<Text style={{fontSize: 64, fontWeight: '800', color: '#FFFFFF', lineHeight: 72}}>{temp}°C</Text>
					<Text style={{fontSize: 20, fontWeight: '600', color: '#FFFFFF', marginTop: 4}}>{condition || '—'}</Text>
					<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.65)', marginTop: 4, marginBottom: 18}}>
						{rain > 2 ? 'बारिश की संभावना' : humidity > 70 ? 'उमस और गर्म' : 'धूप और गर्म'}
					</Text>

					<View style={{flexDirection: 'row', width: '100%', gap: 10}}>
						{[
							{icon: 'weather-windy', label: 'Wind', value: `${wind} km/h`},
							{icon: 'water-percent', label: 'Humidity', value: `${humidity}%`},
							{icon: 'weather-rainy', label: 'Rain', value: `${rain} mm`},
						].map((metric) => (
							<View key={metric.label} style={{flex: 1, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 14, paddingVertical: 12, alignItems: 'center', gap: 4}}>
								<MaterialCommunityIcons name={metric.icon} size={20} color="#FFFFFF" />
								<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.8)'}}>{metric.label}</Text>
								<Text style={{fontSize: 13, color: '#FFFFFF', fontWeight: '700'}}>{metric.value}</Text>
							</View>
						))}
					</View>
				</View>

				{/* Hourly forecast */}
				<Text style={{fontSize: 17, fontWeight: '700', color: '#FFFFFF'}}>
					{t('weather', 'hourlyForecast') || 'Hourly Forecast'} (प्रति घंटा)
				</Text>
				<ScrollView horizontal showsHorizontalScrollIndicator={false} style={{marginHorizontal: -4}}>
					<View style={{flexDirection: 'row', gap: 12, paddingHorizontal: 4}}>
						{hourlySlots.map((slot) => (
							<View key={slot.label} style={{backgroundColor: '#4D1500', borderRadius: 16, paddingVertical: 18, paddingHorizontal: 20, alignItems: 'center', minWidth: 88}}>
								<Text style={{fontSize: 13, color: 'rgba(255,255,255,0.65)', fontWeight: '500'}}>{slot.label}</Text>
								<MaterialCommunityIcons name={weatherIcon(slot.condition)} size={32} color="#FFFFFF" style={{marginVertical: 8}} />
								<Text style={{fontSize: 22, fontWeight: '700', color: '#FFFFFF'}}>{slot.temp}°</Text>
							</View>
						))}
					</View>
				</ScrollView>

				{/* 7-day forecast */}
				<Text style={{fontSize: 17, fontWeight: '700', color: '#FFFFFF'}}>
					{t('weather', 'thisWeek') || 'This Week'} (इस सप्ताह)
				</Text>
				<View style={{gap: 8}}>
					{forecastDays.map((item) => {
						const dateObj = parseLocalDate(item.date);
						const isToday = item.date === now.toISOString().slice(0, 10);
						const dayName = isToday ? 'Today' : dateObj.toLocaleDateString('en-US', {weekday: 'long'});
						const dateStr = dateObj.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
						const rainy = /rain|shower|drizzle/i.test(item.condition ?? '');

						return (
							<View key={item.date} style={{backgroundColor: rainy ? '#2E1545' : '#4D1500', borderRadius: 14, flexDirection: 'row', alignItems: 'center', paddingVertical: 14, paddingHorizontal: 16, gap: 14}}>
								<View style={{width: 40, height: 40, borderRadius: 10, backgroundColor: 'rgba(255,255,255,0.08)', alignItems: 'center', justifyContent: 'center'}}>
									<MaterialCommunityIcons name={weatherIcon(item.condition ?? '')} size={24} color="#FFFFFF" />
								</View>
								<View style={{flex: 1}}>
									<Text style={{fontSize: 15, fontWeight: '700', color: '#FFFFFF'}}>{dayName}</Text>
									<Text style={{fontSize: 12, color: 'rgba(255,255,255,0.55)', marginTop: 2}}>{dateStr}</Text>
								</View>
								<View style={{alignItems: 'flex-end'}}>
									<Text style={{fontSize: 18, fontWeight: '700', color: '#FFFFFF'}}>{item.temperature != null ? parseFloat(item.temperature.toFixed(1)) : 0}°C</Text>
									{(item.rainfall_mm ?? 0) > 0 ? (
										<View style={{flexDirection: 'row', alignItems: 'center', gap: 3, marginTop: 2}}>
											<MaterialCommunityIcons name="weather-rainy" size={11} color="#93C5FD" />
											<Text style={{fontSize: 11, color: '#93C5FD'}}>{Math.round(item.rainfall_mm)} mm</Text>
										</View>
									) : (
										<Text style={{fontSize: 11, color: 'rgba(255,255,255,0.45)', marginTop: 2}}>{item.humidity ? `${Math.round(item.humidity)}% hum` : 'Dry'}</Text>
									)}
								</View>
							</View>
						);
					})}
				</View>

				{/* Live farming tips */}
				<View style={{backgroundColor: '#1A6B3A', borderRadius: 16, padding: 18, flexDirection: 'row', gap: 14, alignItems: 'flex-start'}}>
					<MaterialCommunityIcons name="lightbulb-on-outline" size={24} color="#FFFFFF" style={{marginTop: 2}} />
					<View style={{flex: 1}}>
						<Text style={{fontSize: 15, fontWeight: '700', color: '#FFFFFF', marginBottom: 8}}>
							{t('weather', 'farmingTip') || 'Farming Tip'} (सलाह)
						</Text>
						{farmingTips.map((tip, i) => (
							<Text key={i} style={{fontSize: 13, color: 'rgba(255,255,255,0.88)', lineHeight: 20, marginBottom: 4}}>
								• {tip}
							</Text>
						))}
					</View>
				</View>

			</ScrollView>
		</SafeAreaView>
	);
};

export default WeatherForecastScreen;
