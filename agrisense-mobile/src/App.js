import React, {useEffect, useState} from 'react';
import {ActivityIndicator, Text, View} from 'react-native';
import {GestureHandlerRootView} from 'react-native-gesture-handler';
import {NavigationContainer} from '@react-navigation/native';
import {createStackNavigator} from '@react-navigation/stack';
import {getCurrentFarmer, subscribe as subscribeFarmerSession} from './services/authStore';
import {bootstrapPushNotifications, teardownPushNotifications} from './services/pushNotifications';
import LoginScreen from './screens/LoginScreen';
import SetupScreen from './screens/SetupScreen';
import HomeScreen from './screens/HomeScreen';
import PestAlertScreen from './screens/PestAlertScreen';
import SmartIrrigationScreen from './screens/SmartIrrigationScreen';
import WeatherForecastScreen from './screens/WeatherForecastScreen';
import OfflineDashboardScreen from './screens/OfflineDashboardScreen';
import VoiceAssistantScreen from './screens/VoiceAssistantScreen';
import {requestLocationPermission, getCurrentPosition} from './services/location';
import locationStore from './services/locationStore';

const Stack = createStackNavigator();

const App = () => {
	const [initialRoute, setInitialRoute] = useState(null);
	const [currentSession, setCurrentSession] = useState(null);

	// On app start: request location permission and persist coordinates
	useEffect(() => {
		let mounted = true;
		const initLocation = async () => {
			try {
				const granted = await requestLocationPermission();
				if (!granted) return;
				const pos = await getCurrentPosition({enableHighAccuracy: true, timeout: 10000});
				if (mounted && pos?.coords) {
					await locationStore.setLocation({lat: pos.coords.latitude, lon: pos.coords.longitude});
				}
			} catch (err) {
				console.warn('location init error', err);
			}
		};

		initLocation();

		return () => {
			mounted = false;
		};
	}, []);

	useEffect(() => {
		let mounted = true;
		const loadSession = async () => {
			try {
				const session = await getCurrentFarmer();
				if (mounted) {
					setCurrentSession(session);
					setInitialRoute(session?.farmer?.id ? 'Home' : 'Login');
				}
			} catch (error) {
				if (mounted) {
					setCurrentSession(null);
					setInitialRoute('Login');
				}
			}
		};

		loadSession();
		const unsub = subscribeFarmerSession((session) => {
			setCurrentSession(session);
			if (session?.farmer?.id) setInitialRoute('Home');
		});
		return () => {
			mounted = false;
			if (typeof unsub === 'function') unsub();
		};
	}, []);

	useEffect(() => {
		if (!currentSession?.farmer?.id) return undefined;
		bootstrapPushNotifications(currentSession).catch((error) => {
			console.warn('Unable to bootstrap push notifications', error);
		});
		return () => {};
	}, [currentSession]);

	useEffect(() => {
		return () => {
			teardownPushNotifications();
		};
	}, []);

	if (!initialRoute) {
		return (
			<GestureHandlerRootView style={{flex: 1}}>
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#063B24'}}>
					<ActivityIndicator size="large" color="#0ABF71" />
					<Text style={{marginTop: 12, color: '#E7F7ED'}}>Starting app...</Text>
				</View>
			</GestureHandlerRootView>
		);
	}

	return (
		<GestureHandlerRootView style={{flex: 1}}>
				<NavigationContainer>
					<Stack.Navigator
					initialRouteName={initialRoute}
						screenOptions={{headerShown: false}}>
					<Stack.Screen name="Login" component={LoginScreen} />
						<Stack.Screen name="Setup" component={SetupScreen} />
						<Stack.Screen name="Home" component={HomeScreen} />
						<Stack.Screen name="PestAlert" component={PestAlertScreen} />
						<Stack.Screen
							name="SmartIrrigation"
							component={SmartIrrigationScreen}
						/>
						<Stack.Screen
							name="WeatherForecast"
							component={WeatherForecastScreen}
						/>
						<Stack.Screen
							name="OfflineDashboard"
							component={OfflineDashboardScreen}
						/>
						<Stack.Screen
							name="VoiceAssistant"
							component={VoiceAssistantScreen}
						/>
					</Stack.Navigator>
				</NavigationContainer>
		</GestureHandlerRootView>
	);
};

export default App;
