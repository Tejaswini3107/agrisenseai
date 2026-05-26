import React, {useEffect, useMemo, useState} from 'react';
import {ActivityIndicator, Pressable, SafeAreaView, ScrollView, StatusBar, Text, View} from 'react-native';
import {GoogleSignin, statusCodes} from '@react-native-google-signin/google-signin';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {googleSignInFarmer} from '../services/api';
import {GOOGLE_IOS_CLIENT_ID, GOOGLE_WEB_CLIENT_ID, isGoogleAuthConfigured} from '../services/googleAuthConfig';
import {setCurrentFarmer} from '../services/authStore';

const LoginScreen = ({navigation}) => {
	const [loading, setLoading] = useState(true);
	const [signingIn, setSigningIn] = useState(false);
	const [error, setError] = useState(null);

	const configured = useMemo(() => isGoogleAuthConfigured(), []);

	useEffect(() => {
		const load = async () => {
			try {
				GoogleSignin.configure({
					webClientId: GOOGLE_WEB_CLIENT_ID,
					iosClientId: GOOGLE_IOS_CLIENT_ID,
					offlineAccess: false,
				});
			} catch (configureError) {
				console.warn('Google Sign-In config failed', configureError);
			}
			setLoading(false);
		};
		load();
	}, []);

	const handleLogin = async () => {
		if (!configured) {
			setError('Google Sign-In is not configured. Add OAuth client IDs first.');
			return;
		}
		setSigningIn(true);
		setError(null);
		try {
			await GoogleSignin.hasPlayServices({showPlayServicesUpdateDialog: true});
			const userInfo = await GoogleSignin.signIn();
			const tokenBundle = await GoogleSignin.getTokens();
			const idToken = tokenBundle?.idToken || userInfo?.idToken;
			if (!idToken) throw new Error('Google Sign-In did not return an ID token');
			const response = await googleSignInFarmer({idToken, crop: 'sugarcane'});
			await setCurrentFarmer({farmer: response.farmer, idToken});
			navigation.replace('Setup');
		} catch (loginError) {
			if (loginError?.code === statusCodes.SIGN_IN_CANCELLED) {
				setError('Sign-in was cancelled.');
			} else if (loginError?.code === statusCodes.PLAY_SERVICES_NOT_AVAILABLE) {
				setError('Google Play Services not available.');
			} else {
				setError(loginError?.message || 'Unable to sign in with Google.');
			}
			console.error('Google login failed', loginError);
		} finally {
			setSigningIn(false);
		}
	};

	const handleSkip = () => {
		navigation.replace('Setup');
	};

	if (loading) {
		return (
			<SafeAreaView style={{flex: 1, backgroundColor: '#063B24'}}>
				<StatusBar barStyle="light-content" backgroundColor="#063B24" />
				<View style={{flex: 1, alignItems: 'center', justifyContent: 'center'}}>
					<ActivityIndicator size="large" color="#0ABF71" />
				</View>
			</SafeAreaView>
		);
	}

	return (
		<SafeAreaView style={{flex: 1, backgroundColor: '#063B24'}}>
			<StatusBar barStyle="light-content" backgroundColor="#063B24" />
			<ScrollView contentContainerStyle={{flexGrow: 1, padding: 24, justifyContent: 'center'}}>
				{/* Logo & Title */}
				<View style={{alignItems: 'center', marginBottom: 40}}>
					<View style={{width: 88, height: 88, borderRadius: 26, backgroundColor: '#0ABF71', alignItems: 'center', justifyContent: 'center', marginBottom: 16}}>
						<MaterialCommunityIcons name="sprout" size={44} color="#FFFFFF" />
					</View>
					<Text style={{fontSize: 32, fontWeight: '800', color: '#FFFFFF', letterSpacing: 0.5}}>AgriSense AI</Text>
					<Text style={{marginTop: 8, fontSize: 15, color: '#A7D8BA', textAlign: 'center', lineHeight: 22}}>
						Your smart farming assistant.{'\n'}Sign in to save your profile & history.
					</Text>
				</View>

				{/* Card */}
				<View style={{borderRadius: 24, backgroundColor: '#F7FFF9', padding: 24, shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 20, elevation: 8}}>
					{error ? (
						<View style={{marginBottom: 16, borderRadius: 12, backgroundColor: '#FEE2E2', padding: 12}}>
							<Text style={{color: '#B91C1C', fontSize: 13, fontWeight: '600'}}>{error}</Text>
						</View>
					) : null}

					{/* Google Sign In Button */}
					<Pressable
						onPress={handleLogin}
						disabled={signingIn}
						style={({pressed}) => ({
							borderRadius: 14,
							paddingVertical: 16,
							alignItems: 'center',
							justifyContent: 'center',
							backgroundColor: pressed ? '#F0FDF4' : '#FFFFFF',
							borderWidth: 1.5,
							borderColor: '#0ABF71',
							flexDirection: 'row',
							gap: 10,
							marginBottom: 12,
						})}
					>
						{signingIn
							? <ActivityIndicator size="small" color="#063B24" />
							: <MaterialCommunityIcons name="google" size={22} color="#EA4335" />
						}
						<Text style={{fontSize: 16, fontWeight: '700', color: '#063B24'}}>
							{signingIn ? 'Signing in...' : 'Continue with Google'}
						</Text>
					</Pressable>

					{/* Divider */}
					<View style={{flexDirection: 'row', alignItems: 'center', marginVertical: 12}}>
						<View style={{flex: 1, height: 1, backgroundColor: '#D6E9DC'}} />
						<Text style={{marginHorizontal: 12, color: '#4B6A57', fontSize: 13}}>or</Text>
						<View style={{flex: 1, height: 1, backgroundColor: '#D6E9DC'}} />
					</View>

					{/* Skip Button */}
					<Pressable
						onPress={handleSkip}
						style={({pressed}) => ({
							borderRadius: 14,
							paddingVertical: 16,
							alignItems: 'center',
							justifyContent: 'center',
							backgroundColor: pressed ? '#E8F5EC' : '#F0FDF4',
							borderWidth: 1.5,
							borderColor: '#D6E9DC',
						})}
					>
						<Text style={{fontSize: 16, fontWeight: '700', color: '#4B6A57'}}>Skip for now</Text>
					</Pressable>

					<Text style={{marginTop: 14, fontSize: 12, color: '#9BB8A5', textAlign: 'center'}}>
						You can sign in later from the app settings.
					</Text>
				</View>
			</ScrollView>
		</SafeAreaView>
	);
};

export default LoginScreen;
