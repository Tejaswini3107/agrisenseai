import {Alert, Platform} from 'react-native';
import OneSignal from 'react-native-onesignal';

import {registerNotificationToken} from './notifications';
import cropStore from './cropStore';

let foregroundUnsubscribe = null;
let languageUnsubscribe = null;
let initialized = false;
let currentSession = null;
let latestPushToken = null;

const ONESIGNAL_APP_ID =
	globalThis?.process?.env?.ONESIGNAL_APP_ID ||
	globalThis?.process?.env?.REACT_APP_ONESIGNAL_APP_ID ||
	'';

const addSubscriptionObserver = (callback) => {
	if (typeof OneSignal.addSubscriptionObserver === 'function') {
		OneSignal.addSubscriptionObserver(callback);
		return () => {
			if (typeof OneSignal.removeSubscriptionObserver === 'function') {
				OneSignal.removeSubscriptionObserver(callback);
			}
		};
	}
	return null;
};

const getSessionLanguage = async () => {
	try {
		return (await cropStore.getLanguage()) || 'english';
	} catch {
		return 'english';
	}
};

const getDeviceLabel = () => (Platform.OS === 'ios' ? 'iPhone' : 'Android');

const showForegroundAlert = (remoteMessage) => {
	const title = remoteMessage?.title || remoteMessage?.notification?.title || remoteMessage?.data?.title || 'Emergency alert';
	const body = remoteMessage?.body || remoteMessage?.notification?.body || remoteMessage?.data?.body || 'Open the app for details.';
	Alert.alert(title, body);
};

const registerCurrentDevice = async () => {
	if (!currentSession?.farmer?.id) return;
	if (!latestPushToken) return;

	try {
		await registerNotificationToken({
			token: latestPushToken,
			userUid: currentSession?.farmer?.google_uid || currentSession?.farmer?.uid || String(currentSession?.farmer?.id),
			role: 'farmer',
			deviceName: getDeviceLabel(),
			preferredLanguage: (await getSessionLanguage()) || 'english',
			email: currentSession?.farmer?.email,
		});
	} catch (error) {
		console.warn('pushNotifications registerCurrentDevice error', error);
	}
};

export const bootstrapPushNotifications = async (session) => {
	currentSession = session || null;

	if (!initialized) {
		initialized = true;

		if (!ONESIGNAL_APP_ID) {
			console.warn('pushNotifications: ONESIGNAL_APP_ID is not set');
			return;
		}

		OneSignal.setAppId(ONESIGNAL_APP_ID);
		OneSignal.promptForPushNotificationsWithUserResponse((accepted) => {
			if (!accepted) {
				console.warn('pushNotifications: permission not granted');
			}
		});

		OneSignal.setNotificationWillShowInForegroundHandler((event) => {
			const notification = event?.getNotification?.();
			showForegroundAlert({
				title: notification?.title,
				body: notification?.body,
				data: notification?.additionalData,
			});
			event?.complete?.(notification);
		});

		OneSignal.setNotificationOpenedHandler((openedEvent) => {
			const notification = openedEvent?.notification;
			showForegroundAlert({
				title: notification?.title,
				body: notification?.body,
				data: notification?.additionalData,
			});
		});

		foregroundUnsubscribe = addSubscriptionObserver((state) => {
			latestPushToken = state?.to?.pushToken || null;
			if (latestPushToken && currentSession?.farmer?.id) {
				registerCurrentDevice();
			}
		});

		OneSignal.getDeviceState()
			.then((deviceState) => {
				latestPushToken = deviceState?.pushToken || null;
				if (latestPushToken && currentSession?.farmer?.id) {
					registerCurrentDevice();
				}
			})
			.catch((error) => {
				console.warn('pushNotifications getDeviceState error', error);
			});

		languageUnsubscribe = cropStore.subscribeLanguage(() => {
			if (currentSession?.farmer?.id) {
				registerCurrentDevice();
			}
		});
	}

	await registerCurrentDevice();
};

export const teardownPushNotifications = () => {
	if (typeof foregroundUnsubscribe === 'function') foregroundUnsubscribe();
	if (typeof languageUnsubscribe === 'function') languageUnsubscribe();
	foregroundUnsubscribe = null;
	languageUnsubscribe = null;
	initialized = false;
	currentSession = null;
	latestPushToken = null;
};
