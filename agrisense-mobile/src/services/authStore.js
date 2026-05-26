import {Platform} from 'react-native';

const isWeb = Platform.OS === 'web';
let NativeAsyncStorage = null;
if (!isWeb) {
	// require() so web bundlers don't include the native package when building for web.
	// eslint-disable-next-line global-require
	NativeAsyncStorage = require('@react-native-async-storage/async-storage').default;
}

const STORAGE_KEY = 'agrisense:farmerSession';

const listeners = new Set();
let cachedSession = undefined;

const readValue = async (key) => {
	if (isWeb && typeof localStorage !== 'undefined') {
		return localStorage.getItem(key);
	}
	if (NativeAsyncStorage) {
		return await NativeAsyncStorage.getItem(key);
	}
	return null;
};

const writeValue = async (key, value) => {
	if (isWeb && typeof localStorage !== 'undefined') {
		if (value) localStorage.setItem(key, value);
		else localStorage.removeItem(key);
		return;
	}
	if (NativeAsyncStorage) {
		if (value) await NativeAsyncStorage.setItem(key, value);
		else await NativeAsyncStorage.removeItem(key);
	}
};

const notify = (session) => {
	for (const cb of Array.from(listeners)) {
		try {
			cb(session);
		} catch (error) {
			console.warn('authStore listener error', error);
		}
	}
};

const hydrateSession = async () => {
	if (cachedSession !== undefined) {
		return cachedSession;
	}

	try {
		const raw = await readValue(STORAGE_KEY);
		cachedSession = raw ? JSON.parse(raw) : null;
		return cachedSession;
	} catch (error) {
		console.warn('authStore hydrate error', error);
		cachedSession = null;
		return null;
	}
};

export const getCurrentFarmer = async () => {
	return await hydrateSession();
};

export const setCurrentFarmer = async (session) => {
	cachedSession = session || null;
	try {
		await writeValue(STORAGE_KEY, session ? JSON.stringify(session) : null);
	} catch (error) {
		console.warn('authStore setCurrentFarmer error', error);
	}
	notify(cachedSession);
	return cachedSession;
};

export const clearCurrentFarmer = async () => {
	await setCurrentFarmer(null);
};

export const subscribe = (cb) => {
	listeners.add(cb);
	return () => listeners.delete(cb);
};

export default {getCurrentFarmer, setCurrentFarmer, clearCurrentFarmer, subscribe};