import {Platform} from 'react-native';

const isWeb = Platform.OS === 'web';
let NativeAsyncStorage = null;
if (!isWeb) {
  // eslint-disable-next-line global-require
  NativeAsyncStorage = require('@react-native-async-storage/async-storage').default;
}

const STORAGE_KEY = 'agrisense:lastKnownLocation';
const listeners = new Set();

const readValue = async (key) => {
  if (isWeb && typeof localStorage !== 'undefined') {
    const raw = localStorage.getItem(key);
    return raw;
  }
  if (NativeAsyncStorage) return await NativeAsyncStorage.getItem(key);
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

export const getLocation = async () => {
  try {
    const raw = await readValue(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) {
    console.warn('locationStore.getLocation error', e);
    return null;
  }
};

export const setLocation = async (latLonObj) => {
  try {
    if (!latLonObj) {
      await writeValue(STORAGE_KEY, null);
    } else {
      await writeValue(STORAGE_KEY, JSON.stringify(latLonObj));
    }
  } catch (e) {
    console.warn('locationStore.setLocation error', e);
  }

  for (const cb of Array.from(listeners)) {
    try {
      cb(latLonObj);
    } catch (err) {
      console.warn('locationStore listener error', err);
    }
  }
};

export const subscribe = (cb) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};

export default {getLocation, setLocation, subscribe};
