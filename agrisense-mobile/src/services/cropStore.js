import {Platform} from 'react-native';
import {getCurrentFarmer, setCurrentFarmer} from './authStore';
import {recordFarmerCropSearch} from './api';

const isWeb = Platform.OS === 'web';
let NativeAsyncStorage = null;
if (!isWeb) {
  // require() so web bundlers don't try to include the native package when building for web.
  // eslint-disable-next-line global-require
  NativeAsyncStorage = require('@react-native-async-storage/async-storage').default;
}

const STORAGE_KEY = 'agrisense:selectedCrop';
const LANGUAGE_STORAGE_KEY = 'agrisense:selectedLanguage';

const listeners = new Set();
const languageListeners = new Set();

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

export const getCrop = async () => {
  try {
    return await readValue(STORAGE_KEY);
  } catch (e) {
    console.warn('cropStore.getCrop error', e);
    return null;
  }
};

export const setCrop = async (crop) => {
  try {
    await writeValue(STORAGE_KEY, crop);
  } catch (e) {
    console.warn('cropStore.setCrop error', e);
  }

  // notify listeners
  for (const cb of Array.from(listeners)) {
    try {
      cb(crop);
    } catch (err) {
      console.warn('cropStore listener error', err);
    }
  }

  try {
    const session = await getCurrentFarmer();
    const farmerId = session?.farmer?.id;
    if (farmerId && crop) {
      const response = await recordFarmerCropSearch({farmerId, crop, source: 'search'});
      if (response?.farmer) {
        await setCurrentFarmer({
          ...session,
          farmer: response.farmer,
        });
      }
    }
  } catch (syncError) {
    console.warn('cropStore.setCrop sync error', syncError);
  }
};

export const getLanguage = async () => {
  try {
    return await readValue(LANGUAGE_STORAGE_KEY);
  } catch (e) {
    console.warn('cropStore.getLanguage error', e);
    return null;
  }
};

export const setLanguage = async (language) => {
  try {
    await writeValue(LANGUAGE_STORAGE_KEY, language);
  } catch (e) {
    console.warn('cropStore.setLanguage error', e);
  }

  for (const cb of Array.from(languageListeners)) {
    try {
      cb(language);
    } catch (err) {
      console.warn('cropStore language listener error', err);
    }
  }
};

export const subscribe = (cb) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};

export const subscribeLanguage = (cb) => {
  languageListeners.add(cb);
  return () => languageListeners.delete(cb);
};

export default {getCrop, setCrop, subscribe, getLanguage, setLanguage, subscribeLanguage};
