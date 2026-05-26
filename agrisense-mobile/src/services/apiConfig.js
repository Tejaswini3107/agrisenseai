import {Platform} from 'react-native';

const DEFAULT_BACKEND_BASE_BY_PLATFORM = {
	android: 'http://10.0.2.2:8000',
	ios: 'http://localhost:8000',
	default: 'http://localhost:8000',
};

const normalizeBackendBaseUrl = (value) => {
	if (!value) return null;
	return String(value).trim().replace(/\/+$/, '');
};

const resolveBackendBaseUrl = () => {
	const envBaseUrl = normalizeBackendBaseUrl(
		globalThis?.process?.env?.AGRISENSE_API_BASE_URL ||
		globalThis?.process?.env?.API_BASE_URL ||
		globalThis?.__AGRISENSE_API_BASE_URL__,
	);

	if (envBaseUrl) {
		return envBaseUrl;
	}

	if (typeof Platform?.select === 'function') {
		return Platform.select(DEFAULT_BACKEND_BASE_BY_PLATFORM);
	}

	const osKey = Platform?.OS || 'default';
	return DEFAULT_BACKEND_BASE_BY_PLATFORM[osKey] || DEFAULT_BACKEND_BASE_BY_PLATFORM.default;
};

export const API_BASE_URL = resolveBackendBaseUrl();