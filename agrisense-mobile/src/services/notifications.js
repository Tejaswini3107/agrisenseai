import {Platform} from 'react-native';

import {API_BASE_URL} from './apiConfig';

const fetchJson = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
};

export const getNotificationStatus = async () => {
  return fetchJson(`${API_BASE_URL}/api/notifications/status`);
};

export const registerNotificationToken = async ({
  token,
  role = 'farmer',
  userUid = null,
  deviceName = null,
  preferredLanguage = null,
  email = null,
  platform = Platform.OS,
} = {}) => {
  return fetchJson(`${API_BASE_URL}/api/notifications/register-token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      token,
      role,
      user_uid: userUid,
      email,
      device_name: deviceName,
      preferred_language: preferredLanguage,
      platform,
    }),
  });
};
