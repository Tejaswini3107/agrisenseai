import { Platform, PermissionsAndroid } from 'react-native'

async function requestAndroidLocation() {
  try {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      {
        title: 'Location Permission',
        message:
          'AgriSense AI needs access to your location to provide weather updates and location-based insights.',
        buttonNeutral: 'Ask Me Later',
        buttonNegative: 'Cancel',
        buttonPositive: 'OK'
      }
    )
    return granted === PermissionsAndroid.RESULTS.GRANTED
  } catch (err) {
    console.warn('requestAndroidLocation error', err)
    return false
  }
}

export async function requestLocationPermission() {
  if (Platform.OS === 'android') {
    return await requestAndroidLocation()
  }
  // On iOS, permissions are handled via Info.plist entries and the native prompt
  return true
}

export function getCurrentPosition(options = {}) {
  return new Promise((resolve, reject) => {
    if (typeof navigator !== 'undefined' && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        position => resolve(position),
        err => reject(err),
        options
      )
    } else {
      reject(new Error('Geolocation is not available'))
    }
  })
}

export default {
  requestLocationPermission,
  getCurrentPosition
}
