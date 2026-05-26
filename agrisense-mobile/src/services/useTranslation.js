import {useState, useEffect, useCallback} from 'react';
import {I18nManager, Alert} from 'react-native';
import translations from './translations';
import cropStore from './cropStore';

export default function useTranslation() {
  const [language, setLanguage] = useState('english');

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const lang = await cropStore.getLanguage();
        if (mounted) setLanguage(lang || 'english');
      } catch (e) {
        if (mounted) setLanguage('english');
      }
    })();

    const unsub = cropStore.subscribeLanguage((v) => {
      setLanguage(v || 'english');
    });

    return () => {
      mounted = false;
      if (typeof unsub === 'function') unsub();
    };
  }, []);

  const t = useCallback((scope, key) => {
    return translations.t(language, scope, key);
  }, [language]);

  const setLang = async (l) => {
    await cropStore.setLanguage(l);
    setLanguage(l);
  };

  // toggle RTL settings when language changes to or from Arabic/Urdu
  useEffect(() => {
    if (!language) return;
    const isRtl = ['arabic', 'urdu'].includes((language || '').toLowerCase());
    try {
      if (isRtl && !I18nManager.isRTL) {
        I18nManager.allowRTL(true);
        I18nManager.forceRTL(true);
        // attempt an automatic restart if the native restart module is available
        try {
          // react-native-restart exports a default function
          // require at runtime so projects without the dependency won't crash
          // eslint-disable-next-line global-require
          const RNRestart = require('react-native-restart').default;
          if (typeof RNRestart === 'function') {
            RNRestart();
            return;
          }
        } catch (e) {
          // continue to alert fallback
        }
        Alert.alert('Restart Required', 'Please restart the app to apply RTL layout for this language.');
      } else if (!isRtl && I18nManager.isRTL) {
        I18nManager.allowRTL(false);
        I18nManager.forceRTL(false);
        try {
          const RNRestart = require('react-native-restart').default;
          if (typeof RNRestart === 'function') {
            RNRestart();
            return;
          }
        } catch (e) {
          // continue to alert fallback
        }
        Alert.alert('Restart Required', 'Please restart the app to apply LTR layout.');
      }
    } catch (e) {
      // ignore — platform may restrict RTL toggling at runtime
    }
  }, [language]);

  return {t, language, setLanguage: setLang};
}
