/* eslint-env jest */

// Ensure react-native Platform is mocked before importing cropStore
jest.resetModules();

describe('cropStore (web localStorage)', () => {
  beforeEach(() => {
    // Mock react-native Platform as web
    jest.doMock('react-native', () => ({
      Platform: { OS: 'web' },
    }));

    // Provide a fake localStorage
    const storage = {};
    global.localStorage = {
      getItem: (k) => (k in storage ? storage[k] : null),
      setItem: (k, v) => { storage[k] = v; },
      removeItem: (k) => { delete storage[k]; },
    };
  });

  afterEach(() => {
    jest.resetModules();
    delete global.localStorage;
    jest.dontMock('react-native');
  });

  test('get/set/subscribe work on web', async () => {
    const cropStore = require('../src/services/cropStore').default;

    expect(await cropStore.getCrop()).toBeNull();
    expect(await cropStore.getLanguage()).toBeNull();

    let notified = null;
    const unsub = cropStore.subscribe((v) => { notified = v; });
    let languageNotified = null;
    const unsubLanguage = cropStore.subscribeLanguage((v) => { languageNotified = v; });

    await cropStore.setCrop('rice');
    expect(await cropStore.getCrop()).toBe('rice');
    expect(notified).toBe('rice');

    await cropStore.setLanguage('hindi');
    expect(await cropStore.getLanguage()).toBe('hindi');
    expect(languageNotified).toBe('hindi');

    await cropStore.setCrop(null);
    expect(await cropStore.getCrop()).toBeNull();

    await cropStore.setLanguage(null);
    expect(await cropStore.getLanguage()).toBeNull();

    unsub();
    unsubLanguage();
  });
});


describe('cropStore (native AsyncStorage)', () => {
  beforeEach(() => {
    // Mock react-native Platform as android
    jest.doMock('react-native', () => ({
      Platform: { OS: 'android' },
    }));

    // Mock AsyncStorage module
    const storage = {};
    jest.doMock('@react-native-async-storage/async-storage', () => ({
      default: {
        getItem: async (k) => (k in storage ? storage[k] : null),
        setItem: async (k, v) => { storage[k] = v; },
        removeItem: async (k) => { delete storage[k]; },
      },
    }));
  });

  afterEach(() => {
    jest.resetModules();
    jest.dontMock('react-native');
    jest.dontMock('@react-native-async-storage/async-storage');
  });

  test('get/set/subscribe work with AsyncStorage', async () => {
    const cropStore = require('../src/services/cropStore').default;

    expect(await cropStore.getCrop()).toBeNull();
    expect(await cropStore.getLanguage()).toBeNull();

    let notified = null;
    const unsub = cropStore.subscribe((v) => { notified = v; });
    let languageNotified = null;
    const unsubLanguage = cropStore.subscribeLanguage((v) => { languageNotified = v; });

    await cropStore.setCrop('maize');
    expect(await cropStore.getCrop()).toBe('maize');
    expect(notified).toBe('maize');

    await cropStore.setLanguage('english');
    expect(await cropStore.getLanguage()).toBe('english');
    expect(languageNotified).toBe('english');

    await cropStore.setCrop(null);
    expect(await cropStore.getCrop()).toBeNull();

    await cropStore.setLanguage(null);
    expect(await cropStore.getLanguage()).toBeNull();

    unsub();
    unsubLanguage();
  });
});
