import React, {useEffect, useState} from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import {getCropCatalog} from '../services/api';
import cropStore from '../services/cropStore';

const FALLBACK_CROPS = [
  'rice', 'wheat', 'cotton', 'maize', 'turmeric',
  'sugarcane', 'soybean', 'potato', 'onion', 'garlic',
];

const LANGUAGES = [
  {key: 'english', label: 'English'},
  {key: 'hindi', label: 'हिंदी (Hindi)'},
  {key: 'telugu', label: 'తెలుగు (Telugu)'},
  {key: 'marathi', label: 'मराठी (Marathi)'},
  {key: 'urdu', label: 'اردو (Urdu)'},
  {key: 'arabic', label: 'العربية (Arabic)'},
  {key: 'french', label: 'Français (French)'},
];

/* ── Reusable Dropdown ─────────────────────────────────────── */
const Dropdown = ({label, icon, value, options, onSelect}) => {
  const [open, setOpen] = useState(false);
  const displayLabel = options.find(o => o.key === value)?.label || value;

  return (
    <View style={{marginBottom: 18}}>
      <Text style={{fontSize: 13, fontWeight: '700', color: '#4B6A57', marginBottom: 6}}>
        {label}
      </Text>
      <Pressable
        onPress={() => setOpen(true)}
        style={({pressed}) => ({
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: pressed ? '#F0FDF4' : '#FFFFFF',
          borderWidth: 1.5,
          borderColor: '#0ABF71',
          borderRadius: 14,
          paddingHorizontal: 16,
          paddingVertical: 14,
        })}
      >
        <View style={{flexDirection: 'row', alignItems: 'center', gap: 10}}>
          <MaterialCommunityIcons name={icon} size={20} color="#0ABF71" />
          <Text style={{fontSize: 15, fontWeight: '600', color: '#063B24', textTransform: 'capitalize'}}>
            {displayLabel}
          </Text>
        </View>
        <MaterialCommunityIcons name="chevron-down" size={22} color="#4B6A57" />
      </Pressable>

      {/* Dropdown Modal */}
      <Modal transparent animationType="fade" visible={open} onRequestClose={() => setOpen(false)}>
        <Pressable
          style={{flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 24}}
          onPress={() => setOpen(false)}
        >
          <Pressable onPress={() => {}} style={{backgroundColor: '#FFFFFF', borderRadius: 20, overflow: 'hidden'}}>
            <View style={{backgroundColor: '#0ABF71', paddingHorizontal: 18, paddingVertical: 14}}>
              <Text style={{color: '#FFFFFF', fontWeight: '800', fontSize: 16}}>{label}</Text>
            </View>
            <ScrollView style={{maxHeight: 320}}>
              {options.map((opt, idx) => (
                <TouchableOpacity
                  key={opt.key}
                  onPress={() => {onSelect(opt.key); setOpen(false);}}
                  style={{
                    flexDirection: 'row',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    paddingHorizontal: 18,
                    paddingVertical: 14,
                    backgroundColor: value === opt.key ? '#F0FDF4' : '#FFFFFF',
                    borderTopWidth: idx === 0 ? 0 : 1,
                    borderTopColor: '#F0F0F0',
                  }}
                >
                  <Text style={{
                    fontSize: 15,
                    fontWeight: value === opt.key ? '700' : '500',
                    color: value === opt.key ? '#0ABF71' : '#063B24',
                    textTransform: 'capitalize',
                  }}>
                    {opt.label}
                  </Text>
                  {value === opt.key && (
                    <MaterialCommunityIcons name="check-circle" size={20} color="#0ABF71" />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
};

/* ── Setup Screen ──────────────────────────────────────────── */
const SetupScreen = ({navigation}) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [cropOptions, setCropOptions] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState('sugarcane');
  const [selectedLanguage, setSelectedLanguage] = useState('english');

  useEffect(() => {
    const load = async () => {
      // Load saved preferences
      const [savedCrop, savedLang] = await Promise.all([
        cropStore.getCrop(),
        cropStore.getLanguage(),
      ]);
      if (savedCrop) setSelectedCrop(savedCrop);
      if (savedLang) setSelectedLanguage(savedLang);

      // Load crop catalog
      try {
        const response = await getCropCatalog();
        const catalog = Array.isArray(response?.crops)
          ? response.crops.map(item => item.key).filter(Boolean)
          : [];
        setCropOptions(catalog.length > 0 ? catalog : FALLBACK_CROPS);
        if (!savedCrop) {
          setSelectedCrop(catalog.includes('sugarcane') ? 'sugarcane' : (catalog[0] || 'sugarcane'));
        }
      } catch {
        setCropOptions(FALLBACK_CROPS);
      }

      setLoading(false);
    };
    load();
  }, []);

  const handleDone = async () => {
    setSaving(true);
    await Promise.all([
      cropStore.setCrop(selectedCrop),
      cropStore.setLanguage(selectedLanguage),
    ]);
    setSaving(false);
    navigation.replace('Home');
  };

  const cropDropdownOptions = cropOptions.map(c => ({key: c, label: c}));

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

        {/* Header */}
        <View style={{alignItems: 'center', marginBottom: 36}}>
          <View style={{width: 80, height: 80, borderRadius: 24, backgroundColor: '#0ABF71', alignItems: 'center', justifyContent: 'center', marginBottom: 14}}>
            <MaterialCommunityIcons name="tune-variant" size={40} color="#FFFFFF" />
          </View>
          <Text style={{fontSize: 26, fontWeight: '800', color: '#FFFFFF'}}>Set Your Preferences</Text>
          <Text style={{marginTop: 6, fontSize: 14, color: '#A7D8BA', textAlign: 'center', lineHeight: 20}}>
            Choose your primary crop and preferred language.{'\n'}You can change these anytime.
          </Text>
        </View>

        {/* Card */}
        <View style={{borderRadius: 24, backgroundColor: '#F7FFF9', padding: 24, shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 20, elevation: 8}}>

          <Dropdown
            label="Select Your Crop"
            icon="sprout"
            value={selectedCrop}
            options={cropDropdownOptions}
            onSelect={setSelectedCrop}
          />

          <Dropdown
            label="Select Language"
            icon="translate"
            value={selectedLanguage}
            options={LANGUAGES}
            onSelect={setSelectedLanguage}
          />

          {/* Done Button */}
          <Pressable
            onPress={handleDone}
            disabled={saving}
            style={({pressed}) => ({
              borderRadius: 14,
              paddingVertical: 16,
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: pressed ? '#08A660' : '#0ABF71',
              flexDirection: 'row',
              gap: 8,
              marginTop: 8,
            })}
          >
            {saving
              ? <ActivityIndicator size="small" color="#FFFFFF" />
              : <MaterialCommunityIcons name="arrow-right-circle" size={22} color="#FFFFFF" />
            }
            <Text style={{fontSize: 16, fontWeight: '800', color: '#FFFFFF'}}>
              {saving ? 'Saving...' : 'Get Started'}
            </Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default SetupScreen;
