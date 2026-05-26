import React from 'react';
import {StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import {useNavigation, useRoute} from '@react-navigation/native';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';

const items = [
	{key: 'Home', icon: 'home-variant-outline'},
	{key: 'PestAlert', icon: 'bell-alert-outline', badge: '2'},
	{key: 'VoiceAssistant', icon: 'account-circle-outline'},
	{key: 'OfflineDashboard', icon: 'cog-outline'},
];

const BottomNav = () => {
	const navigation = useNavigation();
	const route = useRoute();

	return (
		<View style={styles.nav}>
			{items.map((item) => {
				const isActive = route.name === item.key;
				return (
					<TouchableOpacity
						key={item.key}
						onPress={() => navigation.navigate(item.key)}
						style={styles.navItem}
						activeOpacity={0.8}>
						{item.badge ? (
							<View style={styles.badge}>
								<Text style={styles.badgeText}>{item.badge}</Text>
							</View>
						) : null}
						<MaterialCommunityIcons
							name={item.icon}
							size={24}
							color={isActive ? '#0A7A4D' : '#8A9490'}
						/>
						{isActive ? <View style={styles.indicator} /> : null}
					</TouchableOpacity>
				);
			})}
		</View>
	);
};

const styles = StyleSheet.create({
	nav: {
		height: 68,
		backgroundColor: '#FFFFFF',
		borderTopLeftRadius: 18,
		borderTopRightRadius: 18,
		borderTopWidth: 1,
		borderTopColor: '#DCEAD8',
		paddingHorizontal: 16,
		flexDirection: 'row',
		alignItems: 'center',
		justifyContent: 'space-around',
		shadowColor: '#000',
		shadowOpacity: 0.08,
		shadowRadius: 12,
		elevation: 8,
	},
	navItem: {
		minWidth: 64,
		alignItems: 'center',
		justifyContent: 'center',
		height: 44,
	},
	indicator: {
		width: 24,
		height: 4,
		borderRadius: 999,
		backgroundColor: '#0A7A4D',
		marginTop: 4,
	},
	badge: {
		position: 'absolute',
		top: -6,
		right: 16,
		minWidth: 18,
		height: 18,
		paddingHorizontal: 4,
		borderRadius: 999,
		backgroundColor: '#10B981',
		alignItems: 'center',
		justifyContent: 'center',
	},
	badgeText: {
		fontSize: 10,
		fontWeight: '700',
		color: '#FFFFFF',
	},
});

export default BottomNav;