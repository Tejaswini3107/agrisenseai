import React from 'react';
import {SafeAreaView, StatusBar, StyleSheet} from 'react-native';
import HomeScreen from './screens/HomeScreen';

const App = () => {
	return (
		<SafeAreaView style={styles.root}>
			<StatusBar barStyle="light-content" backgroundColor="#045E3A" />
			<HomeScreen />
		</SafeAreaView>
	);
};

const styles = StyleSheet.create({
	root: {
		flex: 1,
		backgroundColor: '#045E3A',
	},
});

export default App;
