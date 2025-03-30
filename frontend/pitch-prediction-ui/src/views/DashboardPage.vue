<script setup>
import { ref, onMounted, computed } from 'vue';
import { db } from '../firebase';
import { collection, query, orderBy, limit, getDocs } from 'firebase/firestore';

// Data for dashboard
const predictions = ref([]);
const isLoading = ref(true);
const error = ref(null);

// Chart data
const pitchTypeDistribution = ref({
  labels: [],
  datasets: []
});

const countTypeAnalysis = ref({
  labels: [],
  datasets: []
});

// Fetch data from Firebase on component mount
onMounted(async () => {
  try {
    await fetchPredictions();
    generateChartData();
  } catch (err) {
    console.error('Error loading dashboard data:', err);
    error.value = 'Failed to load dashboard data. Please try again later.';
  } finally {
    isLoading.value = false;
  }
});

// Fetch predictions from Firebase
const fetchPredictions = async () => {
  const predictionsQuery = query(
    collection(db, 'predictions'),
    orderBy('createdAt', 'desc'),
    limit(100)
  );
  
  const querySnapshot = await getDocs(predictionsQuery);
  predictions.value = querySnapshot.docs.map(doc => ({
    id: doc.id,
    ...doc.data(),
    createdAt: doc.data().createdAt?.toDate() || new Date()
  }));
};

// Generate chart data from predictions
const generateChartData = () => {
  if (predictions.value.length === 0) return;
  
  // Count pitch types
  const pitchCounts = {};
  predictions.value.forEach(prediction => {
    const pitch = prediction.topPitch;
    pitchCounts[pitch] = (pitchCounts[pitch] || 0) + 1;
  });
  
  // Prepare data for chart
  pitchTypeDistribution.value = {
    labels: Object.keys(pitchCounts).map(type => getPitchName(type)),
    datasets: [{
      label: 'Predicted Pitch Types',
      data: Object.values(pitchCounts),
      backgroundColor: [
        '#4299E1', // blue
        '#F56565', // red
        '#48BB78', // green
        '#ED8936', // orange
        '#9F7AEA', // purple
        '#667EEA'  // indigo
      ]
    }]
  };
  
  // Count by count type
  const countTypeCounts = {
    'Hitter': 0,
    'Neutral': 0,
    'Pitcher': 0
  };
  
  predictions.value.forEach(prediction => {
    const countType = prediction.gameState?.countType;
    if (countType) {
      countTypeCounts[countType] = (countTypeCounts[countType] || 0) + 1;
    }
  });
  
  countTypeAnalysis.value = {
    labels: Object.keys(countTypeCounts),
    datasets: [{
      label: 'Predictions by Count Type',
      data: Object.values(countTypeCounts),
      backgroundColor: [
        '#F56565', // red for hitter
        '#4299E1', // blue for neutral
        '#48BB78'  // green for pitcher
      ]
    }]
  };
};

// Statistics
const totalPredictions = computed(() => predictions.value.length);

const mostCommonPitch = computed(() => {
  if (predictions.value.length === 0) return 'N/A';
  
  const pitchCounts = {};
  predictions.value.forEach(prediction => {
    const pitch = prediction.topPitch;
    pitchCounts[pitch] = (pitchCounts[pitch] || 0) + 1;
  });
  
  const mostCommon = Object.entries(pitchCounts)
    .sort((a, b) => b[1] - a[1])[0];
    
  return getPitchName(mostCommon[0]);
});

const averageProbability = computed(() => {
  if (predictions.value.length === 0) return 0;
  
  const total = predictions.value.reduce(
    (sum, prediction) => sum + (prediction.topPitchProbability || 0), 0
  );
  
  return (total / predictions.value.length * 100).toFixed(1) + '%';
});

// Calculate model accuracy based on recorded outcomes
const modelAccuracy = computed(() => {
  // Filter predictions that have actual outcomes recorded
  const verifiedPredictions = predictions.value.filter(pred => pred.actualPitch);
  
  if (verifiedPredictions.length === 0) return { percentage: 0, count: 0, total: 0 };
  
  // Count correct predictions
  const correctPredictions = verifiedPredictions.filter(pred => pred.wasCorrect).length;
  
  // Calculate accuracy percentage
  const accuracyPercentage = (correctPredictions / verifiedPredictions.length) * 100;
  
  return {
    percentage: accuracyPercentage.toFixed(1),
    count: correctPredictions,
    total: verifiedPredictions.length
  };
});

const recentPredictions = computed(() => {
  return predictions.value.slice(0, 5).map(p => ({
    ...p,
    formattedDate: new Date(p.timestamp).toLocaleDateString(),
    formattedTime: new Date(p.timestamp).toLocaleTimeString(),
  }));
});

// Helper function to get readable pitch names
const getPitchName = (code) => {
  const pitchNames = {
    'FB': 'Fastball',
    'SL': 'Slider',
    'CH': 'Changeup',
    'CU': 'Curveball',
    'CT': 'Cutter',
    'SI': 'Sinker',
    'KC': 'Knuckle Curve',
    'KN': 'Knuckleball',
    'FS': 'Splitter'
  };
  
  return pitchNames[code] || code;
};
</script>

<template>
  <div class="max-w-7xl mx-auto px-4 py-8">
    <h1 class="text-3xl font-bold text-center mb-8 text-baseball-blue">Pitch Prediction Dashboard</h1>
    
    <div v-if="isLoading" class="flex justify-center items-center py-16">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-baseball-red"></div>
      <span class="ml-3 text-gray-700">Loading dashboard data...</span>
    </div>
    
    <div v-else-if="error" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
      <strong class="font-bold">Error!</strong>
      <span class="block sm:inline">{{ error }}</span>
    </div>
    
    <div v-else>
      <!-- Key Statistics -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-gray-500 text-sm uppercase mb-2">Total Predictions</h2>
          <p class="text-3xl font-bold text-baseball-blue">{{ totalPredictions }}</p>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-gray-500 text-sm uppercase mb-2">Most Common Pitch</h2>
          <p class="text-3xl font-bold text-baseball-blue">{{ mostCommonPitch }}</p>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-gray-500 text-sm uppercase mb-2">Average Confidence</h2>
          <p class="text-3xl font-bold text-baseball-blue">{{ averageProbability }}</p>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-gray-500 text-sm uppercase mb-2">Model Accuracy</h2>
          <p class="text-3xl font-bold" 
             :class="{
               'text-green-600': parseFloat(modelAccuracy.percentage) >= 70,
               'text-yellow-600': parseFloat(modelAccuracy.percentage) >= 50 && parseFloat(modelAccuracy.percentage) < 70,
               'text-red-600': parseFloat(modelAccuracy.percentage) < 50,
               'text-gray-400': modelAccuracy.total === 0
             }">
            {{ modelAccuracy.percentage }}%
          </p>
          <p class="text-sm text-gray-500 mt-1" v-if="modelAccuracy.total > 0">
            Based on {{ modelAccuracy.count }}/{{ modelAccuracy.total }} verified predictions
          </p>
          <p class="text-sm text-gray-500 mt-1" v-else>
            No verified predictions yet
          </p>
        </div>
      </div>
      
      <!-- Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-bold text-baseball-blue mb-4">Pitch Type Distribution</h2>
          <div class="h-64">
            <!-- Chart would go here - Need to implement with a chart library -->
            <div v-if="pitchTypeDistribution.labels.length === 0" class="flex items-center justify-center h-full">
              <p class="text-gray-500">No data available</p>
            </div>
            <div v-else class="p-4 bg-gray-100 rounded">
              <p class="text-sm text-gray-700 mb-4">In a real implementation, this would display a pie chart of pitch types. For demonstration purposes, here's the data:</p>
              <div class="space-y-2">
                <div v-for="(label, index) in pitchTypeDistribution.labels" :key="index" class="flex justify-between">
                  <span class="font-medium">{{ label }}:</span>
                  <span>{{ pitchTypeDistribution.datasets[0].data[index] }} predictions</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="bg-white p-6 rounded-lg shadow-md">
          <h2 class="text-xl font-bold text-baseball-blue mb-4">Count Type Analysis</h2>
          <div class="h-64">
            <!-- Chart would go here - Need to implement with a chart library -->
            <div v-if="countTypeAnalysis.labels.length === 0" class="flex items-center justify-center h-full">
              <p class="text-gray-500">No data available</p>
            </div>
            <div v-else class="p-4 bg-gray-100 rounded">
              <p class="text-sm text-gray-700 mb-4">In a real implementation, this would display a bar chart of count types. For demonstration purposes, here's the data:</p>
              <div class="space-y-2">
                <div v-for="(label, index) in countTypeAnalysis.labels" :key="index" class="flex justify-between">
                  <span class="font-medium">{{ label }} Count:</span>
                  <span>{{ countTypeAnalysis.datasets[0].data[index] }} predictions</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Recent Predictions Table -->
      <div class="bg-white p-6 rounded-lg shadow-md">
        <h2 class="text-xl font-bold text-baseball-blue mb-4">Recent Predictions</h2>
        
        <div v-if="recentPredictions.length === 0" class="text-center py-8">
          <p class="text-gray-500">No predictions have been made yet</p>
        </div>
        
        <div v-else class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted Pitch</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="prediction in recentPredictions" :key="prediction.id">
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm text-gray-900">{{ prediction.formattedDate }}</div>
                  <div class="text-sm text-gray-500">{{ prediction.formattedTime }}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                        :class="{
                          'bg-red-100 text-red-800': prediction.gameState?.countType === 'Hitter',
                          'bg-blue-100 text-blue-800': prediction.gameState?.countType === 'Neutral',
                          'bg-green-100 text-green-800': prediction.gameState?.countType === 'Pitcher'
                        }">
                    {{ prediction.gameState?.balls }}-{{ prediction.gameState?.strikes }}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium text-baseball-blue">{{ getPitchName(prediction.topPitch) }}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="text-sm font-medium" 
                      :class="{
                        'text-green-600': prediction.topPitchProbability >= 0.8,
                        'text-green-500': prediction.topPitchProbability >= 0.6 && prediction.topPitchProbability < 0.8,
                        'text-yellow-500': prediction.topPitchProbability >= 0.4 && prediction.topPitchProbability < 0.6,
                        'text-red-500': prediction.topPitchProbability < 0.4
                      }">
                    {{ (prediction.topPitchProbability * 100).toFixed(1) }}%
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template> 