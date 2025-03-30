<script setup>
import { ref, onMounted, computed } from 'vue';
import { db } from '../firebase';
import { collection, query, orderBy, limit, getDocs, where } from 'firebase/firestore';

// Data for prediction history
const predictions = ref([]);
const isLoading = ref(true);
const error = ref(null);

// Pagination
const currentPage = ref(1);
const itemsPerPage = 10;

// Filters
const dateFilter = ref('all');
const pitchTypeFilter = ref('all');
const countTypeFilter = ref('all');
const confidenceFilter = ref('all');

// Custom date range
const startDate = ref('');
const endDate = ref('');

// Load predictions on component mount
onMounted(async () => {
  try {
    await loadPredictions();
  } catch (err) {
    console.error('Error loading prediction history:', err);
    error.value = 'Failed to load prediction history. Please try again later.';
  } finally {
    isLoading.value = false;
  }
});

// Fetch predictions from Firebase
const loadPredictions = async () => {
  const predictionsQuery = query(
    collection(db, 'predictions'),
    orderBy('createdAt', 'desc'),
    limit(100)
  );
  
  const querySnapshot = await getDocs(predictionsQuery);
  predictions.value = querySnapshot.docs.map(doc => ({
    id: doc.id,
    ...doc.data(),
    createdAt: doc.data().createdAt?.toDate() || new Date(),
    formattedDate: doc.data().createdAt 
      ? doc.data().createdAt.toDate().toLocaleDateString() 
      : new Date().toLocaleDateString(),
    formattedTime: doc.data().createdAt 
      ? doc.data().createdAt.toDate().toLocaleTimeString() 
      : new Date().toLocaleTimeString()
  }));
};

// Apply filters to predictions
const filteredPredictions = computed(() => {
  let result = [...predictions.value];
  
  // Apply date filter
  if (dateFilter.value !== 'all') {
    const now = new Date();
    let filterDate = new Date();
    
    if (dateFilter.value === 'today') {
      filterDate.setHours(0, 0, 0, 0);
    } else if (dateFilter.value === 'week') {
      filterDate.setDate(filterDate.getDate() - 7);
    } else if (dateFilter.value === 'month') {
      filterDate.setMonth(filterDate.getMonth() - 1);
    } else if (dateFilter.value === 'custom' && startDate.value && endDate.value) {
      const start = new Date(startDate.value);
      const end = new Date(endDate.value);
      end.setHours(23, 59, 59, 999); // Include the entire end day
      
      result = result.filter(pred => {
        const predDate = new Date(pred.timestamp);
        return predDate >= start && predDate <= end;
      });
      
      // Skip other date filtering
      return applyRemainingFilters(result);
    }
    
    result = result.filter(pred => new Date(pred.timestamp) >= filterDate);
  }
  
  return applyRemainingFilters(result);
});

// Apply non-date filters
const applyRemainingFilters = (preds) => {
  let result = [...preds];
  
  // Apply pitch type filter
  if (pitchTypeFilter.value !== 'all') {
    result = result.filter(pred => pred.topPitch === pitchTypeFilter.value);
  }
  
  // Apply count type filter
  if (countTypeFilter.value !== 'all') {
    result = result.filter(pred => pred.gameState?.countType === countTypeFilter.value);
  }
  
  // Apply confidence filter
  if (confidenceFilter.value !== 'all') {
    const confidence = parseFloat(confidenceFilter.value);
    if (confidenceFilter.value.startsWith('gt')) {
      result = result.filter(pred => pred.topPitchProbability > confidence);
    } else if (confidenceFilter.value.startsWith('lt')) {
      result = result.filter(pred => pred.topPitchProbability < confidence);
    }
  }
  
  return result;
};

// Get paginated predictions
const paginatedPredictions = computed(() => {
  const startIndex = (currentPage.value - 1) * itemsPerPage;
  return filteredPredictions.value.slice(startIndex, startIndex + itemsPerPage);
});

// Total pages for pagination
const totalPages = computed(() => {
  return Math.ceil(filteredPredictions.value.length / itemsPerPage);
});

// Page control methods
const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++;
  }
};

const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--;
  }
};

const goToPage = (page) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page;
  }
};

// Reset filters
const resetFilters = () => {
  dateFilter.value = 'all';
  pitchTypeFilter.value = 'all';
  countTypeFilter.value = 'all';
  confidenceFilter.value = 'all';
  startDate.value = '';
  endDate.value = '';
};

// Helper to get pitch name
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

// Get available pitch types for filter
const availablePitchTypes = computed(() => {
  const types = new Set();
  predictions.value.forEach(pred => {
    if (pred.topPitch) types.add(pred.topPitch);
  });
  return Array.from(types);
});
</script>

<template>
  <div class="max-w-7xl mx-auto px-4 py-8">
    <h1 class="text-3xl font-bold text-center mb-8 text-baseball-blue">Prediction History</h1>
    
    <div v-if="isLoading" class="flex justify-center items-center py-16">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-baseball-red"></div>
      <span class="ml-3 text-gray-700">Loading prediction history...</span>
    </div>
    
    <div v-else-if="error" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
      <strong class="font-bold">Error!</strong>
      <span class="block sm:inline">{{ error }}</span>
    </div>
    
    <div v-else>
      <!-- Filters -->
      <div class="bg-white p-6 rounded-lg shadow-md mb-8">
        <h2 class="text-xl font-bold text-baseball-blue mb-4">Filter Predictions</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <!-- Date Filter -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Time Period</label>
            <select v-model="dateFilter" class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue">
              <option value="all">All Time</option>
              <option value="today">Today</option>
              <option value="week">Past Week</option>
              <option value="month">Past Month</option>
              <option value="custom">Custom Range</option>
            </select>
          </div>
          
          <!-- Pitch Type Filter -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Pitch Type</label>
            <select v-model="pitchTypeFilter" class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue">
              <option value="all">All Pitches</option>
              <option v-for="type in availablePitchTypes" :key="type" :value="type">
                {{ getPitchName(type) }}
              </option>
            </select>
          </div>
          
          <!-- Count Type Filter -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Count Type</label>
            <select v-model="countTypeFilter" class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue">
              <option value="all">All Counts</option>
              <option value="Hitter">Hitter Counts</option>
              <option value="Neutral">Neutral Counts</option>
              <option value="Pitcher">Pitcher Counts</option>
            </select>
          </div>
          
          <!-- Confidence Filter -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Confidence</label>
            <select v-model="confidenceFilter" class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue">
              <option value="all">Any Confidence</option>
              <option value="gt0.8">High (>80%)</option>
              <option value="gt0.6">Medium (>60%)</option>
              <option value="lt0.6">Low (<60%)</option>
            </select>
          </div>
        </div>
        
        <!-- Custom Date Range -->
        <div v-if="dateFilter === 'custom'" class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
            <input 
              type="date" 
              v-model="startDate"
              class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">End Date</label>
            <input 
              type="date" 
              v-model="endDate"
              class="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-baseball-blue focus:border-baseball-blue"
            />
          </div>
        </div>
        
        <!-- Filter Actions -->
        <div class="flex justify-end">
          <button 
            @click="resetFilters"
            class="bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2 px-4 rounded mr-2"
          >
            Reset Filters
          </button>
        </div>
      </div>
      
      <!-- Prediction Results -->
      <div class="bg-white p-6 rounded-lg shadow-md">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-bold text-baseball-blue">Results ({{ filteredPredictions.length }})</h2>
        </div>
        
        <div v-if="filteredPredictions.length === 0" class="text-center py-8">
          <p class="text-gray-500">No predictions match your filters</p>
        </div>
        
        <div v-else>
          <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-gray-200">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Situation</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted Pitch</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actual Pitch</th>
                </tr>
              </thead>
              <tbody class="bg-white divide-y divide-gray-200">
                <tr v-for="prediction in paginatedPredictions" :key="prediction.id">
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
                    <div class="text-sm text-gray-900">
                      {{ prediction.gameState?.outs }} out{{ prediction.gameState?.outs !== 1 ? 's' : '' }}, 
                      Inning {{ prediction.gameState?.inning }}
                    </div>
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
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div v-if="prediction.actualPitch" class="flex items-center">
                      <span class="text-sm font-medium mr-2">{{ getPitchName(prediction.actualPitch) }}</span>
                      <span v-if="prediction.wasCorrect" class="w-5 h-5 rounded-full bg-green-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                        </svg>
                      </span>
                      <span v-else class="w-5 h-5 rounded-full bg-red-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </span>
                    </div>
                    <div v-else class="text-sm text-gray-500 italic">
                      Not recorded
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <!-- Pagination -->
          <div class="py-4 flex items-center justify-between" v-if="totalPages > 1">
            <div class="flex-1 flex justify-between sm:hidden">
              <button 
                @click="prevPage"
                :disabled="currentPage === 1"
                class="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button 
                @click="nextPage"
                :disabled="currentPage === totalPages"
                class="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
            <div class="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p class="text-sm text-gray-700">
                  Showing <span class="font-medium">{{ (currentPage - 1) * itemsPerPage + 1 }}</span> to 
                  <span class="font-medium">{{ Math.min(currentPage * itemsPerPage, filteredPredictions.length) }}</span> of 
                  <span class="font-medium">{{ filteredPredictions.length }}</span> predictions
                </p>
              </div>
              <div>
                <nav class="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                  <button
                    @click="prevPage"
                    :disabled="currentPage === 1"
                    class="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                  >
                    <span class="sr-only">Previous</span>
                    <!-- ChevronLeft Icon -->
                    <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd" />
                    </svg>
                  </button>
                  
                  <!-- Page numbers -->
                  <template v-for="page in totalPages" :key="page">
                    <button
                      v-if="page === currentPage || 
                             page === 1 || 
                             page === totalPages || 
                             (page >= currentPage - 1 && page <= currentPage + 1)"
                      @click="goToPage(page)"
                      :class="[
                        'relative inline-flex items-center px-4 py-2 border text-sm font-medium',
                        currentPage === page
                          ? 'z-10 bg-baseball-blue border-baseball-blue text-white'
                          : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                      ]"
                    >
                      {{ page }}
                    </button>
                    <span
                      v-else-if="page === currentPage - 2 || page === currentPage + 2"
                      class="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700"
                    >
                      ...
                    </span>
                  </template>
                  
                  <button
                    @click="nextPage"
                    :disabled="currentPage === totalPages"
                    class="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                  >
                    <span class="sr-only">Next</span>
                    <!-- ChevronRight Icon -->
                    <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                    </svg>
                  </button>
                </nav>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template> 