<script setup>
import { defineProps, defineEmits, ref } from 'vue';
import { doc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../firebase';

const props = defineProps({
  prediction: {
    type: Object,
    default: () => null
  },
  isLoading: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(['outcome-recorded']);

// State for tracking actual outcome
const actualPitch = ref(null);
const isOutcomeSaving = ref(false);
const outcomeSaved = ref(false);
const outcomeError = ref(null);

// Function to determine color class based on probability
const getProbabilityColorClass = (probability) => {
  if (probability >= 0.8) return 'text-green-600';
  if (probability >= 0.6) return 'text-green-500';
  if (probability >= 0.4) return 'text-yellow-500';
  return 'text-red-500';
};

// Function to format probability as percentage
const formatProbability = (probability) => {
  return (probability * 100).toFixed(1) + '%';
};

// Map pitch types to more readable names
const pitchTypeNames = {
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

// Get full pitch name
const getPitchName = (pitchType) => {
  return pitchTypeNames[pitchType] || pitchType;
};

// Record the actual pitch outcome
const recordActualPitch = async (pitchType) => {
  if (!props.prediction || !props.prediction.id) {
    outcomeError.value = "Cannot record outcome: prediction has no ID";
    return;
  }
  
  actualPitch.value = pitchType;
  isOutcomeSaving.value = true;
  outcomeError.value = null;
  
  try {
    // Determine if prediction was correct
    const wasCorrect = props.prediction.topPitch === pitchType;
    
    // Update prediction document in Firestore
    await updateDoc(doc(db, 'predictions', props.prediction.id), {
      actualPitch: pitchType,
      wasCorrect: wasCorrect,
      verifiedAt: serverTimestamp(),
      accuracy: wasCorrect ? 1 : 0  // For aggregation purposes
    });
    
    outcomeSaved.value = true;
    emit('outcome-recorded', {
      predictionId: props.prediction.id,
      actualPitch: pitchType,
      wasCorrect: wasCorrect
    });
    
    // Reset after 3 seconds
    setTimeout(() => {
      outcomeSaved.value = false;
    }, 3000);
  } catch (error) {
    console.error('Error saving actual outcome:', error);
    outcomeError.value = `Error saving outcome: ${error.message}`;
  } finally {
    isOutcomeSaving.value = false;
  }
};
</script>

<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-bold text-baseball-blue mb-4">Pitch Prediction</h2>
    
    <div v-if="isLoading" class="flex justify-center items-center py-8">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-baseball-red"></div>
      <span class="ml-3 text-gray-700">Analyzing game situation...</span>
    </div>
    
    <div v-else-if="!prediction" class="bg-gray-100 p-4 rounded mb-4">
      <p class="text-center text-gray-600">
        Enter game situation and player data, then click "Predict Pitch" to see the most likely next pitch.
      </p>
    </div>
    
    <div v-else>
      <!-- Main Prediction -->
      <div class="bg-gray-100 p-4 rounded mb-6">
        <div class="text-center">
          <h3 class="text-xl font-bold mb-2">Most Likely Pitch</h3>
          <div class="text-3xl font-bold text-baseball-blue mb-2">
            {{ getPitchName(prediction.topPitch) }}
          </div>
          <div class="text-lg" :class="getProbabilityColorClass(prediction.topPitchProbability)">
            {{ formatProbability(prediction.topPitchProbability) }} probability
          </div>
        </div>
      </div>
      
      <!-- Other Predictions -->
      <h3 class="font-bold text-lg mb-3">Other Possible Pitches</h3>
      <div class="space-y-2">
        <div v-for="(prob, pitch) in prediction.otherPitches" :key="pitch" 
             class="flex justify-between items-center p-2 border-b border-gray-200">
          <span class="font-medium">{{ getPitchName(pitch) }}</span>
          <span :class="getProbabilityColorClass(prob)">{{ formatProbability(prob) }}</span>
        </div>
      </div>
      
      <!-- Reasoning -->
      <div class="mt-6 bg-blue-50 p-4 rounded">
        <h3 class="font-bold text-lg mb-2">Prediction Reasoning</h3>
        <p class="text-sm text-gray-700">
          {{ prediction.reasoning || 'Based on the current game situation and pitcher/batter tendencies, the model predicts the most likely pitch type.' }}
        </p>
      </div>
      
      <!-- Actual Outcome Section -->
      <div class="mt-6 border-t pt-4">
        <h3 class="font-bold text-lg mb-3">Record Actual Outcome</h3>
        <p v-if="prediction.actualPitch" class="mb-3">
          <span class="font-medium">Actual pitch thrown:</span> 
          <span class="ml-2 px-3 py-1 rounded-full" 
                :class="prediction.wasCorrect ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'">
            {{ getPitchName(prediction.actualPitch) }}
            <span v-if="prediction.wasCorrect" class="ml-1">✓</span>
            <span v-else class="ml-1">✗</span>
          </span>
        </p>
        <div v-else>
          <p class="text-sm text-gray-600 mb-3">
            After the pitch is thrown, record the actual outcome to help improve the model.
          </p>
          <div class="flex flex-wrap gap-2">
            <button 
              v-for="(name, type) in pitchTypeNames" 
              :key="type"
              @click="recordActualPitch(type)"
              :disabled="isOutcomeSaving"
              class="px-3 py-1 rounded border text-sm font-medium hover:bg-gray-100 transition-colors duration-200"
              :class="{
                'bg-green-100 border-green-500 text-green-800': actualPitch === type,
                'border-gray-300': actualPitch !== type
              }"
            >
              {{ name }}
            </button>
          </div>
          
          <!-- Success Message -->
          <div v-if="outcomeSaved" class="mt-3 p-2 bg-green-100 text-green-800 text-sm rounded">
            Outcome recorded successfully! Thank you for your feedback.
          </div>
          
          <!-- Error Message -->
          <div v-if="outcomeError" class="mt-3 p-2 bg-red-100 text-red-800 text-sm rounded">
            {{ outcomeError }}
          </div>
          
          <!-- Loading State -->
          <div v-if="isOutcomeSaving" class="mt-3 flex items-center text-sm text-gray-600">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-baseball-blue mr-2"></div>
            Saving outcome...
          </div>
        </div>
      </div>
      
      <!-- Timestamp -->
      <div class="mt-4 text-right text-xs text-gray-500">
        Prediction made at: {{ new Date(prediction.timestamp).toLocaleString() }}
      </div>
    </div>
  </div>
</template> 