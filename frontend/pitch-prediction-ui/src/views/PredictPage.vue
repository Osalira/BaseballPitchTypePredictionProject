<script setup>
import { ref, computed, onMounted } from 'vue';
import GameSituation from '../components/GameSituation.vue';
import PitcherInfo from '../components/PitcherInfo.vue';
import BatterInfo from '../components/BatterInfo.vue';
import PredictionResult from '../components/PredictionResult.vue';
import { db } from '../firebase';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';

// State for game situation
const balls = ref(0);
const strikes = ref(0);
const outs = ref(0);
const inning = ref(1);

// State for pitcher info
const pitcherFbPct = ref(0.6);
const pitcherCountFbPct = ref(0.6);
const pitcherHitterCountFbPct = ref(0.5);
const pitcherPitcherCountFbPct = ref(0.7);
const pitcherNeutralCountFbPct = ref(0.6);

// State for batter info
const batterAvg = ref(0.250);
const batterObp = ref(0.320);
const batterSlg = ref(0.400);
const batterFbHitPct = ref(0.320);
const batterCountHitPct = ref(0.300);

// Prediction state
const prediction = ref(null);
const isLoading = ref(false);
const showSavedMessage = ref(false);

// Computed value for the count type (Hitter, Neutral, Pitcher)
const countType = computed(() => {
  if ((balls.value === 3 && strikes.value < 2) || 
      (balls.value === 2 && strikes.value === 0)) {
    return 'Hitter';
  } else if ((strikes.value === 2 && balls.value < 3) || 
            (strikes.value === 1 && balls.value === 0)) {
    return 'Pitcher';
  } else {
    return 'Neutral';
  }
});

// Mock API call to get prediction
const predictPitch = async () => {
  isLoading.value = true;
  
  try {
    // In a real app, this would be an API call to your ML model backend
    // For demo purposes, we'll simulate with a timeout and mock data
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Create prediction data to simulate model output
    const pitchTypes = ['FB', 'SL', 'CH', 'CU', 'CT'];
    const randomProbabilities = generateRandomProbabilities(pitchTypes.length);
    
    // Sort probabilities in descending order
    const sortedIndices = randomProbabilities
      .map((prob, index) => ({ prob, index }))
      .sort((a, b) => b.prob - a.prob);
    
    // Get top pitch and its probability
    const topPitchIndex = sortedIndices[0].index;
    const topPitch = pitchTypes[topPitchIndex];
    const topPitchProbability = randomProbabilities[topPitchIndex];
    
    // Create object for other pitches and their probabilities
    const otherPitches = {};
    for (let i = 1; i < sortedIndices.length; i++) {
      const { prob, index } = sortedIndices[i];
      otherPitches[pitchTypes[index]] = prob;
    }
    
    // Generate reasoning based on game situation
    const reasoning = generateReasoning(topPitch, countType.value, inning.value);
    
    // Create prediction object
    prediction.value = {
      topPitch,
      topPitchProbability,
      otherPitches,
      reasoning,
      timestamp: Date.now(),
      gameState: {
        balls: balls.value,
        strikes: strikes.value,
        outs: outs.value,
        inning: inning.value,
        countType: countType.value
      },
      batterStats: {
        avg: batterAvg.value,
        obp: batterObp.value,
        slg: batterSlg.value
      },
      pitcherStats: {
        fbPct: pitcherFbPct.value
      }
    };
  } catch (error) {
    console.error('Error predicting pitch:', error);
    // Handle error appropriately (show error message to user)
  } finally {
    isLoading.value = false;
  }
};

// Helper function to generate random probabilities
const generateRandomProbabilities = (count) => {
  // Get random values
  const values = Array.from({ length: count }, () => Math.random());
  // Calculate sum for normalization
  const sum = values.reduce((acc, val) => acc + val, 0);
  // Normalize to ensure sum is 1
  return values.map(val => val / sum);
};

// Generate reasoning based on game situation and selected pitch
const generateReasoning = (pitch, countType, inning) => {
  let reason = '';
  
  // Base reasoning on countType
  if (countType === 'Hitter') {
    if (pitch === 'FB') {
      reason = `The pitcher is throwing a fastball despite the hitter's count (${balls.value}-${strikes.value}), likely trying to challenge the batter directly rather than risking falling behind further.`;
    } else {
      reason = `With the count in the batter's favor (${balls.value}-${strikes.value}), the pitcher is likely trying to induce a ground ball or get the batter to chase with a ${pitchTypes[pitch]}.`;
    }
  } else if (countType === 'Pitcher') {
    if (pitch === 'FB') {
      reason = `With a pitcher's count (${balls.value}-${strikes.value}), the fastball is a safe choice to challenge the batter while still having room for error.`;
    } else {
      reason = `The pitcher has the advantage (${balls.value}-${strikes.value}) and is likely trying to get the batter to chase a ${pitchTypes[pitch]} out of the zone.`;
    }
  } else { // Neutral
    reason = `In a neutral count (${balls.value}-${strikes.value}), the pitcher is likely to throw a ${pitchTypes[pitch]} to try to get ahead or induce weak contact.`;
  }
  
  // Add inning context
  if (inning <= 3) {
    reason += ` Early in the game (inning ${inning}), pitchers often establish their primary pitches.`;
  } else if (inning >= 7) {
    reason += ` Late in the game (inning ${inning}), fatigue and strategy play bigger roles in pitch selection.`;
  }
  
  return reason;
};

// Helper to access pitch type names
const pitchTypes = {
  'FB': 'Fastball',
  'SL': 'Slider',
  'CH': 'Changeup',
  'CU': 'Curveball',
  'CT': 'Cutter'
};

// Save prediction to Firebase
const savePrediction = async () => {
  if (!prediction.value) return;
  
  try {
    // Add to Firestore and get the document reference
    const docRef = await addDoc(collection(db, 'predictions'), {
      ...prediction.value,
      createdAt: serverTimestamp()
    });
    
    // Save the ID in the prediction object for outcome tracking
    prediction.value.id = docRef.id;
    
    // Show saved message
    showSavedMessage.value = true;
    setTimeout(() => {
      showSavedMessage.value = false;
    }, 3000);
  } catch (error) {
    console.error('Error saving prediction:', error);
    // Handle error appropriately
  }
};

// Handle recorded outcome
const handleOutcomeRecorded = (outcomeData) => {
  console.log('Outcome recorded:', outcomeData);
  // You could add additional logic here, like showing a different message
  // or offering to make a new prediction
};

// Reset form to defaults
const resetForm = () => {
  balls.value = 0;
  strikes.value = 0;
  outs.value = 0;
  inning.value = 1;
  pitcherFbPct.value = 0.6;
  pitcherCountFbPct.value = 0.6;
  batterAvg.value = 0.250;
  batterObp.value = 0.320;
  batterSlg.value = 0.400;
  batterFbHitPct.value = 0.320;
  batterCountHitPct.value = 0.300;
  prediction.value = null;
};
</script>

<template>
  <div class="max-w-7xl mx-auto px-4 py-8">
    <h1 class="text-3xl font-bold text-center mb-8 text-baseball-blue">Predict the Next Pitch</h1>
    
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div class="space-y-8">
        <!-- Game Situation Component -->
        <GameSituation
          v-model:balls="balls"
          v-model:strikes="strikes"
          v-model:outs="outs"
          v-model:inning="inning"
        />
        
        <!-- Pitcher Information Component -->
        <PitcherInfo
          v-model:pitcherFbPct="pitcherFbPct"
          v-model:pitcherCountFbPct="pitcherCountFbPct"
          v-model:pitcherHitterCountFbPct="pitcherHitterCountFbPct"
          v-model:pitcherPitcherCountFbPct="pitcherPitcherCountFbPct"
          v-model:pitcherNeutralCountFbPct="pitcherNeutralCountFbPct"
          :countType="countType"
        />
        
        <!-- Batter Information Component -->
        <BatterInfo
          v-model:batterAvg="batterAvg"
          v-model:batterObp="batterObp"
          v-model:batterSlg="batterSlg"
          v-model:batterFbHitPct="batterFbHitPct"
          v-model:batterCountHitPct="batterCountHitPct"
        />
        
        <!-- Action Buttons -->
        <div class="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
          <button 
            @click="predictPitch" 
            class="bg-baseball-blue hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200 flex-1 disabled:opacity-50"
            :disabled="isLoading"
          >
            {{ isLoading ? 'Analyzing...' : 'Predict Pitch' }}
          </button>
          
          <button 
            @click="resetForm" 
            class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-3 px-6 rounded-lg transition-colors duration-200"
          >
            Reset
          </button>
          
          <button 
            v-if="prediction"
            @click="savePrediction" 
            class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200 flex-1"
          >
            Save Prediction
          </button>
        </div>
        
        <!-- Saved Message -->
        <div 
          v-if="showSavedMessage"
          class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative" 
          role="alert"
        >
          <span class="block sm:inline">Prediction saved successfully!</span>
        </div>
      </div>
      
      <!-- Prediction Result Component -->
      <div>
        <PredictionResult
          :prediction="prediction"
          :isLoading="isLoading"
          @outcome-recorded="handleOutcomeRecorded"
        />
      </div>
    </div>
  </div>
</template> 