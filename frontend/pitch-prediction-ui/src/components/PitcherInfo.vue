<script setup>
import { defineProps, defineEmits, computed } from 'vue';

const props = defineProps({
  pitcherFbPct: {
    type: Number,
    default: 0.6
  },
  pitcherCountFbPct: {
    type: Number,
    default: 0.6
  },
  countType: {
    type: String,
    default: 'Neutral'
  },
  pitcherHitterCountFbPct: {
    type: Number,
    default: 0.5
  },
  pitcherPitcherCountFbPct: {
    type: Number,
    default: 0.7
  },
  pitcherNeutralCountFbPct: {
    type: Number,
    default: 0.6
  }
});

const emit = defineEmits([
  'update:pitcherFbPct', 
  'update:pitcherCountFbPct',
  'update:pitcherHitterCountFbPct',
  'update:pitcherPitcherCountFbPct',
  'update:pitcherNeutralCountFbPct'
]);

// Helper function to format percentage
const formatPct = (value) => {
  return (value * 100).toFixed(1) + '%';
};

// Computed property to determine which count-specific percentage to show
const relevantCountPct = computed(() => {
  if (props.countType === 'Hitter') return props.pitcherHitterCountFbPct;
  if (props.countType === 'Pitcher') return props.pitcherPitcherCountFbPct;
  return props.pitcherNeutralCountFbPct;
});
</script>

<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-bold text-baseball-blue mb-4">Pitcher Information</h2>
    
    <div class="mb-4">
      <label class="block text-gray-700 font-medium mb-2">
        Pitcher's Overall Fastball Percentage
      </label>
      <div class="flex items-center">
        <input 
          type="range" 
          v-model="props.pitcherFbPct" 
          @input="$emit('update:pitcherFbPct', parseFloat($event.target.value))" 
          min="0" 
          max="1" 
          step="0.05"
          class="w-full"
        />
        <span class="ml-4 font-medium">{{ formatPct(props.pitcherFbPct) }}</span>
      </div>
      <p class="text-sm text-gray-500 mt-1">
        How often this pitcher throws fastballs in general
      </p>
    </div>
    
    <div class="mb-4">
      <label class="block text-gray-700 font-medium mb-2">
        Pitcher's Fastball Percentage in Current Count
      </label>
      <div class="flex items-center">
        <input 
          type="range" 
          v-model="props.pitcherCountFbPct" 
          @input="$emit('update:pitcherCountFbPct', parseFloat($event.target.value))" 
          min="0" 
          max="1" 
          step="0.05"
          class="w-full"
        />
        <span class="ml-4 font-medium">{{ formatPct(props.pitcherCountFbPct) }}</span>
      </div>
      <p class="text-sm text-gray-500 mt-1">
        How often this pitcher throws fastballs in the current count
      </p>
    </div>
    
    <div class="mb-4" v-if="countType === 'Hitter'">
      <label class="block text-gray-700 font-medium mb-2">
        Pitcher's Fastball Percentage in Hitter Counts
      </label>
      <div class="flex items-center">
        <input 
          type="range" 
          v-model="props.pitcherHitterCountFbPct" 
          @input="$emit('update:pitcherHitterCountFbPct', parseFloat($event.target.value))" 
          min="0" 
          max="1" 
          step="0.05"
          class="w-full"
        />
        <span class="ml-4 font-medium">{{ formatPct(props.pitcherHitterCountFbPct) }}</span>
      </div>
    </div>
    
    <div class="mb-4" v-if="countType === 'Pitcher'">
      <label class="block text-gray-700 font-medium mb-2">
        Pitcher's Fastball Percentage in Pitcher Counts
      </label>
      <div class="flex items-center">
        <input 
          type="range" 
          v-model="props.pitcherPitcherCountFbPct" 
          @input="$emit('update:pitcherPitcherCountFbPct', parseFloat($event.target.value))" 
          min="0" 
          max="1" 
          step="0.05"
          class="w-full"
        />
        <span class="ml-4 font-medium">{{ formatPct(props.pitcherPitcherCountFbPct) }}</span>
      </div>
    </div>
    
    <div class="mb-4" v-if="countType === 'Neutral'">
      <label class="block text-gray-700 font-medium mb-2">
        Pitcher's Fastball Percentage in Neutral Counts
      </label>
      <div class="flex items-center">
        <input 
          type="range" 
          v-model="props.pitcherNeutralCountFbPct" 
          @input="$emit('update:pitcherNeutralCountFbPct', parseFloat($event.target.value))" 
          min="0" 
          max="1" 
          step="0.05"
          class="w-full"
        />
        <span class="ml-4 font-medium">{{ formatPct(props.pitcherNeutralCountFbPct) }}</span>
      </div>
    </div>
    
    <div class="bg-gray-100 p-3 rounded">
      <p class="text-sm">
        <span class="font-medium">Pitcher Profile:</span> 
        This pitcher throws fastballs {{ formatPct(props.pitcherFbPct) }} of the time overall, 
        and {{ formatPct(relevantCountPct) }} in {{ countType.toLowerCase() }} counts.
      </p>
    </div>
  </div>
</template> 