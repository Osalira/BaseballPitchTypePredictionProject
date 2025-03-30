<script setup>
import { defineProps, defineEmits, computed } from 'vue';

const props = defineProps({
  batterAvg: {
    type: Number,
    default: 0.250
  },
  batterObp: {
    type: Number,
    default: 0.320
  },
  batterSlg: {
    type: Number,
    default: 0.400
  },
  batterFbHitPct: {
    type: Number,
    default: 0.320
  },
  batterCountHitPct: {
    type: Number,
    default: 0.300
  }
});

const emit = defineEmits([
  'update:batterAvg', 
  'update:batterObp',
  'update:batterSlg',
  'update:batterFbHitPct',
  'update:batterCountHitPct'
]);

// Compute the OPS (On-base Plus Slugging)
const batterOps = computed(() => {
  return props.batterObp + props.batterSlg;
});

// Helper function to format batting average
const formatAvg = (value) => {
  return value.toFixed(3).toString().replace(/^0+/, '');
};

// Determine batter skill level based on OPS
const batterSkillLevel = computed(() => {
  const ops = batterOps.value;
  if (ops >= 0.900) return { level: 'Elite', class: 'text-green-600' };
  if (ops >= 0.800) return { level: 'Great', class: 'text-green-500' };
  if (ops >= 0.750) return { level: 'Above Average', class: 'text-blue-500' };
  if (ops >= 0.700) return { level: 'Average', class: 'text-gray-600' };
  if (ops >= 0.650) return { level: 'Below Average', class: 'text-yellow-500' };
  return { level: 'Struggling', class: 'text-red-500' };
});
</script>

<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-bold text-baseball-blue mb-4">Batter Information</h2>
    
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <!-- Batting Average -->
      <div>
        <label class="block text-gray-700 font-medium mb-2">
          Batting Average
        </label>
        <div class="flex items-center">
          <input 
            type="range" 
            v-model="props.batterAvg" 
            @input="$emit('update:batterAvg', parseFloat($event.target.value))" 
            min="0.150" 
            max="0.350" 
            step="0.005"
            class="w-full"
          />
          <span class="ml-2 font-medium w-16 text-right">{{ formatAvg(props.batterAvg) }}</span>
        </div>
      </div>
      
      <!-- On-Base Percentage -->
      <div>
        <label class="block text-gray-700 font-medium mb-2">
          On-Base %
        </label>
        <div class="flex items-center">
          <input 
            type="range" 
            v-model="props.batterObp" 
            @input="$emit('update:batterObp', parseFloat($event.target.value))" 
            min="0.250" 
            max="0.450" 
            step="0.005"
            class="w-full"
          />
          <span class="ml-2 font-medium w-16 text-right">{{ formatAvg(props.batterObp) }}</span>
        </div>
      </div>
      
      <!-- Slugging Percentage -->
      <div>
        <label class="block text-gray-700 font-medium mb-2">
          Slugging %
        </label>
        <div class="flex items-center">
          <input 
            type="range" 
            v-model="props.batterSlg" 
            @input="$emit('update:batterSlg', parseFloat($event.target.value))" 
            min="0.300" 
            max="0.600" 
            step="0.005"
            class="w-full"
          />
          <span class="ml-2 font-medium w-16 text-right">{{ formatAvg(props.batterSlg) }}</span>
        </div>
      </div>
    </div>
    
    <div class="mb-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Fastball Hit Percentage -->
        <div>
          <label class="block text-gray-700 font-medium mb-2">
            Batter's Fastball Hit %
          </label>
          <div class="flex items-center">
            <input 
              type="range" 
              v-model="props.batterFbHitPct" 
              @input="$emit('update:batterFbHitPct', parseFloat($event.target.value))" 
              min="0.200" 
              max="0.400" 
              step="0.005"
              class="w-full"
            />
            <span class="ml-2 font-medium w-16 text-right">{{ formatAvg(props.batterFbHitPct) }}</span>
          </div>
          <p class="text-sm text-gray-500 mt-1">
            How well this batter hits fastballs
          </p>
        </div>
        
        <!-- Count-Specific Hit Percentage -->
        <div>
          <label class="block text-gray-700 font-medium mb-2">
            Batter's Current Count Hit %
          </label>
          <div class="flex items-center">
            <input 
              type="range" 
              v-model="props.batterCountHitPct" 
              @input="$emit('update:batterCountHitPct', parseFloat($event.target.value))" 
              min="0.200" 
              max="0.400" 
              step="0.005"
              class="w-full"
            />
            <span class="ml-2 font-medium w-16 text-right">{{ formatAvg(props.batterCountHitPct) }}</span>
          </div>
          <p class="text-sm text-gray-500 mt-1">
            How well this batter hits in the current count
          </p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-100 p-4 rounded">
      <h3 class="font-semibold mb-2">Batter Profile</h3>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-sm">
            <span class="font-medium">AVG/OBP/SLG:</span> 
            {{ formatAvg(props.batterAvg) }}/{{ formatAvg(props.batterObp) }}/{{ formatAvg(props.batterSlg) }}
          </p>
          <p class="text-sm">
            <span class="font-medium">OPS:</span> 
            {{ formatAvg(batterOps) }}
          </p>
        </div>
        <div>
          <p class="text-sm">
            <span class="font-medium">Skill Level:</span> 
            <span :class="batterSkillLevel.class">{{ batterSkillLevel.level }}</span>
          </p>
          <p class="text-sm">
            <span class="font-medium">FB Hit Rate:</span> 
            {{ formatAvg(props.batterFbHitPct) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template> 