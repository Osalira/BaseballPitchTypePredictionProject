<script setup>
import { defineProps, defineEmits, computed } from 'vue';

const props = defineProps({
  balls: {
    type: Number,
    default: 0
  },
  strikes: {
    type: Number,
    default: 0
  },
  outs: {
    type: Number,
    default: 0
  },
  inning: {
    type: Number,
    default: 1
  }
});

const emit = defineEmits(['update:balls', 'update:strikes', 'update:outs', 'update:inning']);

// Computed property to determine count type
const countType = computed(() => {
  const count = `${props.balls}-${props.strikes}`;
  const hitterCounts = ['1-0', '2-0', '3-0', '2-1', '3-1', '3-2'];
  const pitcherCounts = ['0-1', '0-2', '1-2', '2-2'];
  
  if (hitterCounts.includes(count)) return 'Hitter';
  if (pitcherCounts.includes(count)) return 'Pitcher';
  return 'Neutral';
});

// Computed property to determine inning stage
const inningStage = computed(() => {
  if (props.inning <= 3) return 'Early';
  if (props.inning <= 6) return 'Middle';
  return 'Late';
});
</script>

<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-bold text-baseball-blue mb-4">Game Situation</h2>
    
    <div class="grid md:grid-cols-2 gap-6">
      <!-- Count Section -->
      <div>
        <label class="block text-gray-700 font-medium mb-2">Count</label>
        <div class="flex items-center mb-4">
          <div class="mr-6">
            <label class="block text-sm text-gray-500 mb-1">Balls</label>
            <select v-model="props.balls" @change="$emit('update:balls', parseInt($event.target.value))" class="border rounded py-1 px-2 w-16">
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </div>
          <div>
            <label class="block text-sm text-gray-500 mb-1">Strikes</label>
            <select v-model="props.strikes" @change="$emit('update:strikes', parseInt($event.target.value))" class="border rounded py-1 px-2 w-16">
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
            </select>
          </div>
        </div>
        
        <div class="bg-gray-100 p-3 rounded mb-4">
          <div class="flex justify-between">
            <span class="text-sm text-gray-600">Current Count:</span>
            <span class="font-semibold text-baseball-blue">{{ props.balls }}-{{ props.strikes }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-sm text-gray-600">Count Type:</span>
            <span :class="{
              'font-semibold text-green-600': countType === 'Hitter',
              'font-semibold text-red-600': countType === 'Pitcher',
              'font-semibold text-gray-600': countType === 'Neutral'
            }">{{ countType }} Count</span>
          </div>
        </div>
      </div>
      
      <!-- Game State Section -->
      <div>
        <div class="mb-4">
          <label class="block text-gray-700 font-medium mb-2">Outs</label>
          <select v-model="props.outs" @change="$emit('update:outs', parseInt($event.target.value))" class="border rounded py-1 px-2 w-full">
            <option value="0">0 Out</option>
            <option value="1">1 Out</option>
            <option value="2">2 Outs</option>
          </select>
        </div>
        
        <div>
          <label class="block text-gray-700 font-medium mb-2">Inning</label>
          <div class="flex items-center">
            <input 
              type="number" 
              v-model="props.inning" 
              @input="$emit('update:inning', parseInt($event.target.value))" 
              min="1" 
              max="12"
              class="border rounded py-1 px-2 w-16"
            />
            <span class="ml-4 px-2 py-1 rounded text-sm" :class="{
              'bg-green-100 text-green-800': inningStage === 'Early',
              'bg-yellow-100 text-yellow-800': inningStage === 'Middle',
              'bg-red-100 text-red-800': inningStage === 'Late'
            }">{{ inningStage }} Game</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template> 