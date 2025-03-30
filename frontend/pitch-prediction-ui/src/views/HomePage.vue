<script setup>
import { RouterLink } from 'vue-router';
</script>

<template>
  <div class="bg-white p-8 rounded-lg shadow-md">
    <h1 class="text-3xl font-bold text-baseball-blue mb-6">Baseball Pitch Prediction</h1>
    
    <div class="mb-8">
      <p class="text-lg mb-4">
        Welcome to the Baseball Pitch Prediction tool, powered by machine learning to predict whether the next pitch will be a fastball or an offspeed pitch.
      </p>
      <p class="mb-4">
        This application uses data from MLB games spanning multiple seasons to provide accurate predictions based on game situations, pitcher tendencies, and batter performance.
      </p>
    </div>
    
    <div class="grid md:grid-cols-3 gap-6 mb-8">
      <div class="bg-gray-50 p-6 rounded-lg shadow border border-gray-200">
        <h2 class="text-xl font-semibold text-baseball-blue mb-3">Make Predictions</h2>
        <p class="mb-4">Input game situations and get real-time predictions about the next pitch type.</p>
        <RouterLink to="/predict" class="inline-block bg-baseball-blue text-white py-2 px-4 rounded hover:bg-blue-700 transition">
          Start Predicting
        </RouterLink>
      </div>
      
      <div class="bg-gray-50 p-6 rounded-lg shadow border border-gray-200">
        <h2 class="text-xl font-semibold text-baseball-blue mb-3">View Dashboard</h2>
        <p class="mb-4">Explore model performance metrics and feature importance visualizations.</p>
        <RouterLink to="/dashboard" class="inline-block bg-baseball-blue text-white py-2 px-4 rounded hover:bg-blue-700 transition">
          See Analytics
        </RouterLink>
      </div>
      
      <div class="bg-gray-50 p-6 rounded-lg shadow border border-gray-200">
        <h2 class="text-xl font-semibold text-baseball-blue mb-3">Prediction History</h2>
        <p class="mb-4">Review past predictions and track model performance over time.</p>
        <RouterLink to="/history" class="inline-block bg-baseball-blue text-white py-2 px-4 rounded hover:bg-blue-700 transition">
          View History
        </RouterLink>
      </div>
    </div>
    
    <div class="bg-gray-50 p-6 rounded-lg shadow border border-gray-200">
      <h2 class="text-xl font-semibold text-baseball-blue mb-3">About the Model</h2>
      <p class="mb-2">This tool uses advanced machine learning models trained on MLB Statcast data.</p>
      <p class="mb-2">Key features used for prediction include:</p>
      <ul class="list-disc pl-6 mb-4">
        <li>Game situation (count, outs, inning)</li>
        <li>Pitcher tendencies (overall fastball percentage, fastball percentage by count)</li>
        <li>Batter performance (success against fastball vs. offspeed pitches)</li>
        <li>Previous pitch information (sequence patterns)</li>
      </ul>
      <p>Our XGBoost model achieves over 90% accuracy in predicting pitch types.</p>
    </div>
  </div>
</template> 