<script setup>
import { RouterView, RouterLink } from 'vue-router';
</script>

<template>
  <div class="min-h-screen bg-gray-100">
    <!-- Navigation -->
    <nav class="bg-baseball-blue text-white shadow-md">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div class="flex-shrink-0 flex items-center">
              <h1 class="text-2xl font-bold">Pitch Prediction</h1>
            </div>
            <div class="ml-10 flex items-baseline space-x-4">
              <RouterLink to="/" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-baseball-blue-dark" active-class="bg-baseball-blue-dark">
                Home
              </RouterLink>
              <RouterLink to="/predict" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-baseball-blue-dark" active-class="bg-baseball-blue-dark">
                Make Prediction
              </RouterLink>
              <RouterLink to="/dashboard" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-baseball-blue-dark" active-class="bg-baseball-blue-dark">
                Dashboard
              </RouterLink>
              <RouterLink to="/history" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-baseball-blue-dark" active-class="bg-baseball-blue-dark">
                History
              </RouterLink>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Page Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <RouterView />
    </main>
  </div>
</template>

<style>
@import './assets/tailwind.css';
</style>
