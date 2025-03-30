import { createRouter, createWebHistory } from 'vue-router';

// Import views
import HomePage from '../views/HomePage.vue';
import PredictPage from '../views/PredictPage.vue';
import DashboardPage from '../views/DashboardPage.vue';
import HistoryPage from '../views/HistoryPage.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage
  },
  {
    path: '/predict',
    name: 'Predict',
    component: PredictPage
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: DashboardPage
  },
  {
    path: '/history',
    name: 'History',
    component: HistoryPage
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router; 