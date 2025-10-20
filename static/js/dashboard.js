// Dashboard JavaScript functionality

// Function to initialize all charts
function initializeCharts() {
    // This function would initialize charts using Chart.js
    // Implementation would depend on the specific data structure
    console.log("Initializing charts...");
}

// Function to fetch model health status
async function fetchModelHealth(modelId) {
    try {
        const response = await fetch(`/api/models/${modelId}/health/`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching model health:', error);
        return null;
    }
}

// Function to update model health display
function updateModelHealthDisplay(modelId, healthData) {
    const statusElement = document.getElementById(`health-status-${modelId}`);
    if (statusElement && healthData) {
        statusElement.textContent = healthData.status;
        
        // Remove all existing classes
        statusElement.classList.remove('bg-gray-100', 'text-gray-800', 'bg-green-100', 'text-green-800', 
                                    'bg-yellow-100', 'text-yellow-800', 'bg-red-100', 'text-red-800');
        
        // Add appropriate classes based on status
        if (healthData.status === 'Healthy') {
            statusElement.classList.add('bg-green-100', 'text-green-800');
        } else if (healthData.status === 'Degraded') {
            statusElement.classList.add('bg-yellow-100', 'text-yellow-800');
        } else {
            statusElement.classList.add('bg-red-100', 'text-red-800');
        }
    }
}

// Function to refresh dashboard data
async function refreshDashboard() {
    // Reload the page to refresh all data
    location.reload();
}

// Function to initialize WebSocket connection for real-time updates
function initializeWebSocket() {
    // In a real implementation, this would connect to a WebSocket server
    // and update the dashboard in real-time
    console.log("Initializing WebSocket connection...");
    
    // Example implementation:
    /*
    try {
        const ws = new WebSocket('ws://localhost:8000/ws/dashboard/');
        
        ws.onopen = function(event) {
            console.log("WebSocket connection opened");
        };
        
        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                // Update dashboard with real-time data
                updateDashboardWithRealTimeData(data);
            } catch (e) {
                console.error('Error parsing WebSocket message:', e);
            }
        };
        
        ws.onclose = function(event) {
            console.log("WebSocket connection closed");
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
    } catch (e) {
        console.error('Error initializing WebSocket:', e);
    }
    */
}

// Function to update dashboard with real-time data
function updateDashboardWithRealTimeData(data) {
    // This function would update the dashboard with real-time data
    // from WebSocket messages
    console.log("Updating dashboard with real-time data:", data);
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard initialized");
    initializeCharts();
    initializeWebSocket();
    
    // Example of fetching and updating model health (would need actual model IDs)
    /*
    const modelElements = document.querySelectorAll('[id^="health-status-"]');
    modelElements.forEach(element => {
        const modelId = element.id.replace('health-status-', '');
        fetchModelHealth(modelId).then(healthData => {
            updateModelHealthDisplay(modelId, healthData);
        });
    });
    */
});

// Export functions for use in other scripts
window.dashboard = {
    initializeCharts,
    fetchModelHealth,
    updateModelHealthDisplay,
    refreshDashboard,
    initializeWebSocket,
    updateDashboardWithRealTimeData
};