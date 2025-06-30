// Dashboard JavaScript for Health Monitoring
let socket;
let heartRateChart, activityChart;
let dataPoints = [];
let maxDataPoints = 50;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeEventListeners();
    loadUserProfile();
    updateConnectionStatus(false);
});

// Initialize Chart.js charts
function initializeCharts() {
    const heartRateCtx = document.getElementById('heartRateTempChart');
    const activityCtx = document.getElementById('activityHumidityChart');
    
    if (heartRateCtx) {
        heartRateChart = new Chart(heartRateCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Heart Rate (BPM)',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                }, {
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Heart Rate (BPM)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    if (activityCtx) {
        activityChart = new Chart(activityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Activity Level',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                }, {
                    label: 'Humidity (%)',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Activity Level'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Humidity (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Connection buttons
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    
    if (connectBtn) {
        connectBtn.addEventListener('click', connectToDevice);
    }
    
    if (disconnectBtn) {
        disconnectBtn.addEventListener('click', disconnectFromDevice);
    }
    
    // User profile form
    const userProfileForm = document.getElementById('user-profile-form');
    if (userProfileForm) {
        userProfileForm.addEventListener('submit', saveUserProfile);
    }
    
    // Clear log button
    const clearLogBtn = document.getElementById('clear-log');
    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', clearDataLog);
    }
    
    // Tab switching
    const assessmentTab = document.getElementById('assessment-tab');
    if (assessmentTab) {
        assessmentTab.addEventListener('click', function() {
            // Trigger any assessment-specific initialization if needed
            console.log('Switched to assessment tab');
        });
    }
}

// Connect to Arduino device
function connectToDevice() {
    if (socket) {
        socket.disconnect();
    }
    
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus(true);
        
        // Request connection to Arduino
        socket.emit('connect_arduino');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });
    
    socket.on('arduino_data', function(data) {
        console.log('Received Arduino data:', data);
        updateDashboard(data);
        addToDataLog(data);
        updateHealthInsights(data);
    });
    
    socket.on('connection_status', function(status) {
        console.log('Arduino connection status:', status);
        updateConnectionStatus(status.connected);
    });
    
    socket.on('error', function(error) {
        console.error('Socket error:', error);
        showAlert('Connection error: ' + error.message, 'danger');
    });
}

// Disconnect from Arduino device
function disconnectFromDevice() {
    if (socket) {
        socket.emit('disconnect_arduino');
        socket.disconnect();
        socket = null;
    }
    updateConnectionStatus(false);
}

// Update connection status UI
function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');
    
    if (connected) {
        statusIndicator.className = 'status-dot connected';
        statusText.textContent = 'Connected';
        connectBtn.style.display = 'none';
        disconnectBtn.style.display = 'inline-block';
    } else {
        statusIndicator.className = 'status-dot';
        statusText.textContent = 'Disconnected';
        connectBtn.style.display = 'inline-block';
        disconnectBtn.style.display = 'none';
    }
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update real-time values
    if (data.heart_rate !== undefined) {
        document.getElementById('heart-rate-value').textContent = data.heart_rate + ' BPM';
    }
    if (data.temperature !== undefined) {
        document.getElementById('temp-value').textContent = data.temperature + '°C';
    }
    if (data.humidity !== undefined) {
        document.getElementById('humidity-value').textContent = data.humidity + '%';
    }
    if (data.activity_level !== undefined) {
        document.getElementById('activity-value').textContent = data.activity_level;
    }
    
    // Update charts
    updateCharts(data);
}

// Update charts with new data
function updateCharts(data) {
    const timestamp = new Date().toLocaleTimeString();
    
    // Add data point
    dataPoints.push({
        timestamp: timestamp,
        heartRate: data.heart_rate || 0,
        temperature: data.temperature || 0,
        activity: data.activity_level || 0,
        humidity: data.humidity || 0
    });
    
    // Keep only the last maxDataPoints
    if (dataPoints.length > maxDataPoints) {
        dataPoints.shift();
    }
    
    // Update heart rate and temperature chart
    if (heartRateChart) {
        heartRateChart.data.labels = dataPoints.map(d => d.timestamp);
        heartRateChart.data.datasets[0].data = dataPoints.map(d => d.heartRate);
        heartRateChart.data.datasets[1].data = dataPoints.map(d => d.temperature);
        heartRateChart.update('none');
    }
    
    // Update activity and humidity chart
    if (activityChart) {
        activityChart.data.labels = dataPoints.map(d => d.timestamp);
        activityChart.data.datasets[0].data = dataPoints.map(d => d.activity);
        activityChart.data.datasets[1].data = dataPoints.map(d => d.humidity);
        activityChart.update('none');
    }
}

// Add data to the log
function addToDataLog(data) {
    const dataLog = document.getElementById('data-log');
    if (!dataLog) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    let dataValues = [];
    if (data.heart_rate !== undefined) dataValues.push(`HR: ${data.heart_rate} BPM`);
    if (data.temperature !== undefined) dataValues.push(`Temp: ${data.temperature}°C`);
    if (data.humidity !== undefined) dataValues.push(`Humidity: ${data.humidity}%`);
    if (data.activity_level !== undefined) dataValues.push(`Activity: ${data.activity_level}`);
    
    logEntry.innerHTML = `
        <span class="timestamp">${timestamp}</span>
        <span class="value">${dataValues.join(' | ')}</span>
    `;
    
    dataLog.insertBefore(logEntry, dataLog.firstChild);
    
    // Keep only the last 50 entries
    const entries = dataLog.querySelectorAll('.log-entry');
    if (entries.length > 50) {
        dataLog.removeChild(entries[entries.length - 1]);
    }
}

// Clear data log
function clearDataLog() {
    const dataLog = document.getElementById('data-log');
    if (dataLog) {
        dataLog.innerHTML = '';
    }
}

// Save user profile
function saveUserProfile(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const profile = {
        age: formData.get('age'),
        gender: formData.get('gender')
    };
    
    // Save to localStorage
    localStorage.setItem('userProfile', JSON.stringify(profile));
    
    // Send to server
    if (socket) {
        socket.emit('save_profile', profile);
    }
    
    // Update UI
    displayUserProfile(profile);
    showAlert('Profile saved successfully!', 'success');
}

// Load user profile from localStorage
function loadUserProfile() {
    const savedProfile = localStorage.getItem('userProfile');
    if (savedProfile) {
        const profile = JSON.parse(savedProfile);
        displayUserProfile(profile);
        
        // Populate form fields
        const ageInput = document.getElementById('user-age');
        const genderSelect = document.getElementById('user-gender');
        
        if (ageInput) ageInput.value = profile.age || '';
        if (genderSelect) genderSelect.value = profile.gender || '';
    }
}

// Display user profile
function displayUserProfile(profile) {
    const profileInfo = document.getElementById('profile-info');
    const profileDisplay = document.getElementById('profile-display');
    
    if (profileInfo && profileDisplay) {
        profileDisplay.textContent = `Age: ${profile.age}, Gender: ${profile.gender}`;
        profileInfo.style.display = 'block';
    }
}

// Update health insights based on data and user profile
function updateHealthInsights(data) {
    const insightsContainer = document.getElementById('health-insights');
    if (!insightsContainer) return;
    
    const profile = JSON.parse(localStorage.getItem('userProfile') || '{}');
    const insights = [];
    
    // Heart rate insights
    if (data.heart_rate) {
        const hr = data.heart_rate;
        if (hr < 60) {
            insights.push({
                type: 'warning',
                message: 'Heart rate is below normal range (60-100 BPM)',
                icon: 'fas fa-heartbeat'
            });
        } else if (hr > 100) {
            insights.push({
                type: 'warning',
                message: 'Heart rate is above normal range (60-100 BPM)',
                icon: 'fas fa-heartbeat'
            });
        } else {
            insights.push({
                type: 'success',
                message: 'Heart rate is within normal range',
                icon: 'fas fa-heartbeat'
            });
        }
    }
    
    // Temperature insights
    if (data.temperature) {
        const temp = data.temperature;
        if (temp > 37.5) {
            insights.push({
                type: 'danger',
                message: 'Elevated body temperature detected',
                icon: 'fas fa-thermometer-half'
            });
        } else if (temp < 36.0) {
            insights.push({
                type: 'warning',
                message: 'Body temperature is below normal range',
                icon: 'fas fa-thermometer-half'
            });
        }
    }
    
    // Age-specific insights
    if (profile.age) {
        const age = parseInt(profile.age);
        if (age > 65) {
            insights.push({
                type: 'info',
                message: 'As an older adult, monitor hydration and temperature closely',
                icon: 'fas fa-user-clock'
            });
        }
    }
    
    // Display insights
    if (insights.length > 0) {
        insightsContainer.innerHTML = insights.map(insight => `
            <div class="alert alert-${insight.type}">
                <i class="${insight.icon}"></i> ${insight.message}
            </div>
        `).join('');
    } else {
        insightsContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> All vital signs appear normal.
            </div>
        `;
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page (you can customize where to show it)
    const container = document.querySelector('.container-fluid');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Export functions for use in other scripts
window.dashboard = {
    connectToDevice,
    disconnectFromDevice,
    updateDashboard,
    saveUserProfile,
    loadUserProfile,
    showAlert
}; 