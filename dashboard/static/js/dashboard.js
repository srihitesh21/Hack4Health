// Initialize Socket.IO connection
const socket = io();

// Chart instances
let tempHumidityChart;
let lightPressureChart;
let heartRateChart;

// Data storage for charts
let chartData = {
    labels: [],
    temperature: [],
    humidity: [],
    light: [],
    pressure: []
};

// Heart rate data storage
let heartRateData = {
    labels: [],
    bpm: []
};

// DOM elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const connectBtn = document.getElementById('connect-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const clearLogBtn = document.getElementById('clear-log');
const dataLog = document.getElementById('data-log');
const requestHeartRateBtn = document.getElementById('request-heart-rate');
const refreshCsvAnalysisBtn = document.getElementById('refresh-csv-analysis');

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    requestInitialData();
    loadCsvAnalysis(); // Load CSV analysis on page load
});

function initializeCharts() {
    // Temperature & Humidity Chart
    const tempHumidityCtx = document.getElementById('tempHumidityChart').getContext('2d');
    tempHumidityChart = new Chart(tempHumidityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Humidity (%)',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });

    // Light & Pressure Chart
    const lightPressureCtx = document.getElementById('lightPressureChart').getContext('2d');
    lightPressureChart = new Chart(lightPressureCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Light',
                    data: [],
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Pressure (hPa)',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });

    // Heart Rate Chart
    const heartRateCtx = document.getElementById('heartRateChart').getContext('2d');
    heartRateChart = new Chart(heartRateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Heart Rate (BPM)',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Heart Rate (BPM)'
                    },
                    min: 40,
                    max: 200,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

function setupEventListeners() {
    // Connect button
    connectBtn.addEventListener('click', function() {
        socket.emit('connect_arduino');
        this.disabled = true;
        this.innerHTML = '<span class="loading"></span> Connecting...';
    });

    // Disconnect button
    disconnectBtn.addEventListener('click', function() {
        socket.emit('disconnect_arduino');
    });

    // Clear log button
    clearLogBtn.addEventListener('click', function() {
        dataLog.innerHTML = '';
    });

    // Request heart rate data button
    requestHeartRateBtn.addEventListener('click', function() {
        socket.emit('request_heart_rate_data');
    });

    // Refresh CSV analysis button
    refreshCsvAnalysisBtn.addEventListener('click', function() {
        refreshCsvAnalysis();
    });
}

function requestInitialData() {
    socket.emit('request_data');
}

function loadCsvAnalysis() {
    // Load CSV analysis from API
    fetch('/api/csv_analysis')
        .then(response => response.json())
        .then(data => {
            if (data && (data.bpm !== undefined || data.analysis_time)) {
                displayCsvAnalysis(data);
            } else {
                // If no analysis available, trigger one
                socket.emit('request_csv_analysis');
            }
        })
        .catch(error => {
            console.error('Error loading CSV analysis:', error);
            document.getElementById('csv-status-display').textContent = 'Error';
        });
}

function refreshCsvAnalysis() {
    const btn = refreshCsvAnalysisBtn;
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    socket.emit('request_csv_analysis');
    
    // Reset button after a delay
    setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }, 3000);
}

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('connection_status', function(data) {
    updateConnectionStatus(data);
});

socket.on('arduino_data', function(data) {
    updateDashboard(data);
    addToDataLog(data);
});

socket.on('sensor_data', function(data) {
    // Load historical data into charts
    if (data.temperature && data.temperature.length > 0) {
        loadHistoricalData(data);
    }
});

// Heart rate data handlers
socket.on('heart_rate_data', function(data) {
    updateHeartRateDisplay(data);
    updateHeartRateChart(data);
    updateSpectrogram(data);
});

socket.on('heart_rate_history', function(data) {
    loadHeartRateHistory(data);
});

// CSV analysis result handler
socket.on('csv_analysis_result', function(data) {
    displayCsvAnalysis(data);
});

function updateConnectionStatus(data) {
    if (data.connected) {
        statusIndicator.classList.add('connected');
        statusText.textContent = 'Connected';
        connectBtn.style.display = 'none';
        disconnectBtn.style.display = 'inline-block';
        
        // Reset connect button
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<i class="fas fa-plug"></i> Connect Arduino';
    } else {
        statusIndicator.classList.remove('connected');
        statusText.textContent = data.message || 'Disconnected';
        connectBtn.style.display = 'inline-block';
        disconnectBtn.style.display = 'none';
        
        // Reset connect button
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<i class="fas fa-plug"></i> Connect Arduino';
    }
}

function updateDashboard(data) {
    // Update sensor values
    if (data.temperature !== undefined) {
        document.getElementById('temp-value').textContent = data.temperature.toFixed(1) + '°C';
    }
    if (data.humidity !== undefined) {
        document.getElementById('humidity-value').textContent = data.humidity.toFixed(1) + '%';
    }
    if (data.light !== undefined) {
        document.getElementById('light-value').textContent = data.light.toFixed(0);
    }
    if (data.pressure !== undefined) {
        document.getElementById('pressure-value').textContent = data.pressure.toFixed(1) + ' hPa';
    }
    if (data.ppg !== undefined) {
        document.getElementById('ppg-value').textContent = data.ppg.toFixed(0);
    }
    
    // Update charts
    updateCharts(data);
    
    // Add to data log
    addToDataLog(data);
}

function updateHeartRateDisplay(data) {
    if (data.bpm !== undefined) {
        document.getElementById('bpm-value').textContent = `${data.bpm.toFixed(1)} BPM`;
    }
}

function updateSpectrogram(data) {
    const spectrogramImage = document.getElementById('spectrogram-image');
    const spectrogramPlaceholder = document.getElementById('spectrogram-placeholder');
    
    if (data.spectrogram) {
        spectrogramImage.src = `data:image/png;base64,${data.spectrogram}`;
        spectrogramImage.style.display = 'block';
        spectrogramPlaceholder.style.display = 'none';
    }
}

function updateCharts(data) {
    const timestamp = new Date().toLocaleTimeString();
    
    // Add new data point
    chartData.labels.push(timestamp);
    if (data.temperature !== undefined) chartData.temperature.push(data.temperature);
    if (data.humidity !== undefined) chartData.humidity.push(data.humidity);
    if (data.light !== undefined) chartData.light.push(data.light);
    if (data.pressure !== undefined) chartData.pressure.push(data.pressure);

    // Keep only last 20 data points
    const maxPoints = 20;
    if (chartData.labels.length > maxPoints) {
        chartData.labels.shift();
        chartData.temperature.shift();
        chartData.humidity.shift();
        chartData.light.shift();
        chartData.pressure.shift();
    }

    // Update Temperature & Humidity Chart
    tempHumidityChart.data.labels = chartData.labels;
    tempHumidityChart.data.datasets[0].data = chartData.temperature;
    tempHumidityChart.data.datasets[1].data = chartData.humidity;
    tempHumidityChart.update('none');

    // Update Light & Pressure Chart
    lightPressureChart.data.labels = chartData.labels;
    lightPressureChart.data.datasets[0].data = chartData.light;
    lightPressureChart.data.datasets[1].data = chartData.pressure;
    lightPressureChart.update('none');
}

function updateHeartRateChart(data) {
    const timestamp = new Date().toLocaleTimeString();
    
    // Add new heart rate data point
    heartRateData.labels.push(timestamp);
    if (data.bpm !== undefined) heartRateData.bpm.push(data.bpm);

    // Keep only last 20 data points
    const maxPoints = 20;
    if (heartRateData.labels.length > maxPoints) {
        heartRateData.labels.shift();
        heartRateData.bpm.shift();
    }

    // Update Heart Rate Chart
    heartRateChart.data.labels = heartRateData.labels;
    heartRateChart.data.datasets[0].data = heartRateData.bpm;
    heartRateChart.update('none');
}

function loadHistoricalData(data) {
    // Load historical data into charts
    const timestamps = data.timestamp.map(ts => new Date(ts * 1000).toLocaleTimeString());
    
    chartData.labels = timestamps.slice(-20); // Last 20 points
    chartData.temperature = data.temperature.slice(-20);
    chartData.humidity = data.humidity.slice(-20);
    chartData.light = data.light.slice(-20);
    chartData.pressure = data.pressure.slice(-20);

    // Update charts with historical data
    tempHumidityChart.data.labels = chartData.labels;
    tempHumidityChart.data.datasets[0].data = chartData.temperature;
    tempHumidityChart.data.datasets[1].data = chartData.humidity;
    tempHumidityChart.update();

    lightPressureChart.data.labels = chartData.labels;
    lightPressureChart.data.datasets[0].data = chartData.light;
    lightPressureChart.data.datasets[1].data = chartData.pressure;
    lightPressureChart.update();
}

function loadHeartRateHistory(data) {
    if (data.bpm && data.bpm.length > 0) {
        const timestamps = data.timestamp.map(ts => new Date(ts * 1000).toLocaleTimeString());
        
        heartRateData.labels = timestamps.slice(-20); // Last 20 points
        heartRateData.bpm = data.bpm.slice(-20);

        // Update heart rate chart with historical data
        heartRateChart.data.labels = heartRateData.labels;
        heartRateChart.data.datasets[0].data = heartRateData.bpm;
        heartRateChart.update();
    }
}

function addToDataLog(data) {
    const timestamp = new Date().toLocaleString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    let dataValues = '';
    if (data.temperature !== undefined) {
        dataValues += `<span class="data-item">Temp: ${data.temperature.toFixed(1)}°C</span>`;
    }
    if (data.humidity !== undefined) {
        dataValues += `<span class="data-item">Humidity: ${data.humidity.toFixed(1)}%</span>`;
    }
    if (data.light !== undefined) {
        dataValues += `<span class="data-item">Light: ${data.light.toFixed(0)}</span>`;
    }
    if (data.pressure !== undefined) {
        dataValues += `<span class="data-item">Pressure: ${data.pressure.toFixed(1)} hPa</span>`;
    }
    if (data.ppg !== undefined) {
        dataValues += `<span class="data-item">PPG: ${data.ppg.toFixed(0)}</span>`;
    }
    if (data.bpm !== undefined) {
        dataValues += `<span class="data-item">BPM: ${data.bpm.toFixed(1)}</span>`;
    }
    
    logEntry.innerHTML = `
        <div class="timestamp">${timestamp}</div>
        <div class="data-values">${dataValues}</div>
    `;
    
    dataLog.insertBefore(logEntry, dataLog.firstChild);
    
    // Keep only last 50 log entries
    while (dataLog.children.length > 50) {
        dataLog.removeChild(dataLog.lastChild);
    }
}

function displayCsvAnalysis(data) {
    // Update analysis results
    document.getElementById('csv-bpm-display').textContent = data.bpm.toFixed(1);
    
    // Update main BPM value in heart rate card with estimated BPM
    document.getElementById('bpm-value').textContent = `${data.bpm.toFixed(1)} BPM`;
    
    if (data.analysis_time) {
        // analysis_time is already a formatted string from the backend
        document.getElementById('csv-time-display').textContent = data.analysis_time;
    }
    
    document.getElementById('csv-status-display').textContent = 'Complete';
    document.getElementById('csv-status-display').style.color = '#27ae60';
    
    // Display time domain plot
    if (data.time_domain_b64) {
        const timeDomainImage = document.getElementById('time-domain-image');
        const timeDomainPlaceholder = document.getElementById('time-domain-placeholder');
        
        timeDomainImage.src = `data:image/png;base64,${data.time_domain_b64}`;
        timeDomainImage.style.display = 'block';
        timeDomainPlaceholder.style.display = 'none';
    }
    
    // Display spectrogram
    if (data.spectrogram_b64) {
        const csvSpectrogramImage = document.getElementById('csv-spectrogram-image');
        const csvSpectrogramPlaceholder = document.getElementById('csv-spectrogram-placeholder');
        
        csvSpectrogramImage.src = `data:image/png;base64,${data.spectrogram_b64}`;
        csvSpectrogramImage.style.display = 'block';
        csvSpectrogramPlaceholder.style.display = 'none';
    }
}

// Handle window resize for responsive charts
window.addEventListener('resize', function() {
    if (tempHumidityChart) {
        tempHumidityChart.resize();
    }
    if (lightPressureChart) {
        lightPressureChart.resize();
    }
    if (heartRateChart) {
        heartRateChart.resize();
    }
}); 