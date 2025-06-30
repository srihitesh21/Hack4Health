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

// Hospital search functionality
let hospitalMap = null;
let hospitalMarker = null;
let userMarker = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    requestInitialData();
    loadCSVAnalysis(); // Load CSV analysis on page load
    
    // Setup heatstroke prediction
    setupHeatstrokePrediction();
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

function loadCSVAnalysis() {
    fetch('/api/csv_analysis')
        .then(response => response.json())
        .then(data => {
            updateCSVAnalysis(data);
        })
        .catch(error => {
            console.error('Error loading CSV analysis:', error);
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
    
    // Update risk scores if available
    if (data.infection_score !== undefined) {
        updateRiskCard('infectionRisk', 'infectionIndicator', data.infection_score, 'Infection');
    }
    if (data.dehydration_score !== undefined) {
        updateRiskCard('dehydrationRisk', 'dehydrationIndicator', data.dehydration_score, 'Dehydration');
    }
    if (data.arrhythmia_score !== undefined) {
        updateRiskCard('arrhythmiaRisk', 'arrhythmiaIndicator', data.arrhythmia_score, 'Arrhythmia');
    }
    
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

function updateCSVAnalysis(data) {
    if (data.bpm > 0) {
        document.getElementById('heartRate').textContent = data.bpm.toFixed(1);
        document.getElementById('heartRate').style.color = '#dc3545';
        document.querySelector('.heartbeat-animation').style.display = 'block';
    } else {
        document.getElementById('heartRate').textContent = '--';
        document.getElementById('heartRate').style.color = '#6c757d';
        document.querySelector('.heartbeat-animation').style.display = 'none';
    }
    
    // Update risk scores
    updateRiskCard('infectionRisk', 'infectionIndicator', data.infection_score, 'Infection');
    updateRiskCard('dehydrationRisk', 'dehydrationIndicator', data.dehydration_score, 'Dehydration');
    updateRiskCard('arrhythmiaRisk', 'arrhythmiaIndicator', data.arrhythmia_score, 'Arrhythmia');
    
    // Update analysis time
    if (data.analysis_time) {
        const analysisTime = new Date(data.analysis_time * 1000).toLocaleString();
        document.getElementById('analysisTime').textContent = analysisTime;
    }
    
    // Update plots if available
    if (data.time_domain_plot) {
        document.getElementById('timeDomainPlot').src = 'data:image/png;base64,' + data.time_domain_plot;
        document.getElementById('timeDomainPlot').style.display = 'block';
    }
    
    if (data.spectrogram_plot) {
        document.getElementById('spectrogramPlot').src = 'data:image/png;base64,' + data.spectrogram_plot;
        document.getElementById('spectrogramPlot').style.display = 'block';
    }
}

function updateRiskCard(valueId, indicatorId, score, riskType) {
    const valueElement = document.getElementById(valueId);
    const indicatorElement = document.getElementById(indicatorId);
    const cardElement = indicatorElement.closest('.risk-card');
    
    // Update value to show "High Chance" or "Low Chance" instead of numerical score
    if (score >= 1.0) {
        valueElement.textContent = 'High Chance';
        valueElement.style.color = '#dc3545';
        indicatorElement.className = 'risk-indicator high-risk';
        cardElement.className = 'card risk-card high-risk';
    } else {
        valueElement.textContent = 'Low Chance';
        valueElement.style.color = '#28a745';
        indicatorElement.className = 'risk-indicator low-risk';
        cardElement.className = 'card risk-card low-risk';
    }
    
    // Add tooltip with explanation
    const tooltipText = getRiskExplanation(score, riskType);
    cardElement.title = tooltipText;
}

function getRiskExplanation(score, riskType) {
    if (score >= 1.0) {
        switch (riskType) {
            case 'Infection':
                return 'High chance of infection detected. Elevated heart rate (>90 BPM) and/or elevated skin temperature (>37.5°C) may indicate fever or infection.';
            case 'Dehydration':
                return 'High chance of dehydration detected. Elevated heart rate with low variability and/or low skin temperature (<35.5°C) suggests poor circulation.';
            case 'Arrhythmia':
                return 'High chance of arrhythmia detected. Abnormal heart rate patterns detected (too low, too high, or irregular).';
            default:
                return 'High risk detected.';
        }
    } else {
        switch (riskType) {
            case 'Infection':
                return 'Low chance of infection. Heart rate and skin temperature within normal ranges.';
            case 'Dehydration':
                return 'Low chance of dehydration. Heart rate variability and skin temperature appear normal.';
            case 'Arrhythmia':
                return 'Low chance of arrhythmia. Heart rate patterns appear normal and regular.';
            default:
                return 'Low risk detected.';
        }
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

function findNearestHospital() {
    const locationInput = document.getElementById('locationInput').value.trim();
    
    if (!locationInput) {
        alert('Please enter a location to search for hospitals.');
        return;
    }
    
    const hospitalInfo = document.getElementById('hospitalInfo');
    hospitalInfo.innerHTML = '<p>Searching for hospitals near: ' + locationInput + '...</p>';
    
    console.log('Searching for location:', locationInput);
    
    // Immediately use fallback system since external APIs are rate-limited
    console.log('Using fallback hospital data for immediate results');
    
    // Use Vancouver coordinates as fallback
    const userLat = 49.2827;
    const userLon = -123.1207;
    
    // Display sample hospital data immediately
    displaySampleHospitalInfo(locationInput, userLat, userLon);
}

function geocodeLocation(address) {
    return new Promise((resolve, reject) => {
        // Skip external API call and use fallback immediately
        console.log('Using fallback coordinates for:', address);
        resolve({ lat: 49.2827, lon: -123.1207 });
    });
}

function searchNearbyHospitals(lat, lon, userLocation) {
    // Skip external API call and use fallback immediately
    console.log('Using fallback hospital data');
    displaySampleHospitalInfo(userLocation, lat, lon);
}

function displaySampleHospitalInfo(userLocation, userLat, userLon) {
    const hospitalInfo = document.getElementById('hospitalInfo');
    
    // Sample hospital data for Vancouver area
    const sampleHospitals = [
        {
            name: "Vancouver General Hospital",
            address: "899 W 12th Ave, Vancouver, BC V5Z 1M9",
            phone: "(604) 875-4111",
            website: "https://vch.ca/locations-services/result?res_id=1",
            lat: 49.2627,
            lon: -123.1234,
            distance: 2.1
        },
        {
            name: "St. Paul's Hospital",
            address: "1081 Burrard St, Vancouver, BC V6Z 1Y6",
            phone: "(604) 682-2344",
            website: "https://www.providencehealthcare.org/hospitals-residences/st-pauls-hospital",
            lat: 49.2817,
            lon: -123.1307,
            distance: 0.8
        },
        {
            name: "UBC Hospital",
            address: "2211 Wesbrook Mall, Vancouver, BC V6T 2B5",
            phone: "(604) 822-7121",
            website: "https://www.ubchospital.com/",
            lat: 49.2527,
            lon: -123.2434,
            distance: 3.2
        },
        {
            name: "Mount Saint Joseph Hospital",
            address: "3080 Prince Edward St, Vancouver, BC V5T 3N4",
            phone: "(604) 874-1141",
            website: "https://www.providencehealthcare.org/hospitals-residences/mount-saint-joseph-hospital",
            lat: 49.2487,
            lon: -123.0894,
            distance: 1.5
        }
    ];
    
    // Find closest sample hospital
    const closest = sampleHospitals.reduce((closest, hospital) => {
        const distance = calculateDistance(userLat, userLon, hospital.lat, hospital.lon);
        return distance < closest.distance ? { ...hospital, distance } : closest;
    }, { distance: Infinity });
    
    // Display hospital information
    hospitalInfo.innerHTML = `
        <div class="hospital-details">
            <h4>🏥 ${closest.name}</h4>
            <p><strong>📍 Address:</strong> ${closest.address}</p>
            <p><strong>📞 Phone:</strong> <a href="tel:${closest.phone}" style="color: white;">${closest.phone}</a></p>
            <p><strong>🌐 Website:</strong> <a href="${closest.website}" target="_blank" style="color: white;">Visit Website</a></p>
            <p><strong>📏 Distance:</strong> ${closest.distance.toFixed(1)} km from your location</p>
            <p><strong>🚨 Emergency:</strong> Call 911 for immediate assistance</p>
        </div>
    `;
    
    // Update the map
    updateHospitalMap(userLat, userLon, closest.lat, closest.lon, closest.name, userLocation);
    
    // Add note about demo data
    hospitalInfo.innerHTML += '<p style="font-size: 0.8em; color: #666; margin-top: 10px;"><em>Note: Showing sample hospital data for demonstration purposes.</em></p>';
}

function displayHospitalInfo(hospital, userLocation, userLat, userLon) {
    const hospitalInfo = document.getElementById('hospitalInfo');
    
    let hospitalLat, hospitalLon;
    if (hospital.lat && hospital.lon) {
        hospitalLat = hospital.lat;
        hospitalLon = hospital.lon;
    } else if (hospital.center) {
        hospitalLat = hospital.center.lat;
        hospitalLon = hospital.center.lon;
    }
    
    const distance = hospital.distance.toFixed(1);
    
    hospitalInfo.innerHTML = `
        <div class="hospital-details">
            <h4>${hospital.tags.name}</h4>
            <p><strong>Address:</strong> ${hospital.tags['addr:street'] || 'Address not available'}</p>
            <p class="distance"><strong>Distance:</strong> ${distance} km</p>
            <p class="phone"><strong>Phone:</strong> ${hospital.tags.phone || 'Phone not available'}</p>
            ${hospital.tags.website ? `<p><strong>Website:</strong> <a href="${hospital.tags.website}" target="_blank" style="color: #4caf50;">Visit Website</a></p>` : ''}
        </div>
    `;
    
    // Update map
    updateHospitalMap(userLat, userLon, hospitalLat, hospitalLon, hospital.tags.name, userLocation);
}

function updateHospitalMap(userLat, userLon, hospitalLat, hospitalLon, hospitalName, userLocation) {
    try {
        // Check if Leaflet is available
        if (typeof L === 'undefined') {
            console.error('Leaflet is not loaded');
            return;
        }
        
        if (!hospitalMap) {
            hospitalMap = L.map('hospitalMap').setView([userLat, userLon], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(hospitalMap);
        } else {
            hospitalMap.setView([userLat, userLon], 13);
        }
        
        // Clear existing markers
        hospitalMap.eachLayer((layer) => {
            if (layer instanceof L.Marker) {
                hospitalMap.removeLayer(layer);
            }
        });
        
        // Add user location marker
        userMarker = L.marker([userLat, userLon])
            .addTo(hospitalMap)
            .bindPopup(`<b>Your Location:</b><br>${userLocation}`)
            .openPopup();
        
        // Add hospital marker
        hospitalMarker = L.marker([hospitalLat, hospitalLon])
            .addTo(hospitalMap)
            .bindPopup(`<b>Hospital:</b><br>${hospitalName}`)
            .openPopup();
        
        // Draw route line
        const routeLine = L.polyline([[userLat, userLon], [hospitalLat, hospitalLon]], {
            color: 'red',
            weight: 3,
            opacity: 0.7,
            dashArray: '10, 10'
        }).addTo(hospitalMap);
        
        // Fit map to show both markers
        const bounds = L.latLngBounds([[userLat, userLon], [hospitalLat, hospitalLon]]);
        hospitalMap.fitBounds(bounds, { padding: [20, 20] });
        
        console.log('Map updated successfully');
    } catch (error) {
        console.error('Error updating map:', error);
    }
}

// Add event listener for Enter key on location input
document.addEventListener('DOMContentLoaded', function() {
    const locationInput = document.getElementById('locationInput');
    if (locationInput) {
        locationInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                findNearestHospital();
            }
        });
    }
    
    // Add a test button for debugging
    const testButton = document.createElement('button');
    testButton.textContent = 'Test Hospital Search';
    testButton.onclick = testHospitalSearch;
    testButton.style.marginTop = '10px';
    testButton.style.padding = '5px 10px';
    testButton.style.backgroundColor = '#007bff';
    testButton.style.color = 'white';
    testButton.style.border = 'none';
    testButton.style.borderRadius = '4px';
    testButton.style.cursor = 'pointer';
    
    const hospitalInfo = document.getElementById('hospitalInfo');
    if (hospitalInfo) {
        hospitalInfo.appendChild(testButton);
    }
});

function testHospitalSearch() {
    console.log('🧪 Testing hospital search functionality...');
    
    // Test with a sample address
    const testAddress = "Vancouver General Hospital, Vancouver, BC";
    document.getElementById('locationInput').value = testAddress;
    
    console.log('Testing with address:', testAddress);
    
    // Add a visual indicator that test is running
    const hospitalInfo = document.getElementById('hospitalInfo');
    hospitalInfo.innerHTML = '<p style="color: #007bff;">🧪 Test Mode: Searching for hospitals near ' + testAddress + '...</p>';
    
    // Trigger the search
    findNearestHospital();
    
    // Also test the fallback functionality after a short delay
    setTimeout(() => {
        console.log('Testing fallback functionality...');
        const fallbackAddress = "Test Location, Vancouver, BC";
        document.getElementById('locationInput').value = fallbackAddress;
        hospitalInfo.innerHTML = '<p style="color: #007bff;">🧪 Test Mode: Testing fallback with ' + fallbackAddress + '...</p>';
        findNearestHospital();
    }, 2000);
}

// Add CSS animation for emergency pulsing
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
`;
document.head.appendChild(style);

// Heatstroke Prediction Functions
function setupHeatstrokePrediction() {
    const quickHeatstrokeBtn = document.getElementById('quickHeatstrokeBtn');
    if (quickHeatstrokeBtn) {
        quickHeatstrokeBtn.addEventListener('click', performQuickHeatstrokePrediction);
    }
    
    // Check if we have health assessment data and show the button
    checkHealthDataAndUpdateHeatstrokeUI();
    
    // Automatically calculate heatstroke risk on page load if we have data
    setTimeout(() => {
        autoCalculateHeatstrokeRisk();
    }, 2000); // Wait 2 seconds for other data to load
}

function checkHealthDataAndUpdateHeatstrokeUI() {
    const healthData = JSON.parse(localStorage.getItem('healthAssessmentData') || '{}');
    const quickHeatstrokeBtn = document.getElementById('quickHeatstrokeBtn');
    
    if (healthData.age && healthData.gender && quickHeatstrokeBtn) {
        quickHeatstrokeBtn.style.display = 'inline-block';
        quickHeatstrokeBtn.textContent = 'Calculate Heatstroke Risk';
    }
}

async function performQuickHeatstrokePrediction() {
    const button = document.getElementById('quickHeatstrokeBtn');
    const riskElement = document.getElementById('heatstrokeRisk');
    const indicatorElement = document.getElementById('heatstrokeIndicator');
    
    // Disable button and show loading
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';
    riskElement.textContent = 'Calculating...';
    
    try {
        // Get current health assessment data
        const healthData = JSON.parse(localStorage.getItem('healthAssessmentData') || '{}');
        
        if (!healthData.age || !healthData.gender) {
            throw new Error('Please complete the health assessment first');
        }
        
        // Get current BPM data
        const bpmData = await getCurrentBPMData();
        
        // Prepare data for prediction
        const predictionData = {
            health_data: {
                age: parseInt(healthData.age),
                gender: healthData.gender,
                symptoms: healthData.symptoms || [],
                medical_history: healthData.medical_history || [],
                risk_factors: healthData.risk_factors || []
            },
            bpm_data: bpmData
        };
        
        // Send prediction request
        const response = await fetch('/api/heatstroke_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Update UI with prediction results
        updateHeatstrokeRiskDisplay(result);
        
    } catch (error) {
        console.error('Heatstroke prediction error:', error);
        riskElement.textContent = 'Error';
        indicatorElement.className = 'risk-indicator high-risk';
        showNotification('Failed to calculate heatstroke risk: ' + error.message, 'error');
    } finally {
        // Re-enable button
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-calculator"></i> Calculate';
    }
}

async function getCurrentBPMData() {
    try {
        // Try to get BPM data from the dashboard API
        const response = await fetch('/api/csv_analysis');
        if (response.ok) {
            const data = await response.json();
            return {
                bpm: data.bpm || 75,
                skin_temperature: data.skin_temperature || 36.5
            };
        }
    } catch (error) {
        console.warn('Could not fetch BPM data:', error);
    }
    
    // Fallback to default values
    return {
        bpm: 75,
        skin_temperature: 36.5
    };
}

async function autoCalculateHeatstrokeRisk() {
    try {
        console.log('🔄 Auto-calculating heatstroke risk...');
        
        // First try to get the latest prediction from stored data
        const latestResponse = await fetch('/api/latest_heatstroke_prediction');
        if (latestResponse.ok) {
            const latestData = await latestResponse.json();
            if (latestData.success) {
                updateHeatstrokeRiskDisplay(latestData.prediction);
                console.log('✅ Auto heatstroke risk loaded from stored data');
                return;
            }
        }
        
        // Fallback: Check if we have health assessment data in localStorage
        const healthData = JSON.parse(localStorage.getItem('healthAssessmentData') || '{}');
        
        if (!healthData.age || !healthData.gender) {
            console.log('No health assessment data available for auto heatstroke calculation');
            return;
        }
        
        // Get current BPM data
        const bpmData = await getCurrentBPMData();
        
        // Prepare data for prediction
        const predictionData = {
            health_data: {
                age: parseInt(healthData.age),
                gender: healthData.gender,
                symptoms: healthData.symptoms || [],
                medical_history: healthData.medical_history || [],
                risk_factors: healthData.risk_factors || []
            },
            bpm_data: bpmData
        };
        
        // Send prediction request
        const response = await fetch('/api/heatstroke_prediction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });
        
        if (response.ok) {
            const result = await response.json();
            updateHeatstrokeRiskDisplay(result);
            console.log('✅ Auto heatstroke risk calculation completed');
        } else {
            console.warn('Auto heatstroke calculation failed:', response.status);
        }
        
    } catch (error) {
        console.warn('Auto heatstroke calculation error:', error);
    }
}

function updateHeatstrokeRiskDisplay(result) {
    const riskElement = document.getElementById('heatstrokeRisk');
    const indicatorElement = document.getElementById('heatstrokeIndicator');
    const heatstrokeCard = document.querySelector('.heatstroke-card');
    
    const isHighRisk = result.heatstroke_prediction;
    const probability = result.heatstroke_probability || result.probability || 0;
    const riskLevel = result.risk_level || (isHighRisk ? 'HIGH' : (probability > 0.3 ? 'MODERATE' : 'LOW'));
    
    // Update risk value with percentage
    if (isHighRisk || probability > 0.7) {
        riskElement.textContent = `${riskLevel}\n${(probability * 100).toFixed(1)}%`;
        riskElement.style.color = '#d32f2f';
        indicatorElement.className = 'risk-indicator high-risk';
        if (heatstrokeCard) {
            heatstrokeCard.classList.add('high-risk');
        }
    } else if (probability > 0.3) {
        riskElement.textContent = `${riskLevel}\n${(probability * 100).toFixed(1)}%`;
        riskElement.style.color = '#f57c00';
        indicatorElement.className = 'risk-indicator moderate-risk';
        if (heatstrokeCard) {
            heatstrokeCard.classList.remove('high-risk');
        }
    } else {
        riskElement.textContent = `${riskLevel}\n${(probability * 100).toFixed(1)}%`;
        riskElement.style.color = '#388e3c';
        indicatorElement.className = 'risk-indicator low-risk';
        if (heatstrokeCard) {
            heatstrokeCard.classList.remove('high-risk');
        }
    }
    
    // Show notification with more details
    const notificationText = `Heatstroke Risk: ${riskLevel} (${(probability * 100).toFixed(1)}%)`;
    showNotification(notificationText, isHighRisk ? 'warning' : 'info');
    
    // Log detailed information to console
    console.log('🔥 Heatstroke Risk Assessment:', {
        prediction: isHighRisk,
        probability: probability,
        risk_level: riskLevel,
        features_used: result.features_used || [],
        feature_values: result.feature_values || {}
    });
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}