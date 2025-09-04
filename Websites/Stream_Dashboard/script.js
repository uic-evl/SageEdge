// Configuration
const SERVER_URL = '';  // Use relative URLs when served from the same server
const VIDEO_URL = '/video_feed';

// Global state
let isPlaying = true;
let showTimestamp = true;
let currentLoopMode = 'full';
let currentCamera = 'top';

// Cookie consent functionality
            // Refresh the video stream to avoid showing cached last frame
            const video = document.getElementById('video');
            setTimeout(() => {
                video.src = `${VIDEO_URL}?t=${Date.now()}`;
            }, 300);
function showCookieConsent() {
    const consent = localStorage.getItem('cookieConsent');
    if (!consent) {
        document.getElementById('cookieConsent').classList.add('show');
    }
}

function acceptCookies() {
    localStorage.setItem('cookieConsent', 'accepted');
    document.getElementById('cookieConsent').classList.remove('show');
    showNotification('Cookie preferences saved', 'success');
}

function declineCookies() {
    localStorage.setItem('cookieConsent', 'declined');
    document.getElementById('cookieConsent').classList.remove('show');
    showNotification('Some features may not work properly without cookies', 'warning');
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('NIU Time-lapse Controller initialized');
    
    // Show cookie consent
    showCookieConsent();
    
    // Set default date to start date and guard excluded dates
    const dateInput = document.getElementById('dateInput');
    // Fetch status to get dynamic constraints
    makeAPICall('/status').then((data) => {
        const DEFAULT_START = (data && data.data_range && data.data_range.start_date) ? data.data_range.start_date.slice(0,10) : '2021-07-23';
        const END_DATE = (data && data.data_range && data.data_range.end_date) ? data.data_range.end_date.slice(0,10) : '2024-09-04';
    // excluded_dates removed from backend
        dateInput.min = DEFAULT_START;
        dateInput.max = END_DATE;
        dateInput.value = DEFAULT_START;
        dateInput.addEventListener('change', () => {
            const v = dateInput.value;
            if (!v) return;
            if (v < DEFAULT_START) {
                dateInput.value = DEFAULT_START;
            }
            if (v > END_DATE) {
                dateInput.value = END_DATE;
            }
        });
    }).catch(() => {
        // Fallback defaults if status fails (updated range)
        dateInput.value = '2021-07-23';
        dateInput.min = '2021-07-23';
        dateInput.max = '2024-09-04';
    });
    
    // Initialize video
    initializeVideo();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initial status update
    updateStatus();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Show new feature notifications
    setTimeout(() => {
        showNotification('NEW: You can now select specific minutes for precise navigation!', 'info');
    }, 2000);
    
    setTimeout(() => {
        showNotification('NEW: Switch between Top and Bottom camera views!', 'success');
    }, 4000);
});

// Initialize video element
function initializeVideo() {
    const video = document.getElementById('video');
    const overlay = document.getElementById('videoOverlay');
    
    video.onload = function() {
        console.log('Video stream loaded successfully');
        overlay.style.display = 'none';
        video.style.opacity = '1';
        showNotification('Video stream connected!', 'success');
    };
    
    video.onerror = function() {
        console.error('❌ Video stream error');
        overlay.style.display = 'block';
        overlay.innerHTML = '<div class="loading-spinner"></div><p>Video stream error - Retrying...</p>';
        
        // Retry after 3 seconds
        setTimeout(() => {
            video.src = `${VIDEO_URL}?t=${Date.now()}`;
        }, 3000);
    };
}

// Setup event listeners
function setupEventListeners() {
    // Camera control
    document.getElementById('cameraSelect').addEventListener('change', function() {
        const camera = this.value;
        setCamera(camera);
    });
    
    // Speed control
    document.getElementById('speedSelect').addEventListener('change', function() {
        const speed = parseFloat(this.value);
        setPlaybackSpeed(speed);
    });
    
    // Loop mode control
    document.getElementById('loopSelect').addEventListener('change', function() {
        const loopMode = this.value;
        setLoopMode(loopMode);
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        switch(e.key) {
            case ' ':
                e.preventDefault();
                togglePlayback();
                break;
            case 'r':
                updateStatus();
                break;
            case 't':
                toggleTimestamp();
                break;
        }
    });
}

// API Functions
async function makeAPICall(endpoint, data = null) {
    try {
        const options = {
            method: data ? 'POST' : 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${SERVER_URL}${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API call failed for ${endpoint}:`, error);
        showNotification(`Network error: ${error.message}`, 'error');
        throw error;
    }
}

// Control Functions
async function jumpToDateTime() {
    const date = document.getElementById('dateInput').value;
    const hour = document.getElementById('hourSelect').value;
    const minute = document.getElementById('minuteSelect').value;
    
    if (!date) {
        showNotification('Please select a date first!', 'warning');
        return;
    }
    
    const datetime = `${date} ${hour.padStart(2, '0')}:${minute.padStart(2, '0')}:00`;
    console.log(`Jumping to: ${datetime}`);
    
    try {
        const response = await makeAPICall('/control', {
            action: 'jump_to_datetime',
            datetime: datetime
        });
        
        if (response.status === 'success') {
            const requested = response.requested_time || `${date} ${hour.padStart(2,'0')}:${minute.padStart(2,'0')}:00`;
            const actual = response.actual_time || requested;
            const same = actual === requested;
            const actualLocal = formatDateTime(actual);
            if (same) {
                showNotification(`Jumped to ${actualLocal}`, 'success');
            } else {
                showNotification(`Requested ${requested}. Jumped to nearest available: ${actualLocal}`, 'info');
            }

            // Sync controls to the actual time we jumped to
            try {
                const d = new Date(actual);
                const yyyy = d.getFullYear();
                const mm = String(d.getMonth() + 1).padStart(2, '0');
                const dd = String(d.getDate()).padStart(2, '0');
                const hh = String(d.getHours()); // hourSelect expects non-padded value strings
                const min = String(d.getMinutes());
                document.getElementById('dateInput').value = `${yyyy}-${mm}-${dd}`;
                document.getElementById('hourSelect').value = hh;
                // snap minutes to our 5-min options if needed
                const allowed = [0,5,10,15,20,25,30,35,40,45,50,55];
                const nearest = allowed.reduce((a,b)=> Math.abs(b - d.getMinutes()) < Math.abs(a - d.getMinutes()) ? b : a, 0);
                document.getElementById('minuteSelect').value = String(nearest);
            } catch {}

            updateStatus();
        } else {
            showNotification(`Error: ${response.message || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showNotification('Failed to jump to date/time', 'error');
    }
}

async function togglePlayback() {
    console.log('Toggling playback...');
    
    try {
        const response = await makeAPICall('/control', {
            action: 'toggle_playback'
        });
        
        if (response.status === 'success') {
            isPlaying = response.is_playing;
            updatePlaybackButton();
            showNotification(isPlaying ? 'Playback resumed' : 'Playback paused', 'info');
            updateStatus();
        }
    } catch (error) {
        showNotification('Failed to toggle playback', 'error');
    }
}

async function toggleTimestamp() {
    console.log('Toggling timestamp...');
    
    try {
        const response = await makeAPICall('/control', {
            action: 'toggle_timestamp'
        });
        
        if (response.status === 'success') {
            showTimestamp = response.show_timestamp;
            updateTimestampButton();
            showNotification(showTimestamp ? 'Timestamp enabled' : 'Timestamp disabled', 'info');
            updateStatus();
        }
    } catch (error) {
        showNotification('Failed to toggle timestamp', 'error');
    }
}

async function setCamera(camera) {
    console.log(`📹 Switching to ${camera} camera...`);
    
    try {
        const response = await makeAPICall('/control', {
            action: 'set_camera',
            camera: camera
        });
        
        if (response.status === 'success') {
            currentCamera = camera;
            showNotification(`Switched to ${camera} camera view`, 'success');
            updateStatus();
            
            // Add visual feedback
            const cameraControl = document.querySelector('.camera-control');
            cameraControl.classList.add('success-flash');
            setTimeout(() => cameraControl.classList.remove('success-flash'), 500);
            
            // Refresh video stream
            const video = document.getElementById('video');
            setTimeout(() => {
                video.src = `${VIDEO_URL}?t=${Date.now()}`;
            }, 1000);
        } else {
            showNotification(`Error: ${response.message || 'Failed to switch camera'}`, 'error');
        }
    } catch (error) {
        showNotification('Failed to switch camera', 'error');
    }
}

async function setPlaybackSpeed(fps) {
    console.log(`🎬 Setting frame rate to: ${fps} frames per second`);
    
    try {
        const response = await makeAPICall('/control', {
            action: 'set_speed',
            frames_per_second: parseFloat(fps)
        });
        
        if (response.status === 'success') {
            showNotification(`Frame rate set to ${fps} frames per second`, 'success');
            updateStatus();
        }
    } catch (error) {
        showNotification('Failed to set frame rate', 'error');
    }
}

async function setLoopMode(loopMode) {
    console.log(`Setting loop mode to: ${loopMode}`);
    
    try {
        const response = await makeAPICall('/control', {
            action: 'set_loop_mode',
            loop_mode: loopMode
        });
        
        if (response.status === 'success') {
            currentLoopMode = loopMode;
            showNotification(`Loop mode changed to: ${getLoopModeDescription(loopMode)}`, 'success');
            updateStatus();
            
            // Add visual feedback
            const loopControl = document.querySelector('.loop-control');
            loopControl.classList.add('success-flash');
            setTimeout(() => loopControl.classList.remove('success-flash'), 500);
        }
    } catch (error) {
        showNotification('Failed to set loop mode', 'error');
    }
}

// Status Functions
async function updateStatus() {
    console.log('Updating status...');
    
    try {
        const data = await makeAPICall('/status');
        
        // Update status display
        document.getElementById('statusValue').textContent = data.status || 'Unknown';
        document.getElementById('cameraValue').textContent = data.current_camera ? `📹 ${data.current_camera.toUpperCase()} Camera` : '-';
        document.getElementById('playingValue').textContent = data.is_playing ? '✅ Playing' : '❌ Paused';
        document.getElementById('currentTimeValue').textContent = formatDateTime(data.current_datetime) || '-';
        document.getElementById('speedValue').textContent = `${data.frames_per_second || 4} fps`;
        document.getElementById('loopModeValue').textContent = getLoopModeDescription(data.loop_mode) || 'Unknown';
        document.getElementById('bufferValue').textContent = `${data.buffer_size || 0} frames`;
        
        // Update global state
        isPlaying = data.is_playing;
        showTimestamp = data.show_timestamp;
        currentLoopMode = data.loop_mode;
        currentCamera = data.current_camera || 'top';
        
        // Update button states
        updatePlaybackButton();
        updateTimestampButton();
        
        // Update form values
        document.getElementById('loopSelect').value = data.loop_mode || 'full';
        document.getElementById('cameraSelect').value = data.current_camera || 'top';
        
    } catch (error) {
        console.error('Status update failed:', error);
        document.getElementById('statusValue').textContent = 'Error';
    }
}

// UI Update Functions
function updatePlaybackButton() {
    const btn = document.getElementById('playPauseBtn');
    btn.textContent = isPlaying ? '⏸️ Pause' : '▶️ Play';
}

function updateTimestampButton() {
    const btn = document.getElementById('timestampBtn');
    btn.textContent = showTimestamp ? '🏷️ Hide Timestamp' : '🏷️ Show Timestamp';
}

// Utility Functions
function formatDateTime(isoString) {
    if (!isoString) return '';
    
    try {
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    } catch (error) {
        return isoString;
    }
}

function getLoopModeDescription(mode) {
    const descriptions = {
        'full': '🌍 Full Dataset',
        'day': '📅 Current Day',
        'hour': '🕐 Current Hour',
        'none': '⏹️ No Loop'
    };
    return descriptions[mode] || mode;
}

// Notification System
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        max-width: 400px;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function getNotificationColor(type) {
    const colors = {
        'success': '#28a745',
        'error': '#dc3545',
        'warning': '#ffc107',
        'info': '#17a2b8'
    };
    return colors[type] || colors.info;
}

// Auto-refresh
function startAutoRefresh() {
    // Update status every 5 seconds
    setInterval(updateStatus, 5000);
    
    // Check video stream every 10 seconds
    setInterval(checkVideoStream, 10000);
}

function checkVideoStream() {
    const video = document.getElementById('video');
    if (!video.complete || video.naturalWidth === 0) {
        console.warn('Video stream may be interrupted, refreshing...');
        video.src = `${VIDEO_URL}?t=${Date.now()}`;
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification button {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        padding: 0;
        margin: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background 0.2s;
    }
    
    .notification button:hover {
        background: rgba(255, 255, 255, 0.2);
    }
`;
document.head.appendChild(style);

// Export functions for global access
window.jumpToDateTime = jumpToDateTime;
window.togglePlayback = togglePlayback;
window.toggleTimestamp = toggleTimestamp;
window.updateStatus = updateStatus;

// AI Analysis Functions
async function loadAnalysisDashboard() {
    try {
        showNotification('Loading AI analysis dashboard...', 'info');
        
        const response = await fetch('/api/analysis/dashboard');
        const data = await response.json();
        
        if (data.error) {
            showNotification(`Error: ${data.error}`, 'error');
            return;
        }
        
        // Update stats
        document.getElementById('totalAnalyzed').textContent = data.total_analyzed.toLocaleString();
        
        // Update weather distribution
        const topWeather = data.weather_stats.slice(0, 3).map(w => `${w.category} (${w.count})`).join(', ');
        document.getElementById('weatherDistribution').textContent = topWeather || 'No data';
        
        // Update people distribution
        const topPeople = data.people_stats.slice(0, 3).map(p => `${p.category} (${p.count})`).join(', ');
        document.getElementById('peopleDistribution').textContent = topPeople || 'No data';
        
        // Update time distribution
        const topTime = data.time_stats.slice(0, 3).map(t => `${t.category} (${t.count})`).join(', ');
        document.getElementById('timeDistribution').textContent = topTime || 'No data';
        
        showNotification('AI dashboard loaded successfully!', 'success');
        
    } catch (error) {
        console.error('Error loading analysis dashboard:', error);
        showNotification('Failed to load AI dashboard', 'error');
    }
}

async function searchPhotos() {
    try {
        const weather = document.getElementById('weatherFilter').value;
        const people = document.getElementById('peopleFilter').value;
        const visibility = document.getElementById('visibilityFilter').value;
        const time = document.getElementById('timeFilter').value;
        
        const params = new URLSearchParams();
        if (weather) params.append('weather', weather);
        if (people) params.append('people', people);
        if (visibility) params.append('visibility', visibility);
        if (time) params.append('time', time);
        params.append('limit', '20');
        
        showNotification('Searching photos...', 'info');
        
        const response = await fetch(`/api/analysis/search?${params}`);
        const data = await response.json();
        
        if (data.error) {
            showNotification(`Search error: ${data.error}`, 'error');
            return;
        }
        
        displaySearchResults(data);
        showNotification(`Found ${data.count} photos matching your criteria`, 'success');
        
    } catch (error) {
        console.error('Error searching photos:', error);
        showNotification('Search failed', 'error');
    }
}

function displaySearchResults(data) {
    const resultsContainer = document.getElementById('searchResults');
    const resultsGrid = document.getElementById('resultsGrid');
    
    if (data.count === 0) {
        resultsGrid.innerHTML = '<p>No photos found matching your criteria.</p>';
        resultsContainer.style.display = 'block';
        return;
    }
    
    resultsGrid.innerHTML = data.photos.map(photo => {
        const weatherConf = getConfidenceBadge(photo.weather_confidence);
        const peopleConf = getConfidenceBadge(photo.people_confidence);
        
        return `
            <div class="result-card">
                <h4>${photo.filename}</h4>
                <div class="result-meta">
                    <div class="meta-item">
                        <span>📅 Date:</span>
                        <span>${new Date(photo.datetime).toLocaleDateString()}</span>
                    </div>
                    <div class="meta-item">
                        <span>🌤️ Weather:</span>
                        <span>${photo.weather} ${weatherConf}</span>
                    </div>
                    <div class="meta-item">
                        <span>👥 People:</span>
                        <span>${photo.people} ${peopleConf}</span>
                    </div>
                    <div class="meta-item">
                        <span>👁️ Visibility:</span>
                        <span>${photo.visibility}</span>
                    </div>
                    <div class="meta-item">
                        <span>🕐 Time:</span>
                        <span>${photo.time}</span>
                    </div>
                    <div class="meta-item">
                        <span>🎨 Color:</span>
                        <span style="background: ${photo.dominant_color}; padding: 2px 8px; border-radius: 4px; color: white; font-size: 0.8rem;">●</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    resultsContainer.style.display = 'block';
}

function getConfidenceBadge(confidence) {
    if (confidence >= 0.8) {
        return '<span class="confidence-badge confidence-high">High</span>';
    } else if (confidence >= 0.6) {
        return '<span class="confidence-badge confidence-medium">Med</span>';
    } else {
        return '<span class="confidence-badge confidence-low">Low</span>';
    }
}

function clearFilters() {
    document.getElementById('weatherFilter').value = '';
    document.getElementById('peopleFilter').value = '';
    document.getElementById('visibilityFilter').value = '';
    document.getElementById('timeFilter').value = '';
    
    // Hide search results
    document.getElementById('searchResults').style.display = 'none';
    
    showNotification('Filters cleared', 'info');
}

// Auto-load AI dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load AI dashboard after a short delay
    setTimeout(() => {
        loadAnalysisDashboard();
    }, 3000);
});

// Export new functions
window.loadAnalysisDashboard = loadAnalysisDashboard;
window.searchPhotos = searchPhotos;
window.clearFilters = clearFilters;
