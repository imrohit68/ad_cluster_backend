// Dataset URLs configuration
const DATASET_URLS = {
    nike: [],
    go_noise: [],
    levis: []
};

// DOM Elements
const processBtn = document.getElementById('processBtn');
const clearBtn = document.getElementById('clearBtn');
const refreshBtn = document.getElementById('refreshBtn');
const urlsInput = document.getElementById('urlsInput');
const clusterNameInput = document.getElementById('clusterName');
const statusDiv = document.getElementById('status');
const urlCount = document.getElementById('urlCount');
const btnText = document.getElementById('btnText');
const clustersContainer = document.getElementById('clustersContainer');
const datasetButtons = document.querySelectorAll('.dataset-btn');

let selectedDataset = null;

// Dataset selection handlers
datasetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const dataset = btn.dataset.dataset;
        if (selectedDataset === dataset) {
            selectedDataset = null;
            btn.classList.remove('active');
        } else {
            datasetButtons.forEach(b => b.classList.remove('active'));
            selectedDataset = dataset;
            btn.classList.add('active');
            clusterNameInput.focus();
            urlsInput.value = '';
            updateUrlCount();
        }
    });
});

// Update URL count
function updateUrlCount() {
    const urls = urlsInput.value.split('\n').map(u => u.trim()).filter(u => u);
    urlCount.textContent = urls.length;
}

urlsInput.addEventListener('input', () => {
    updateUrlCount();
    if (urlsInput.value.trim()) {
        selectedDataset = null;
        datasetButtons.forEach(b => b.classList.remove('active'));
    }
});

// Clear button handler
clearBtn.addEventListener('click', () => {
    urlsInput.value = '';
    clusterNameInput.value = '';
    selectedDataset = null;
    datasetButtons.forEach(b => b.classList.remove('active'));
    updateUrlCount();
    statusDiv.classList.remove('show');
});

// Show status message
function showStatus(message, type) {
    statusDiv.className = `show ${type}`;
    let icon = type === 'loading' ? '⏳' : type === 'success' ? '✅' : '❌';
    statusDiv.innerHTML = `<span class="status-icon">${icon}</span>${message}`;
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load clusters from API
async function loadClusters() {
    try {
        const response = await fetch('/jobs');
        if (!response.ok) {
            throw new Error(`Failed to load clusters: ${response.status} ${response.statusText}`);
        }

        const clusters = await response.json();

        if (clusters.length === 0) {
            clustersContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📂</div>
                    <p><strong>No clusters found</strong></p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Process some images to create your first cluster!</p>
                </div>
            `;
            return;
        }

        // Sort clusters by timestamp (most recent first)
        clusters.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        clustersContainer.innerHTML = clusters.map(cluster => `
            <div class="cluster-card" data-job-id="${cluster.id}">
                <div class="cluster-header">
                    <div>
                        <div class="cluster-name">${escapeHtml(cluster.name)}</div>
                        <div class="cluster-id">ID: ${escapeHtml(cluster.id)}</div>
                    </div>
                    <div class="cluster-timestamp">📅 ${formatTimestamp(cluster.timestamp)}</div>
                </div>
                <div class="cluster-stats">
                    <div class="stat-box">
                        <span class="stat-value">${cluster.total_images}</span>
                        <span class="stat-label">Images</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-value">${cluster.total_clusters}</span>
                        <span class="stat-label">Clusters</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-value">${cluster.embedding_dimension}</span>
                        <span class="stat-label">Dimensions</span>
                    </div>
                </div>
                <div class="cluster-details">
                    <div class="detail-item">
                        <span class="detail-label">Avg Cluster Size:</span> ${cluster.cluster_sizes.mean.toFixed(2)}
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Max Cluster Size:</span> ${cluster.cluster_sizes.max}
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Avg Similarity:</span> ${(cluster.similarities.mean * 100).toFixed(2)}%
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Avg Depth:</span> ${cluster.depths.mean.toFixed(2)}
                    </div>
                </div>
            </div>
        `).join('');

        // Add click handlers to cluster cards
        document.querySelectorAll('.cluster-card').forEach(card => {
            card.addEventListener('click', handleClusterClick);
        });

    } catch (err) {
        console.error('Error loading clusters:', err);
        clustersContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <p><strong>Error loading clusters</strong></p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #c62828;">${escapeHtml(err.message)}</p>
            </div>
        `;
    }
}

// Handle cluster card click
async function handleClusterClick(e) {
    const clusterCard = e.currentTarget;
    const jobId = clusterCard.dataset.jobId;

    clusterCard.innerHTML = `
        <div class="loading-message">
            <div class="spinner" style="border-color: #ddd; border-top-color: #667eea; margin: 0 auto;"></div>
            <p style="margin-top: 1rem;">Loading image clusters for job ${escapeHtml(jobId)}...</p>
        </div>
    `;

    try {
        const response = await fetch(`/clusters/${jobId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch clusters for job ${jobId}: ${response.status}`);
        }

        let clusters = await response.json();

        // Filter clusters with 2+ images and sort
        clusters = clusters.filter(c => c.size >= 2).sort((a, b) => {
            if (a.size === b.size) return b.similarity - a.similarity;
            return a.size - b.size;
        });

        if (clusters.length === 0) {
            clusterCard.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🕳️</div>
                    <p><strong>No clusters found</strong></p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">This job has no multi-image clusters.</p>
                </div>
            `;
            return;
        }

        const clustersHTML = clusters.map(cluster => `
            <div style="background:white; border:1px solid #ddd; box-shadow:0 2px 8px rgba(0,0,0,0.1); border-radius:15px; padding:1.5rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:1rem;">
                    <div>
                        <div style="font-size:1.1rem; font-weight:600; color:#333; margin-bottom:0.3rem;">Path: ${escapeHtml(cluster.hierarchy_path)}</div>
                        <div style="font-size:0.85rem; color:#666; background:rgba(0,0,0,0.05); padding:0.2rem 0.5rem; border-radius:5px; display:inline-block;">Similarity: ${(cluster.similarity * 100).toFixed(2)}%</div>
                    </div>
                    <div style="font-size:0.85rem; color:#666;">🕒 ${formatTimestamp(cluster.created_at)}</div>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:10px; justify-content:center;">
                    ${cluster.images_urls.map(url => `<img src="${escapeHtml(url)}" alt="ad image" style="width:140px; height:140px; object-fit:cover; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.2);" onerror="this.style.display='none'" />`).join('')}
                </div>
                <div style="margin-top:1rem; text-align:center; color:#555; font-size:0.9rem;">
                    <strong>${cluster.size}</strong> images · Depth: ${cluster.depth}
                </div>
            </div>
        `).join('');

        clusterCard.innerHTML = `
            <div class="cluster-header">
                <div><div class="cluster-name">🧩 Clusters for Job ${escapeHtml(jobId)}</div></div>
                <button style="padding:6px 10px; border:none; background:#eee; border-radius:8px; cursor:pointer; font-weight:600;" onclick="loadClusters()">⬅️ Back</button>
            </div>
            <div style="max-height:600px; overflow-y:auto; padding-right:0.5rem;">${clustersHTML}</div>
        `;

    } catch (err) {
        console.error('Error loading cluster details:', err);
        clusterCard.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <p><strong>Error loading cluster details</strong></p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #c62828;">${escapeHtml(err.message)}</p>
                <button style="margin-top:1rem; padding:8px 16px; border:none; background:#667eea; color:white; border-radius:8px; cursor:pointer; font-weight:600;" onclick="loadClusters()">⬅️ Back to Clusters</button>
            </div>
        `;
    }
}

// Refresh button handler
refreshBtn.addEventListener('click', () => {
    clustersContainer.innerHTML = `
        <div class="loading-message">
            <div class="spinner" style="border-color: #ddd; border-top-color: #667eea; margin: 0 auto;"></div>
            <p style="margin-top: 1rem;">Refreshing clusters...</p>
        </div>
    `;
    loadClusters();
});

// Process button handler
processBtn.addEventListener('click', async () => {
    const clusterJobName = clusterNameInput.value.trim();

    if (!clusterJobName) {
        showStatus('Please enter a name for the clustering job.', 'error');
        clusterNameInput.focus();
        return;
    }

    let urls = [];

    if (selectedDataset) {
        urls = DATASET_URLS[selectedDataset];
    } else {
        urls = urlsInput.value.split('\n').map(u => u.trim()).filter(u => u);

        if (urls.length === 0) {
            showStatus('Please select a dataset or enter at least one URL to process.', 'error');
            return;
        }

        // Validate URLs
        const invalidUrls = urls.filter(url => {
            try {
                new URL(url);
                return false;
            } catch {
                return true;
            }
        });

        if (invalidUrls.length > 0) {
            showStatus(`Found ${invalidUrls.length} invalid URL(s). Please check your input.`, 'error');
            return;
        }
    }

    // Disable button and show loading state
    processBtn.disabled = true;
    btnText.innerHTML = '<div class="spinner"></div> Processing...';
    showStatus(`Processing ${urls.length} image${urls.length > 1 ? 's' : ''}... This may take some time.`, 'loading');

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cluster_job_name: clusterJobName,
                urls: urls
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const jobId = data.job_id || data.id || 'N/A';
        showStatus(`✅ Successfully processed images for Job ID: ${jobId}`, 'success');

        // Clear form
        urlsInput.value = '';
        clusterNameInput.value = '';
        selectedDataset = null;
        datasetButtons.forEach(b => b.classList.remove('active'));
        updateUrlCount();

        // Reload clusters after a short delay
        setTimeout(() => loadClusters(), 2000);

    } catch (err) {
        console.error('Processing error:', err);
        showStatus(`❌ Error: ${err.message}`, 'error');
    } finally {
        processBtn.disabled = false;
        btnText.textContent = 'Process Images';
    }
});

// Initialize on page load
updateUrlCount();
loadClusters();

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter' && !processBtn.disabled) {
        processBtn.click();
    }
});