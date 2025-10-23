// DOM Elements
const processBtn = document.getElementById('processBtn');
const clearBtn = document.getElementById('clearBtn');
const refreshBtn = document.getElementById('refreshBtn');
const urlsInput = document.getElementById('urlsInput');
const clusterNameInput = document.getElementById('clusterName');
const brandKeyInput = document.getElementById('brandKey');
const vibemyadSection = document.getElementById('vibemyadSection');
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
            if (dataset === 'vibemyad') {
                vibemyadSection.style.display = 'none';
                brandKeyInput.value = '';
            }
        } else {
            datasetButtons.forEach(b => b.classList.remove('active'));
            selectedDataset = dataset;
            btn.classList.add('active');
            clusterNameInput.focus();
            urlsInput.value = '';
            updateUrlCount();

            if (dataset === 'vibemyad') {
                vibemyadSection.style.display = 'block';
                brandKeyInput.focus();
            } else {
                vibemyadSection.style.display = 'none';
                brandKeyInput.value = '';
            }
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
        vibemyadSection.style.display = 'none';
        brandKeyInput.value = '';
    }
});

// Clear button handler
clearBtn.addEventListener('click', () => {
    urlsInput.value = '';
    clusterNameInput.value = '';
    brandKeyInput.value = '';
    selectedDataset = null;
    datasetButtons.forEach(b => b.classList.remove('active'));
    vibemyadSection.style.display = 'none';
    updateUrlCount();
    statusDiv.classList.remove('show');
});

// Show status message
function showStatus(message, type) {
    statusDiv.className = `show ${type}`;
    let icon = type === 'loading' ? '⏳' : type === 'success' ? '✅' : '❌';
    statusDiv.innerHTML = `<span class="status-icon">${icon}</span><span>${message}</span>`;
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

// Fetch URLs from predefined dataset
async function fetchDatasetUrls(datasetKey) {
    try {
        const response = await fetch(`/fetch-images/${datasetKey}`);

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error(`No images found for dataset: ${datasetKey}`);
            }
            throw new Error(`Failed to fetch dataset: ${response.status} ${response.statusText}`);
        }

        const urls = await response.json();

        if (!urls || urls.length === 0) {
            throw new Error(`No images found for dataset: ${datasetKey}`);
        }

        return urls;
    } catch (err) {
        console.error(`Error fetching dataset ${datasetKey}:`, err);
        throw new Error(`Failed to fetch ${datasetKey} dataset: ${err.message}`);
    }
}

// Fetch URLs from custom brand key
async function fetchVibeMyAdUrls(brandKey) {
    try {
        const response = await fetch(`/fetch-images/${encodeURIComponent(brandKey)}`);

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error(`No images found for brand key: ${brandKey}`);
            }
            throw new Error(`Failed to fetch images: ${response.status} ${response.statusText}`);
        }

        const urls = await response.json();

        if (!urls || urls.length === 0) {
            throw new Error(`No images found for brand key: ${brandKey}`);
        }

        return urls;
    } catch (err) {
        console.error('Error fetching images:', err);
        throw new Error(`Failed to fetch images for "${brandKey}": ${err.message}`);
    }
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
                    <strong>No clusters found</strong>
                    <p>Process some images to create your first cluster!</p>
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
                        <span class="detail-label">Avg Size:</span> ${cluster.cluster_sizes.mean.toFixed(2)}
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Max Size:</span> ${cluster.cluster_sizes.max}
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Similarity:</span> ${(cluster.similarities.mean * 100).toFixed(2)}%
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
                <strong>Error loading clusters</strong>
                <p style="color: #c62828; margin-top: 0.5rem;">${escapeHtml(err.message)}</p>
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
            <div class="spinner"></div>
            <p>Loading clusters for job ${escapeHtml(jobId)}...</p>
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
                    <strong>No clusters found</strong>
                    <p>This job has no multi-image clusters.</p>
                    <button class="back-btn" onclick="loadClusters()">⬅️ Back to Clusters</button>
                </div>
            `;
            return;
        }

        const clustersHTML = clusters.map(cluster => `
            <div class="cluster-detail-card">
                <div class="cluster-detail-header">
                    <div>
                        <div class="cluster-path">Path: ${escapeHtml(cluster.hierarchy_path)}</div>
                        <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">🕒 ${formatTimestamp(cluster.created_at)}</div>
                    </div>
                    <span class="similarity-badge">${(cluster.similarity * 100).toFixed(2)}% Similar</span>
                </div>
                <div class="cluster-images">
                    ${cluster.images_urls.map(url => `
                        <img src="${escapeHtml(url)}" alt="ad image" class="cluster-image" onerror="this.style.display='none'" />
                    `).join('')}
                </div>
                <div class="cluster-meta">
                    <span><strong>${cluster.size}</strong> images</span>
                    <span>Depth: <strong>${cluster.depth}</strong></span>
                </div>
            </div>
        `).join('');

        clusterCard.innerHTML = `
            <div class="cluster-header">
                <div><div class="cluster-name">🧩 Clusters for Job ${escapeHtml(jobId)}</div></div>
                <button class="back-btn" onclick="loadClusters()">⬅️ Back</button>
            </div>
            <div style="max-height: 600px; overflow-y: auto; padding-right: 0.5rem;">${clustersHTML}</div>
        `;

    } catch (err) {
        console.error('Error loading cluster details:', err);
        clusterCard.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <strong>Error loading cluster details</strong>
                <p style="color: #c62828; margin-top: 0.5rem;">${escapeHtml(err.message)}</p>
                <button class="back-btn" onclick="loadClusters()">⬅️ Back to Clusters</button>
            </div>
        `;
    }
}

// Refresh button handler
refreshBtn.addEventListener('click', () => {
    clustersContainer.innerHTML = `
        <div class="loading-message">
            <div class="spinner"></div>
            <p>Refreshing clusters...</p>
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

    try {
        if (selectedDataset === 'vibemyad') {
            const brandKey = brandKeyInput.value.trim();

            if (!brandKey) {
                showStatus('Please enter a brand key for VibeMyAd API.', 'error');
                brandKeyInput.focus();
                return;
            }

            processBtn.disabled = true;
            btnText.innerHTML = '<div class="spinner"></div> Fetching images...';
            showStatus(`Fetching images for brand: ${brandKey}...`, 'loading');

            urls = await fetchVibeMyAdUrls(brandKey);
            showStatus(`Found ${urls.length} images for "${brandKey}". Processing...`, 'loading');

        } else if (selectedDataset) {
            processBtn.disabled = true;
            btnText.innerHTML = '<div class="spinner"></div> Fetching images...';
            showStatus(`Fetching ${selectedDataset} dataset...`, 'loading');

            urls = await fetchDatasetUrls(selectedDataset);
            showStatus(`Found ${urls.length} images. Processing...`, 'loading');

        } else {
            urls = urlsInput.value.split('\n').map(u => u.trim()).filter(u => u);

            if (urls.length === 0) {
                showStatus('Please select a dataset or enter at least one URL to process.', 'error');
                return;
            }

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

        processBtn.disabled = true;
        btnText.innerHTML = '<div class="spinner"></div> Processing...';
        showStatus(`Processing ${urls.length} image${urls.length > 1 ? 's' : ''}... This may take some time.`, 'loading');

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
        showStatus(`Successfully processed images for Job ID: ${jobId}`, 'success');

        // Clear form
        urlsInput.value = '';
        clusterNameInput.value = '';
        brandKeyInput.value = '';
        selectedDataset = null;
        datasetButtons.forEach(b => b.classList.remove('active'));
        vibemyadSection.style.display = 'none';
        updateUrlCount();

        // Reload clusters after a short delay
        setTimeout(() => loadClusters(), 2000);

    } catch (err) {
        console.error('Processing error:', err);
        showStatus(`Error: ${err.message}`, 'error');
    } finally {
        processBtn.disabled = false;
        btnText.textContent = 'Process Images';
    }
});

// Initialize on page load
updateUrlCount();
loadClusters();

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter' && !processBtn.disabled) {
        processBtn.click();
    }
});