"""
Stateless recursive hierarchical clustering for ad image embeddings
"""
import faiss
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List
import logging

import numpy as np

from app.models.cluster_models import ClusteringJob, Cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2 normalize rows of a matrix."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def mean_pairwise_similarity(emb: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity for rows in emb."""
    m = emb.shape[0]
    if m <= 1:
        return 1.0
    sims = np.dot(emb, emb.T)
    iu = np.triu_indices(m, k=1)
    vals = sims[iu]
    vals = np.clip(vals, -1.0, 1.0)
    return float(np.mean(vals))


class RecursiveAdClusterer:
    """Stateless recursive hierarchical clustering for ad embeddings"""

    def __init__(self):
        """Initialize stateless clusterer - no data stored."""
        logger.info("Initialized stateless RecursiveAdClusterer")

    def cluster_recursive(
        self,
        embeddings_data: Dict,
        initial_threshold: float = 0.75,
        min_threshold: float = 0.95,
        threshold_step: float = 0.05,
        min_group_size: int = 2,
        max_depth: int = 10,
        k_neighbors: int = 30,
        min_split_improvement: float = 0.02
    ) -> Dict:
        """
        Perform recursive hierarchical clustering on provided embeddings.
        
        Args:
            embeddings_data: Dict containing 'embeddings' (np.ndarray or list) 
                           and 'image_paths' (list)
            initial_threshold: Starting similarity threshold
            min_threshold: Maximum similarity threshold
            threshold_step: Threshold increment per level
            min_group_size: Minimum cluster size
            max_depth: Maximum recursion depth
            k_neighbors: Number of neighbors for graph construction
            min_split_improvement: Minimum improvement to accept a split
            
        Returns:
            Dict with clustering results and metadata
        """
        # Extract and normalize embeddings
        embeddings = np.array(embeddings_data['embeddings'], dtype=np.float32)
        image_paths = list(embeddings_data['image_paths'])
        metadata = embeddings_data.get('metadata', {})

        n_images, embedding_dim = embeddings.shape
        embeddings = normalize_rows(embeddings)

        logger.info(f"Clustering {n_images} images, dim={embedding_dim}")

        # Initialize clustering
        root_indices = list(range(n_images))
        result_leaf_clusters = []
        stack = deque()
        stack.append((root_indices, 0, initial_threshold, "root"))

        while stack:
            cur_indices, level, threshold, node_id = stack.pop()
            cur_emb = embeddings[cur_indices]
            m = len(cur_indices)

            # Base case
            if m <= 1 or level >= max_depth or threshold > min_threshold:
                result_leaf_clusters.append({
                    'indices': cur_indices,
                    'level': level,
                    'mean_sim': mean_pairwise_similarity(cur_emb),
                    'node_id': node_id
                })
                continue

            mean_sim = mean_pairwise_similarity(cur_emb)
            if mean_sim >= min_threshold:
                result_leaf_clusters.append({
                    'indices': cur_indices,
                    'level': level,
                    'mean_sim': mean_sim,
                    'node_id': node_id
                })
                continue

            # FAISS index
            d = cur_emb.shape[1]
            local_index = faiss.IndexFlatIP(d)
            local_index.add(cur_emb)

            k = min(k_neighbors + 1, m)
            distances, indices = local_index.search(cur_emb, k)

            # Build adjacency
            adjacency = defaultdict(set)
            for i in range(m):
                for neighbor_idx, sim in zip(indices[i], distances[i]):
                    if neighbor_idx != i and sim >= threshold:
                        adjacency[i].add(neighbor_idx)
                        adjacency[neighbor_idx].add(i)

            # No edges
            if not adjacency:
                result_leaf_clusters.append({
                    'indices': cur_indices,
                    'level': level,
                    'mean_sim': mean_sim,
                    'node_id': node_id
                })
                continue

            # Connected components
            visited = set()
            components = []
            for start_node in range(m):
                if start_node in visited:
                    continue
                comp = []
                stack_comp = [start_node]
                while stack_comp:
                    node = stack_comp.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.append(node)
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            stack_comp.append(neighbor)
                components.append(sorted(comp))

            # Single component -> tighten threshold
            if len(components) == 1:
                next_threshold = min(mean_sim + threshold_step, min_threshold)
                if m < min_group_size or next_threshold <= threshold + 1e-6 or level + 1 >= max_depth:
                    result_leaf_clusters.append({
                        'indices': cur_indices,
                        'level': level,
                        'mean_sim': mean_sim,
                        'node_id': node_id
                    })
                else:
                    stack.append((cur_indices, level + 1, next_threshold, node_id))
                continue

            # Multiple components -> check improvement
            subcluster_sims = []
            for comp in components:
                if len(comp) > 1:
                    comp_indices = [cur_indices[i] for i in comp]
                    comp_emb = embeddings[comp_indices]
                    subcluster_sims.append(mean_pairwise_similarity(comp_emb))

            if subcluster_sims:
                improvement = np.mean(subcluster_sims) - mean_sim
                if improvement < min_split_improvement:
                    result_leaf_clusters.append({
                        'indices': cur_indices,
                        'level': level,
                        'mean_sim': mean_sim,
                        'node_id': node_id
                    })
                    continue

            # Valid split
            next_threshold = min(threshold + threshold_step, min_threshold)
            for i, comp in enumerate(components):
                comp_global_indices = [cur_indices[idx] for idx in comp]
                comp_emb = embeddings[comp_global_indices]
                comp_mean = mean_pairwise_similarity(comp_emb)

                if len(comp_global_indices) == 1 or comp_mean >= min_threshold or len(comp_global_indices) < min_group_size or level + 1 >= max_depth:
                    result_leaf_clusters.append({
                        'indices': comp_global_indices,
                        'level': level + 1,
                        'mean_sim': comp_mean,
                        'node_id': f"{node_id}.{i}"
                    })
                else:
                    stack.append((comp_global_indices, level + 1, next_threshold, f"{node_id}.{i}"))

        # Convert to flat clusters
        flat_clusters = []
        for cluster_id, lc in enumerate(result_leaf_clusters):
            flat_clusters.append({
                'cluster_id': cluster_id,
                'depth': lc['level'],
                'size': len(lc['indices']),
                'indices': lc['indices'],
                'images': [image_paths[i] for i in lc['indices']],
                'similarity': lc['mean_sim'],
                'path': lc['node_id']
            })

        return {
            'method': 'recursive_hierarchical',
            'n_images': n_images,
            'embedding_dim': embedding_dim,
            'flat_clusters': flat_clusters,
            'parameters': {
                'initial_threshold': initial_threshold,
                'min_threshold': min_threshold,
                'threshold_step': threshold_step,
                'min_group_size': min_group_size,
                'max_depth': max_depth,
                'k_neighbors': k_neighbors,
                'min_split_improvement': min_split_improvement
            }
        }

    def get_metadata_model(self, clustering_result: Dict,clustering_job_name: str) -> ClusteringJob:
        """
        Generate metadata model from clustering result.
        
        Args:
            clustering_result: Result dict from cluster_recursive()
            
        Returns:
            ClusteringJob model with statistics
        """
        flat_clusters = clustering_result['flat_clusters']
        sizes = [c['size'] for c in flat_clusters]
        depths = [c['depth'] for c in flat_clusters]
        sims = [c['similarity'] for c in flat_clusters]

        metadata = ClusteringJob(
            name=clustering_job_name,
            total_images=clustering_result['n_images'],
            total_clusters=len(flat_clusters),
            embedding_dimension=clustering_result['embedding_dim'],
            timestamp=datetime.now(),
            cluster_sizes={
                'min': int(min(sizes)),
                'max': int(max(sizes)),
                'mean': float(np.mean(sizes)),
                'median': float(np.median(sizes)),
                'std': float(np.std(sizes))
            },
            depths={
                'min': int(min(depths)),
                'max': int(max(depths)),
                'mean': float(np.mean(depths))
            },
            similarities={
                'min': float(min(sims)),
                'max': float(max(sims)),
                'mean': float(np.mean(sims)),
                'median': float(np.median(sims)),
                'std': float(np.std(sims))
            }
        )
        return metadata

    def get_cluster_models(self, clustering_result: Dict, parent_id: str) -> List[Cluster]:
        """
        Generate Cluster models from clustering result.
        
        Args:
            clustering_result: Result dict from cluster_recursive()
            parent_id: Parent job ID
            
        Returns:
            List of Cluster models
        """
        cluster_models = []
        for c in clustering_result['flat_clusters']:
            cluster_models.append(
                Cluster(
                    job_id=parent_id,
                    size=c['size'],
                    depth=c['depth'],
                    similarity=c['similarity'],
                    hierarchy_path=c['path'],
                    images_urls=c['images']
                )
            )
        return cluster_models