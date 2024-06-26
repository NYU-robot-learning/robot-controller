import numpy as np

import torch
import torch.nn.functional as F

import clip

from home_robot.utils.voxel import VoxelizedPointcloud

from typing import List, Optional, Tuple, Union
from torch import Tensor

from transformers import AutoProcessor, AutoModel

from sklearn.cluster import DBSCAN

def find_clusters(vertices: np.ndarray, similarity: np.ndarray):
    # Calculate the number of top values directly
    top_positions = vertices
    # top_values = probability_over_all_points[top_indices].flatten()

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    clusters = dbscan.fit(top_positions)
    labels = clusters.labels_

    # Initialize empty lists to store centroids and extends of each cluster
    centroids = []
    extends = []
    similarity_max_list = []
    points = []

    for cluster_id in set(labels):
        if cluster_id == -1:  # Ignore noise
            continue

        members = top_positions[labels == cluster_id]
        similarity_values = similarity[labels == cluster_id]
        simiarity_max = np.max(similarity_values)
        centroid = np.mean(members, axis=0)

        sx = np.max(members[:, 0]) - np.min(members[:, 0])
        sy = np.max(members[:, 1]) - np.min(members[:, 1])
        sz = np.max(members[:, 2]) - np.min(members[:, 2])

        # Append centroid and extends to the lists
        centroids.append(centroid)
        extends.append((sx, sy, sz))
        similarity_max_list.append(simiarity_max)
        points.append(members)

    return centroids, extends, similarity_max_list, points

class VoxelMapLocalizer():
    def __init__(self, model_config = 'ViT-B/16', device = 'cuda', siglip = True):
        self.device = device
        # self.clip_model, self.preprocessor = clip.load(model_config, device=device)
        self.siglip = siglip
        if not self.siglip:
            self.clip_model, self.preprocessor = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
            self.preprocessor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.clip_model.eval()
        self.voxel_pcd = VoxelizedPointcloud().to(self.device)

    def add(self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
    ):
        points = points.to(self.device)
        if features is not None:
            features = features.to(self.device)
        if rgb is not None:
            rgb = rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.voxel_pcd.add(points = points, 
                        features = features,
                        rgb = rgb,
                        weights = weights)

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries] 
        if self.siglip:
            inputs = self.preprocessor(text=queries, padding="max_length", return_tensors="pt")
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
        else:
            text = clip.tokenize(queries).to(self.device)
            all_clip_tokens = self.clip_model.encode_text(text)
        # text = clip.tokenize(queries).to(self.device)
        # all_clip_tokens = self.clip_model.encode_text(text)
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        points, features, weights, _ = self.voxel_pcd.get_pointcloud()
        features = F.normalize(features, p=2, dim=-1)
        point_alignments = clip_text_tokens.float() @ features.float().T
    
        print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A, B = None, k_A = 10, k_B = 50):
        print("A is ", A)
        print("B is ", B)
        if B is None or B == '':
            target = self.find_alignment_for_A([A])[0]
        else:
            points, _, _, _ = self.voxel_pcd.get_pointcloud()
            alignments = self.find_alignment_over_model([A, B]).cpu()
            A_points = points[alignments[0].topk(k = k_A, dim = -1).indices].reshape(-1, 3)
            B_points = points[alignments[1].topk(k = k_B, dim = -1).indices].reshape(-1, 3)
            distances = torch.norm(A_points.unsqueeze(1) - B_points.unsqueeze(0), dim=2)
            target = A_points[torch.argmin(torch.min(distances, dim = 1).values)]
        return target.detach().cpu()

    def find_alignment_for_A(self, A):
        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        return points[alignments.argmax(dim = -1)].detach().cpu()

    def find_clusters_for_A(self, A):
        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu().reshape(-1).detach().numpy()
        turning_point = np.percentile(alignments, 99)
        mask = alignments > turning_point
        alignments = alignments[mask]
        points = points[mask]
        centroids, extends, similarity_max_list, points = find_clusters(points.detach().cpu().numpy(), alignments)
        return centroids, extends, similarity_max_list, points