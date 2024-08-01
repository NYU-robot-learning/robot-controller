import numpy as np

import torch
import torch.nn.functional as F

import clip

from home_robot.utils.voxel import VoxelizedPointcloud

from typing import List, Optional, Tuple, Union
from torch import Tensor

from transformers import AutoProcessor, AutoModel

from sklearn.cluster import DBSCAN

# from ultralytics import YOLOWorld
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def find_clusters(vertices: np.ndarray, similarity: np.ndarray, obs = None):
    # Calculate the number of top values directly
    top_positions = vertices
    # top_values = probability_over_all_points[top_indices].flatten()

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.25, min_samples=3)
    clusters = dbscan.fit(top_positions)
    labels = clusters.labels_

    # Initialize empty lists to store centroids and extends of each cluster
    centroids = []
    extends = []
    similarity_max_list = []
    points = []
    obs_max_list = []

    for cluster_id in set(labels):
        if cluster_id == -1:  # Ignore noise
            continue

        members = top_positions[labels == cluster_id]
        centroid = np.mean(members, axis=0)

        similarity_values = similarity[labels == cluster_id]
        simiarity_max = np.max(similarity_values)

        if obs is not None:
            obs_values = obs[labels == cluster_id]
            obs_max = np.max(obs_values)

        sx = np.max(members[:, 0]) - np.min(members[:, 0])
        sy = np.max(members[:, 1]) - np.min(members[:, 1])
        sz = np.max(members[:, 2]) - np.min(members[:, 2])

        # Append centroid and extends to the lists
        centroids.append(centroid)
        extends.append((sx, sy, sz))
        similarity_max_list.append(simiarity_max)
        points.append(members)
        if obs is not None:
            obs_max_list.append(obs_max)

    if obs is not None:
        return centroids, extends, similarity_max_list, points, obs_max_list
    else:
        return centroids, extends, similarity_max_list, points

class VoxelMapLocalizer():
    def __init__(self, voxel_map_wrapper = None, model_config = 'ViT-B/16', device = 'cuda', siglip = True):
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        # self.clip_model, self.preprocessor = clip.load(model_config, device=device)
        self.siglip = siglip
        if not self.siglip:
            self.clip_model, self.preprocessor = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
            self.preprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        self.voxel_pcd = VoxelizedPointcloud().to(self.device)
        # self.exist_model = YOLOWorld("yolov8l-worldv2.pt")
        self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.exist_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        

    def add(self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        obs_count: Optional[Tensor] = None,
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
                        weights = weights,
                        obs_count = obs_count)

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries] 
        if self.siglip:
            inputs = self.preprocessor(text=queries, padding="max_length", return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.device)
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
        if points is None:
            return None
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
    
    def find_obs_id_for_A(self, A):
        obs_counts = self.voxel_pcd._obs_counts
        alignments = self.find_alignment_over_model(A).cpu()
        return obs_counts[alignments.argmax(dim = -1)].detach().cpu()

    def localize_A_v2(self, A, debug = True, return_debug = False):
        centroids, extends, similarity_max_list, points, obs_max_list, debug_text = self.find_clusters_for_A(A, return_obs_counts = True, debug = debug)
        if len(centroids) == 0:
            if not debug:
                return None
            else:
                return None, debug_text
        target_point = None
        obs = None
        similarity = None
        point = None
        for idx, (centroid, obs, similarity, point) in enumerate(sorted(zip(centroids, obs_max_list, similarity_max_list, points), key=lambda x: x[2], reverse=True)):
            if self.siglip:
                cosine_similarity_check = similarity > 0.14
            else:
                cosine_similarity_check = similarity > 0.3
            if cosine_similarity_check:
                target_point = centroid

                debug_text += '#### - Instance ' +  str(idx + 1) + ' has high cosine similarity (' + str(round(similarity, 3)) +  '). **😃** Directly navigate to it.\n'

                break
            else:
                debug_text += '#### - Instance ' +  str(idx + 1) + ' has low confidence(' + str(round(similarity, 3)) +  '). **😞** Double check past observations. \n'
                detection_model_check = self.check_existence(A, obs)
                if detection_model_check:
                    target_point = centroid

                    debug_text += '#### - Obejct is detected in observations where instance' + str(idx + 1) + ' comes from. **😃** Directly navigate to it.\n'

                    break
                
                debug_text += '#### - Also not find target object in in past observations. **😞** \n'

        if target_point is None:
            debug_text += '#### - All instances are not the target! Maybe target object has not been observed yet. **😭**\n'
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs, point

    def check_existence(self, text, obs_id):
        if obs_id <= 0:
            return False
        # rgb = np.load(self.log + '/rgb' + str(obs_id) + '.npy')
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)
        inputs = self.exist_processor(text=[[text]], images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)
        with torch.no_grad():
            outputs = self.exist_model(**inputs)
        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
        results = self.exist_processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)
        xyxys = results[0]['boxes']
        # rgb = rgb.astype(np.uint8)[:, :, [2, 1, 0]]
        # self.exist_model.set_classes([text])
        # results = self.exist_model.predict(rgb, conf=0.2, verbose=False)
        # xyxy_tensor = results[0].boxes.xyxy
        # # return len(xyxy_tensor) != 0
        # xyxys = xyxy_tensor.detach().cpu().numpy()
        # depth = np.load(self.log + '/depth' + str(obs_id) + '.npy')
        depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        for xyxy in xyxys:
            w, h = depth.shape
            tl_x, tl_y, br_x, br_y = xyxy
            tl_x, tl_y, br_x, br_y = int(max(0, tl_x.item())), int(max(0, tl_y.item())), int(min(h, br_x.item())), int(min(w, br_y.item()))
            min_depth = np.min(depth[tl_y: br_y, tl_x: br_x].flatten())
            if min_depth < 2.5 and not np.isnan(min_depth):
                return True
        return False

    def find_clusters_for_A(self, A, return_obs_counts = False, debug = False):

        debug_text = ''

        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu().reshape(-1).detach().numpy()
        # turning_point = max(np.percentile(alignments, 99), 0.08)
        if self.siglip:
            turning_point = min(0.14, alignments[np.argsort(alignments)[-20]])
        else:
            turning_point = min(0.3, alignments[np.argsort(alignments)[-20]])
        mask = alignments >= turning_point
        alignments = alignments[mask]
        points = points[mask]
        if len(points) == 0:

            debug_text += '### - No instance found! Maybe target object has not been observed yet. **😭**\n'

            output = [[], [], [], []]
            if return_obs_counts:
                output.append([])
            if debug:
                output.append(debug_text)

            return output
        else:
            if return_obs_counts:
                obs_ids = self.voxel_pcd._obs_counts.detach().cpu().numpy()[mask]
                centroids, extends, similarity_max_list, points, obs_max_list = find_clusters(points.detach().cpu().numpy(), alignments, obs = obs_ids)
                output = [centroids, extends, similarity_max_list, points, obs_max_list]
            else:
                centroids, extends, similarity_max_list, points = find_clusters(points.detach().cpu().numpy(), alignments, obs = None)
                output = [centroids, extends, similarity_max_list, points]

            debug_text += '### - Found ' + str(len(centroids)) + ' instances that might be target object.\n'
            if debug:
                output.append(debug_text)
            
            return output