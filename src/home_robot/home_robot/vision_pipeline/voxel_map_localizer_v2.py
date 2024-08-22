import numpy as np

import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import clip

from home_robot.utils.voxel import VoxelizedPointcloud

from typing import List, Optional, Tuple, Union
from torch import Tensor

from transformers import AutoProcessor, AutoModel, AutoTokenizer

from sklearn.cluster import DBSCAN

# from ultralytics import YOLOWorld
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import math
from PIL import Image

from openai import OpenAI
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def find_clusters(vertices: np.ndarray, similarity: np.ndarray, obs = None, crops = None):
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
    crops_list = []
    
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
            if crops is not None:
                target_crops = crops[labels == cluster_id]
                target_crops = target_crops[np.argmax(obs_values)]

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
            if crops is not None:
                crops_list.append(target_crops)

    if obs is not None:
        if crops is not None:
            return centroids, extends, similarity_max_list, points, obs_max_list, crops_list
        else:
            return centroids, extends, similarity_max_list, points, obs_max_list, None
    else:
        return centroids, extends, similarity_max_list, points

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class VoxelMapLocalizer():
    def __init__(self, voxel_map_wrapper = None, exist_model = 'internvl', clip_model = None, processor = None, device = 'cuda', siglip = True):
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        # self.clip_model, self.preprocessor = clip.load(model_config, device=device)
        self.siglip = siglip
        if clip_model is not None and processor is not None:
            self.clip_model, self.preprocessor = clip_model, processor
        elif not self.siglip:
            self.clip_model, self.preprocessor = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
            self.preprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        self.voxel_pcd = VoxelizedPointcloud().to(self.device)
        # self.exist_model = YOLOWorld("yolov8l-worldv2.pt")
        self.existence_checking_model = exist_model
        if exist_model == 'owlv2':
            print('WE ARE USING OWLV2!')
            self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.exist_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        elif exist_model == 'internvl' and torch.cuda.is_available():
            print('WE ARE USING INTERNVL!')
            path = 'OpenGVLab/InternVL2-4B'
            self.exist_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=True).eval().cuda()
            self.exist_processor = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        elif exist_model == 'gpt4o':
            print('WE ARE USING OPENAI GPT4o')
            self.gpt_client = OpenAI()
        else:
            print('YOU ARE USING NOTHING!')
        

    def add(self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        obs_count: Optional[Tensor] = None,
        crops: Optional[Tensor] = None
    ):
        points = points.to(self.device)
        if features is not None:
            features = features.to(self.device)
        if rgb is not None:
            rgb = rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        if crops is not None:
            crops = crops.to(self.device)
        self.voxel_pcd.add(points = points, 
                        features = features,
                        rgb = rgb,
                        weights = weights,
                        obs_count = obs_count,
                        crops = crops)

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries] 
        if self.siglip:
            inputs = self.preprocessor(text=queries, padding="max_length", return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.clip_model.device)
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
        else:
            text = clip.tokenize(queries).to(self.clip_model.device)
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
    
        # print(point_alignments.shape)
        return point_alignments

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    def localize_AonB(self, A, B = None, k_A = 10, k_B = 50):
        print("A is ", A)
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
        centroids, extends, similarity_max_list, points, obs_max_list, crops_list, debug_text = self.find_clusters_for_A(A, return_obs_counts = True, debug = debug)
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
                # detection_model_check = self.check_existence(A, obs)
                if self.existence_checking_model == 'owlv2':
                    # print('Checking existence with OWLV2')
                    detection_model_check = self.check_existence_with_owl(A, obs)
                elif self.existence_checking_model == 'internvl':
                    # print('Checking existence with Internvl')
                    if crops_list:
                        detection_model_check = self.check_existence_with_internvl(A, obs, crops = crops_list[idx])
                    else:
                        detection_model_check = self.check_existence_with_internvl(A, obs, crops = None)
                elif self.existence_checking_model == 'gpt4o':
                    detection_model_check = self.check_existence_with_gpt(A, obs)
                else:
                    # print('No available detection model')
                    detection_model_check = False
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

    def check_existence_with_gpt(self, text, obs_id):
        if obs_id <= 0:
            return False
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
        # depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        # rgb[depth > 3.0] = 0
        rgb = rgb[:, :, [2, 1, 0]]
        cv2.imwrite('temp.png', rgb.detach().numpy())
        base64_image = encode_image('temp.png')
        response = self.gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Question: Is there a " + text + " in the image? (format your answer in yes/no) Answer: "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }]
            }],
            max_tokens=300,
        )

        message = response.choices[0].message.content
        return message[:3].lower() == 'yes'

    def check_existence_with_internvl(self, text, obs_id, crops = None):
        if obs_id <= 0:
            return False
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
        # depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        # rgb[depth > 3.0] = 0
        rgb = rgb[:, :, [2, 1, 0]]
        if crops is not None:
            tl_x, tl_y, br_x, br_y = crops
            rgb = rgb[max(tl_y, 0): min(br_y, rgb.shape[0]), max(tl_x, 0): min(br_x, rgb.shape[1])]
        cv2.imwrite(text + '.png', rgb.detach().numpy())
        pixel_values = load_image(text + '.png', max_num=6).to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=200,
            do_sample=False,
        )
        with torch.no_grad():
            question = '<image>\n Question: is there a ' + text +  ' in the image? (format your answer in yes/no) Answer: '
            response = self.exist_model.chat(self.exist_processor, pixel_values, question, generation_config)
        return response[:3].lower() == 'yes'

    def check_existence_with_owl(self, text, obs_id):
        if obs_id <= 0:
            return False
        # rgb = np.load(self.log + '/rgb' + str(obs_id) + '.npy')
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
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
            min_depth = torch.min(depth[tl_y: br_y, tl_x: br_x].flatten())
            if min_depth < 2.5 and not torch.isnan(min_depth):
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
                if self.voxel_pcd._crops is not None:
                    crops = self.voxel_pcd._crops.detach().cpu().numpy()[mask]
                else:
                    crops = None
                centroids, extends, similarity_max_list, points, obs_max_list, crops_list = find_clusters(points.detach().cpu().numpy(), alignments, obs = obs_ids, crops = crops)
                if crops_list:
                    output = [centroids, extends, similarity_max_list, points, obs_max_list, crops_list]
                else:
                    output = [centroids, extends, similarity_max_list, points, obs_max_list, None]
            else:
                centroids, extends, similarity_max_list, points = find_clusters(points.detach().cpu().numpy(), alignments, obs = None)
                output = [centroids, extends, similarity_max_list, points]

            debug_text += '### - Found ' + str(len(centroids)) + ' instances that might be target object.\n'
            if debug:
                output.append(debug_text)
            
            return output
