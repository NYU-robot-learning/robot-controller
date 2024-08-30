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
    def __init__(self, voxel_map_wrapper = None, exist_model = 'owl', clip_model = None, processor = None, device = 'cuda', siglip = True):
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
            path = 'OpenGVLab/InternVL2-2B'
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
                # detection_model_check = self.check_existence(A, obs)
                if self.existence_checking_model == 'owlv2':
                    # print('Checking existence with OWLV2')
                    detection_model_check = self.check_existence_with_owl(A, obs)
                elif self.existence_checking_model == 'internvl':
                    # print('Checking existence with Internvl')
                    detection_model_check = self.check_existence_with_internvl(A, obs)
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

    def check_existence_with_internvl(self, text, obs_id):
        if obs_id <= 0:
            return False
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
        # depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        # rgb[depth > 3.0] = 0
        rgb = rgb[:, :, [2, 1, 0]]
        cv2.imwrite('temp.png', rgb.detach().numpy())
        pixel_values = load_image('temp.png', max_num=6).to(torch.bfloat16).cuda()

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
                centroids, extends, similarity_max_list, points, obs_max_list = find_clusters(points.detach().cpu().numpy(), alignments, obs = obs_ids)
                output = [centroids, extends, similarity_max_list, points, obs_max_list]
            else:
                centroids, extends, similarity_max_list, points = find_clusters(points.detach().cpu().numpy(), alignments, obs = None)
                output = [centroids, extends, similarity_max_list, points]

            debug_text += '### - Found ' + str(len(centroids)) + ' instances that might be target object.\n'
            if debug:
                output.append(debug_text)
            
            return output

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
class LLM_VoxelMapLocalizer():
    def __init__(self, voxel_map_wrapper = None, exist_model = 'gpt-4o', loc_model = 'owlv2', device = 'cuda'):
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        self.voxel_pcd = VoxelizedPointcloud(voxel_size=0.2).to(self.device)
        # self.exist_model = YOLOWorld("yolov8l-worldv2.pt")
        self.existence_checking_model = exist_model
        if exist_model == 'gpt-4o':
            print('WE ARE USING OPENAI GPT4o')
            self.gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            print('YOU ARE USING NOTHING!')
        self.location_checking_model = loc_model
        if loc_model == 'owlv2':
            self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.exist_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
            print('WE ARE USING OWLV2 FOR LOCALIZATION!')
        else:
            print('YOU ARE USING VOXEL MAP FOR LOCALIZATION!')
        
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

    def owl_locater(self, A, encoded_image, timestamps_lst):
        for i in sorted(timestamps_lst, reverse=True):
            image_info = encoded_image[i][-1]
            image = image_info['image']
            box = None
                
            inputs = self.exist_processor(text=A, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.exist_model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.exist_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)[0]

            if len(results["scores"]) > 0:
                cur_score = torch.max(results["scores"]).item()
                max_score_index = torch.argmax(results["scores"])
                box = results["boxes"][max_score_index].tolist()
            if box is not None:
                xmin, ymin, xmax, ymax = map(int, box)
                mask = np.zeros(image_info['depth'].shape, dtype=np.uint8)
                mask[ymin:ymax, xmin:xmax] = 255
                xyz = image_info['xyz']        
                masked_xyz = xyz[mask.flatten() > 0]
                centroid = np.stack([torch.mean(masked_xyz[:, 0]), torch.mean(masked_xyz[:, 1]), torch.mean(masked_xyz[:, 2])]).T
                debug_text = '#### - Obejct is detected in observations where instance' + str(i + 1) + ' comes from. **😃** Directly navigate to it.\n'
                return centroid, debug_text, i, masked_xyz
        debug_text = '#### - All instances are not the target! Maybe target object has not been observed yet. **😭**\n'
        return None, debug_text, None, None

    def process_chunk(self, chunk, sys_prompt, user_prompt):
        for i in range(10):
            try:
                response = self.gpt_client.chat.completions.create(
                    model=self.existence_checking_model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.0,
                )                
                timestamps = response.choices[0].message.content
                if 'None' in timestamps:
                    return None
                else:
                    return list(map(int, timestamps.split(', ')))
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(15)
        return "Execution Failed"

    def llm_locator(self, A, encoded_image, context_length = 30):
        timestamps_lst = []

        sys_prompt = f"""
        You need to find all the timestamp that the object exist in the image in plain text, without any unnecessary explanation or space. If the object never exist, please directly output None.
        
        Example 1:
        Input:
        bottle

        Output: 
        1, 4, 6, 9

        Example 2:
        Input: desk

        Output: 
        None
        """

        user_prompt = f"""The object you need to find is {A}"""
        
        content = [item for sublist in list(encoded_image.values()) for item in sublist[:2]]
        content_chunks = [content[i:i + 2 * context_length] for i in range(0, len(content), 2 * context_length)]
        chunk_num = len(content_chunks)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(self.process_chunk, chunk, sys_prompt, user_prompt): chunk for chunk in content_chunks}
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        timestamps_lst.extend(result)    
                except Exception as e:
                    print(f"Exception occurred: {e}")
        
        return self.owl_locater(A, encoded_image, timestamps_lst)

    def localize_A_v2(self, A, debug = True, return_debug = False, count_threshold = 3):
        encoded_image = {}

        counts = torch.bincount(self.voxel_map_wrapper.voxel_pcd._obs_counts)
        cur_obs = max(self.voxel_map_wrapper.voxel_pcd._obs_counts)
        filtered_obs = (counts > count_threshold).nonzero(as_tuple=True)[0].tolist()
        filtered_obs = set(filtered_obs + [i for i in range(cur_obs-10, cur_obs+1)])

        for obs_id in filtered_obs: 
            rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb.numpy()
            depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
            camera_pose = self.voxel_map_wrapper.observations[obs_id - 1].camera_pose
            camera_K = self.voxel_map_wrapper.observations[obs_id - 1].camera_K

            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )
            depth = depth.numpy()
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            encoded_image[obs_id] = [{"type": "text", "text": f"Following is the image took on timestep {obs_id}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_encoded}"}
                }, {'image':image, 'xyz':full_world_xyz, 'depth':depth}]
        target_point, debug_text, obs, point = self.llm_locator(A, encoded_image, context_length = 30)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs, point
