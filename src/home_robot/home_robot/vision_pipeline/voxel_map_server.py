import zmq

from scannet import CLASS_LABELS_200

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, OwlViTForObjectDetection
import clip
from torchvision import transforms

import os
import wget
import time

import open3d as o3d

from matplotlib import pyplot as plt
# This VoxelizedPointCloud is exactly the same thing as that in home_robot.util.voxel, rewrite here just for easy debugging
from voxel import VoxelizedPointcloud
from voxel_map_localizer import VoxelMapLocalizer
from home_robot.agent.multitask import get_parameters
from home_robot.mapping.voxel import (
    SparseVoxelMapVoxel as SparseVoxelMap,
    SparseVoxelMapNavigationSpaceVoxel as SparseVoxelMapNavigationSpace,
    plan_to_frontier,
)
from home_robot.motion import (
    ConfigurationSpace,
    PlanResult,
    RRTConnect,
    Shortcut,
    SimplifyXYT,
    AStar
)
from home_robot.motion.stretch import HelloStretchKinematics

import datetime

import threading
import scipy

from transformers import AutoProcessor, AutoModel

def load_socket(port_number):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(port_number))

    return socket

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    A = np.array(A)
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(np.ascontiguousarray(A), flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def numpy_to_pcd(xyz: np.ndarray, rgb: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Create an open3d pointcloud from a single xyz/rgb pair"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def get_inv_intrinsics(intrinsics):
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics

def get_xyz(depth, pose, intrinsics):
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.from_numpy(pose)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.from_numpy(intrinsics)
    while depth.ndim < 4:
        depth = depth.unsqueeze(0)
    while pose.ndim < 3:
        pose = pose.unsqueeze(0)
    while intrinsics.ndim < 3:
        intrinsics = intrinsics.unsqueeze(0)
    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]
    
    xyz = xyz.unflatten(1, (height, width))

    return xyz

class ImageProcessor:
    def __init__(self,  
        owl = True, 
        siglip = True,
        device = 'cuda',
        min_depth = 0.25,
        max_depth = 2.0,
        img_port = 5555,
        text_port = 5556,
        pcd_path: str = None,
        navigation_only = False
    ):
        self.siglip = siglip
        current_datetime = datetime.datetime.now()
        self.log = 'debug_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.obs_count = 0
        self.owl = owl
        # If cuda is not available, then device will be forced to be cpu
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.pcd_path = pcd_path

        self.create_vision_model()
        self.create_obstacle_map()

        self.img_socket = load_socket(img_port)
        self.text_socket = load_socket(text_port)

        self.voxel_map_lock = threading.Lock()  # Create a lock for synchronizing access to `self.voxel_map_localizer`

        if not navigation_only:
            self.img_thread = threading.Thread(target=self._recv_image)
            self.img_thread.daemon = True
            self.img_thread.start()

        self.visualization_lock = threading.Lock()

        # self.text_thread = threading.Thread(target=self._recv_text)
        # self.text_thread.daemon = True
        # self.text_thread.start()
    
    def create_obstacle_map(self):
        print("- Load parameters")
        parameters = get_parameters("/data/peiqi/robot-controller/src/robot_hw_python/configs/default.yaml")
        self.default_expand_frontier_size = parameters["default_expand_frontier_size"]
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density = parameters["obs_min_density"],
            exp_min_density = parameters["exp_min_density"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get(
                "add_local_radius_points", default=True
            ),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get(
                "filters/median_filter_max_error", 0.01
            ),
            use_derivative_filter=parameters.get(
                "filters/use_derivative_filter", False
            ),
            derivative_filter_threshold=parameters.get(
                "filters/derivative_filter_threshold", 0.5
            )
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            HelloStretchKinematics(urdf_path = '/data/peiqi/robot-controller/assets/hab_stretch/urdf'),
            # step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Create a simple motion planner
        self.planner = AStar(self.space)

    def create_vision_model(self):
        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
        # self.clip_model.eval()
        if not self.siglip:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
            self.clip_preprocess = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.clip_model.eval()
        if self.owl:
            self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(self.device)
            if not os.path.exists('sam_vit_b_01ec64.pth'):
                wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
            sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
            self.mask_predictor = SamPredictor(sam)
            self.mask_predictor.model = self.mask_predictor.model.eval().to(self.device)
            self.texts = [['a photo of ' + text for text in CLASS_LABELS_200]]
        # self.voxel_map_localizer = VoxelMapLocalizer(device = self.device)
        self.voxel_map_localizer = VoxelMapLocalizer(device = 'cpu', siglip = self.siglip)
        if self.pcd_path is not None:
            print('Loading old semantic memory')
            self.voxel_map_localizer.voxel_pcd = torch.load(self.pcd_path)
            print('Finish loading old semantic memory')


    def recv_text(self):
        text = self.text_socket.recv_string()
        self.text_socket.send_string('Text recevied, waiting for robot pose')
        start_pose = recv_array(self.text_socket)

        # Do visual grounding
        if text != '':
            with self.voxel_map_lock:
                localized_point = self.voxel_map_localizer.localize_AonB(text)
                print('\n', text, localized_point, '\n')
            with self.visualization_lock:
                point = self.sample_navigation(start_pose, localized_point)
                plt.savefig(self.log + '/debug_' + text + '.png')
                plt.cla()
        # Do Frontier based exploration
        else:
            point = self.sample_frontier()
            # plt.savefig(self.log + '/get_frontier_debug_' + str(self.obs_count) + '.jpg')

        if point is None:
            print('Unable to find any target point, some exception might happen')
            send_array(self.text_socket, [])
        else:
            print('Target point is', point)
            res = self.planner.plan(start_pose, point)
            if res.success:
                traj = [pt.state for pt in res.trajectory]
                # If we are navigating to some object of interst, send (x, y, z) of 
                # the object so that we can make sure the robot looks at the object after navigation
                if text != '': 
                    traj.append(np.asarray(localized_point))
                send_array(self.text_socket, traj)
            else:
                print('[FAILURE]', res.reason)
                send_array(self.text_socket, [])

    def sample_navigation(self, start, point, max_tries = 10):
        goal = self.space.sample_target_point(start, point, self.planner, max_tries)
        if goal is not None:
            print("Sampled Goal:", goal)
            obstacles, explored = self.voxel_map.get_2d_map()
            start_pt = self.planner.to_pt(start)
            goal_pt = self.planner.to_pt(goal)
            point_pt = self.planner.to_pt(point)
            plt.scatter(start_pt[1], start_pt[0], s = 10)
            plt.scatter(goal_pt[1], goal_pt[0], s = 10)
            plt.scatter(point_pt[1], point_pt[0], s = 10)
            plt.imshow(obstacles)
        return goal

        # target_grid = self.voxel_map.xy_to_grid_coords(point[:2]).int()
        # obstacles, explored = self.voxel_map.get_2d_map()
        # point_mask = torch.zeros_like(explored)
        # point_mask[target_grid[0]: target_grid[0] + 2, target_grid[1]: target_grid[1] + 2] = True
        # try_count = 0
        # for goal in self.space.sample_near_mask(point_mask, radius_m=radius_m, debug = True):
        #     goal = goal.cpu().numpy()
        #     print("Sampled Goal:", goal)
        #     goal_is_valid = self.space.is_valid(goal, verbose=False)
        #     if verbose:
        #         print(" Goal is valid:", goal_is_valid)
        #     try_count += 1
        #     if try_count > max_tries:
        #         return None
        #     if not goal_is_valid:
        #         print(" -> resample goal.")
        #         continue
        #     return goal

    def sample_frontier(self):
        for goal in self.space.sample_closest_frontier(
            [0, 0, 0], verbose=True, debug=False, expand_size=self.default_expand_frontier_size
        ):
            if goal is None:
                return None
            goal = goal.cpu().numpy()
            print("Sampled Goal:", goal)
            show_goal = np.zeros(3)
            show_goal[:2] = goal[:2]
            goal_is_valid = self.space.is_valid(goal)
            print(" Goal is valid:", goal_is_valid)
            if not goal_is_valid:
                print(" -> resample goal.")
                continue
            return goal
            

    def _recv_image(self):
        while True:
            data = recv_array(self.img_socket)
            print('Image received')
            start_time = time.time()
            self.process_rgbd_images(data)
            process_time = time.time() - start_time
            print('Image processing takes', process_time, 'seconds')
            self.img_socket.send_string('processing took ' + str(process_time) + ' seconds')

    def forward_one_block(self, resblocks, x):
        q, k, v = None, None, None
        y = resblocks.ln_1(x)
        y = F.linear(y, resblocks.attn.in_proj_weight, resblocks.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C//3).permute(2, 0, 1, 3).reshape(3*N, L, C//3)
        y = F.linear(y, resblocks.attn.out_proj.weight, resblocks.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + resblocks.mlp(resblocks.ln_2(v))

        return v

    def extract_mask_clip_features(self, x, image_shape):
        with torch.no_grad():
            x = self.clip_model.visual.conv1(x)
            N, L, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            for idx in range(self.clip_model.visual.transformer.layers):
                if idx == self.clip_model.visual.transformer.layers - 1:
                    break
                x = self.clip_model.visual.transformer.resblocks[idx](x)
            x = self.forward_one_block(self.clip_model.visual.transformer.resblocks[-1], x)
            x = x[1:]
            x = x.permute(1, 0, 2)
            x = self.clip_model.visual.ln_post(x)
            x = x @ self.clip_model.visual.proj
            feat = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode = 'bilinear', align_corners = True)
        feat = F.normalize(feat, dim = 1)
        return feat.permute(0, 2, 3, 1)
    
    def run_mask_clip(self, rgb, mask, world_xyz):
        # This code verify whether image is BGR, if it is RGB, then you should transform images into BGR
        # cv2.imwrite('debug.jpg', np.asarray(transforms.ToPILImage()(rgb), dtype = np.uint8))

        with torch.no_grad():
            if self.device == 'cpu':
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device)
            else:
                input = self.clip_preprocess(transforms.ToPILImage()(rgb)).unsqueeze(0).to(self.device).half()
            features = self.extract_mask_clip_features(input, rgb.shape[-2:])[0].cpu()

        # Let MaskClip do segmentation, the results should be reasonable but do not expect it to be accurate

        # text = clip.tokenize(["a keyboard", "a human"]).to(self.device)
        # image_vis = np.array(rgb.permute(1, 2, 0))
        # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text)
        #     text_features = F.normalize(text_features, dim = -1)
        #     output = torch.argmax(features.float() @ text_features.T.float().cpu(), dim = -1)
        # segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
        # segmentation_color_map[np.asarray(output) == 0] = [0, 255, 0]
        # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
        # cv2.imwrite("seg" + str(self.obs_count) + ".jpg", cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            
        valid_xyz = world_xyz[~mask]
        features = features[~mask]
        valid_rgb = rgb.permute(1, 2, 0)[~mask]
        if len(valid_xyz) != 0:
            self.add_to_voxel_pcd(valid_xyz, features, valid_rgb)
    
    def run_owl_sam_clip(self, rgb, mask, world_xyz):
        with torch.no_grad():
            inputs = self.owl_processor(text=self.texts, images=rgb, return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.device)
            outputs = self.owl_model(**inputs)
            target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
            results = self.owl_processor.post_process_object_detection(outputs=outputs, threshold=0.15, target_sizes=target_sizes)
            if len(results[0]['boxes']) == 0:
                return

            self.mask_predictor.set_image(rgb.permute(1,2,0).numpy())
            bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
            transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(bounding_boxes.detach().to(self.device), rgb.shape[-2:])
            masks, _, _= self.mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks[:, 0, :, :].cpu()
            
            # Debug code, visualize all bounding boxes and segmentation masks

            image_vis = np.array(rgb.permute(1, 2, 0))
            segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            for idx, box in enumerate(bounding_boxes):
                tl_x, tl_y, br_x, br_y = box
                tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
                cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
            for vis_mask in masks:
                segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            cv2.imwrite(self.log + "/seg" + str(self.obs_count) + ".jpg", image_vis)
    
            crops = []
            if not self.siglip:
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
                features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            else:
                for box in bounding_boxes:
                    tl_x, tl_y, br_x, br_y = box
                    crops.append(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])
                inputs = self.clip_preprocess(images = crops, padding="max_length", return_tensors="pt").to(self.device)
                features = self.clip_model.get_image_features(**inputs)
            # for box in bounding_boxes:
            #     tl_x, tl_y, br_x, br_y = box
            #     crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
            # features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
            features = F.normalize(features, dim = -1).cpu()
            
            # Debug code, let the clip select bounding boxes most aligned with a text query, used to check whether clip embeddings for
            # bounding boxes are reasonable

            # text = clip.tokenize(["a coco cola"]).to(self.device)

            # with torch.no_grad():
            #     text_features = self.clip_model.encode_text(text)
            #     text_features = F.normalize(text_features, dim = -1)
            #     i = torch.argmax(features.float() @ text_features.T.float().cpu()).item()
            # image_vis = np.array(rgb.permute(1, 2, 0))
            # segmentation_color_map = np.zeros(image_vis.shape, dtype=np.uint8)
            # cv2.imwrite('clean_' + str(self.obs_count) + '.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            # tl_x, tl_y, br_x, br_y = bounding_boxes[i]
            # tl_x, tl_y, br_x, br_y = tl_x.item(), tl_y.item(), br_x.item(), br_y.item()
            # cv2.rectangle(image_vis, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), (255, 0, 0), 2)
            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR) 
            # for vis_mask in masks:
            #     segmentation_color_map[vis_mask.detach().cpu().numpy()] = [0, 255, 0]
            # image_vis = cv2.addWeighted(image_vis, 0.7, segmentation_color_map, 0.3, 0)
            # cv2.imwrite("seg" + str(self.obs_count) + ".jpg", image_vis)


        for idx, (sam_mask, feature) in enumerate(zip(masks.cpu(), features.cpu())):
            valid_mask = torch.logical_and(~mask, sam_mask)
            # Debug code, check whether every mask makes sense

            # plt.subplot(2, 2, 1)
            # plt.imshow(~mask)
            # plt.axis('off')
            # plt.subplot(2, 2, 2)
            # plt.imshow(sam_mask)
            # plt.axis('off')
            # plt.subplot(2, 2, 3)
            # plt.imshow(valid_mask)
            # plt.axis('off')
            # plt.savefig('seg_' + str(idx) + '.jpg')
            valid_xyz = world_xyz[valid_mask]
            if valid_xyz.shape[0] == 0:
                continue
            feature = feature.repeat(valid_xyz.shape[0], 1)
            valid_rgb = rgb.permute(1, 2, 0)[valid_mask]
            self.add_to_voxel_pcd(valid_xyz, feature, valid_rgb)
    
    def add_to_voxel_pcd(self, valid_xyz, feature, valid_rgb, weights = None, threshold = 0.95):
        # Adding all points to voxelizedPointCloud is useless and expensive, we should exclude threshold of all points
        selected_indices = torch.randperm(len(valid_xyz))[:int((1 - threshold) * len(valid_xyz))]
        if len(selected_indices) == 0:
            return
        if valid_xyz is not None:
            valid_xyz = valid_xyz[selected_indices]
        if feature is not None:
            feature = feature[selected_indices]
        if valid_rgb is not None:
            valid_rgb = valid_rgb[selected_indices]
        if weights is not None:
            weights = weights[selected_indices]
        with self.voxel_map_lock:
            self.voxel_map_localizer.add(points = valid_xyz, 
                                    features = feature,
                                    rgb = valid_rgb,
                                    weights = weights)

    def debug(self, log, number):
        for i in range(number):
            if i % 20 == 0:
                print(i)
                if i % 100 != 20:
                    continue
                points, _, _, rgb = self.voxel_map_localizer.voxel_pcd.get_pointcloud()
                points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
                pcd = numpy_to_pcd(points, rgb / 255)
                if not os.path.exists('debug'):
                    os.mkdir('debug')
                o3d.io.write_point_cloud('debug' + '/debug_' + str(i) + '.pcd', pcd)
            rgb = np.load(log + '/rgb' + str(i) + '.npy')
            depth = np.load(log + '/depth' + str(i) + '.npy')
            intrinsics = np.load(log + '/intrinsics' + str(i) + '.npy')
            pose = np.load(log + '/pose' + str(i) + '.npy')
            world_xyz = get_xyz(depth, pose, intrinsics).squeeze(0)

            rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth)
            rgb = rgb.permute(2, 0, 1).to(torch.uint8)

            median_depth = torch.from_numpy(
                scipy.ndimage.median_filter(depth, size=5)
            )
            median_filter_error = (depth - median_depth).abs()
            valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
            valid_depth = (
                valid_depth
                & (median_filter_error < 0.01).bool()
            )

            if self.owl:
                self.run_owl_sam_clip(rgb, ~valid_depth, world_xyz)
            else:
                self.run_mask_clip(rgb, ~valid_depth, world_xyz)

    def test_DBSCAN(self, text):
        centroids, extends, similarity_max_list, target_points = self.voxel_map_localizer.find_clusters_for_A(text)
        target_point = target_points[np.array(similarity_max_list).argmax()]
        points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
        # points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
        # points = np.concatenate((points, target_point))
        # rgb = np.concatenate((rgb / 255, np.array([[1, 0, 0] for _ in range(len(target_point))])))
        if not os.path.exists('debug'):
            os.mkdir('debug')
        pcd = numpy_to_pcd(points, rgb / 255)
        o3d.io.write_point_cloud('debug/debug.pcd', pcd)
        pcd = numpy_to_pcd(target_point, np.ones((len(target_point), 3)))
        o3d.io.write_point_cloud('debug/' + text + '.pcd', pcd)

    # def visualize_res(self, text = 'red cup', threshold = [10, 50, 100, 500, 1000]):
    #     points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
    #     points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
    #     rgb = rgb / 255
    #     pcd = numpy_to_pcd(points, rgb)
    #     if not os.path.exists(text):
    #         os.mkdir(text)
    #     o3d.io.write_point_cloud(text + '/debug.pcd', pcd)
    #     alignments = self.voxel_map_localizer.find_alignment_over_model(text)
    #     for k_A in threshold:
    #         rgb[alignments[0].topk(k = k_A, dim = -1).indices.numpy()] = np.array([1, 0, 0])
    #         pcd = numpy_to_pcd(points, rgb)
    #         o3d.io.write_point_cloud(text + '/debug_' + str(k_A) + '.pcd', pcd)

    # def visualize_hist(self, text = 'red cup'):
    #     alignments = self.voxel_map_localizer.find_alignment_over_model(text)
    #     # negatives = ['object', 'texture', 'stuff', 'thing']
    #     # negative_alignments = self.voxel_map_localizer.find_alignment_over_model(negatives)
    #     # alignments = (alignments.exp() / (negative_alignments.exp() + alignments.exp())).min(dim = 0).values
    #     plt.title(text)
    #     plt.hist(alignments.detach().numpy())
    #     if not os.path.exists('debug'):
    #         os.mkdir('debug')
    #     plt.savefig('debug/' + text + '.jpg')
    #     plt.cla()

    # def visualize_cs(self, text, threshold = [0.1]):
    #     # points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
    #     # points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
    #     # rgb = rgb / 255
    #     # pcd = numpy_to_pcd(points, rgb)
    #     # if not os.path.exists('debug1'):
    #     #     os.mkdir('debug1')
    #     # o3d.io.write_point_cloud('debug1' + '/debug.pcd', pcd)
    #     # alignments = self.voxel_map_localizer.find_alignment_over_model(text)
    #     # negatives = ['object', 'texture', 'stuff', 'thing']
    #     # negative_alignments = self.voxel_map_localizer.find_alignment_over_model(negatives)
    #     # alignments = (alignments.exp() / (negative_alignments.exp() + alignments.exp())).min(dim = 0).values
    #     # for k_A in threshold:
    #     #     rgb[alignments.detach().numpy() > k_A] = np.array([1, 0, 0])
    #     #     pcd = numpy_to_pcd(points, rgb)
    #     #     o3d.io.write_point_cloud('debug1' + '/debug_' + text + '_' + str(k_A) + '.pcd', pcd)

    #     points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
    #     points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
    #     rgb = rgb / 255
    #     pcd = numpy_to_pcd(points, rgb)
    #     if not os.path.exists('debug'):
    #         os.mkdir('debug')
    #     o3d.io.write_point_cloud('debug' + '/debug.pcd', pcd)
    #     alignments = self.voxel_map_localizer.find_alignment_over_model(text)
    #     for k_A in threshold:
    #         rgb[alignments[0].detach().numpy() > k_A] = np.array([1, 0, 0])
    #         pcd = numpy_to_pcd(points, rgb)
    #         o3d.io.write_point_cloud('debug' + '/debug_' + text + '_' + str(k_A) + '.pcd', pcd)

    def process_rgbd_images(self, data):
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.obs_count += 1
        w, h = data[:2]
        w, h = int(w), int(h)
        rgb = data[2: 2 + w * h * 3].reshape(w, h, 3)
        depth = data[2 + w * h * 3: 2 + w * h * 3 + w * h].reshape(w, h)
        intrinsics = data[2 + w * h * 3 + w * h: 2 + w * h * 3 + w * h + 9].reshape(3, 3)
        pose = data[2 + w * h * 3 + w * h + 9: 2 + w * h * 3 + w * h + 9 + 16].reshape(4, 4)
        world_xyz = get_xyz(depth, pose, intrinsics).squeeze(0)

        # cv2.imwrite('debug/rgb' + str(self.obs_count) + '.jpg', rgb[:, :, [2, 1, 0]])
        np.save(self.log + '/rgb' + str(self.obs_count) + '.npy', rgb)
        np.save(self.log + '/depth' + str(self.obs_count) + '.npy', depth)
        np.save(self.log + '/intrinsics' + str(self.obs_count) + '.npy', intrinsics)
        np.save(self.log + '/pose' + str(self.obs_count) + '.npy', pose)

        rgb, depth = torch.from_numpy(rgb), torch.from_numpy(depth)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)

        median_depth = torch.from_numpy(
            scipy.ndimage.median_filter(depth, size=5)
        )
        median_filter_error = (depth - median_depth).abs()
        valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
        valid_depth = (
            valid_depth
            & (median_filter_error < 0.01).bool()
        )

        # with self.voxel_map_lock:
        #     self.voxel_map_localizer.voxel_pcd.clear_points(depth, intrinsics, pose)

        if self.owl:
            self.run_owl_sam_clip(rgb, ~valid_depth, world_xyz)
        else:
            self.run_mask_clip(rgb, ~valid_depth, world_xyz)

        self.voxel_map.add(
            camera_pose = torch.Tensor(pose), 
            rgb = torch.Tensor(rgb).permute(1, 2, 0), 
            depth = torch.Tensor(depth), 
            camera_K = torch.Tensor(intrinsics)
        )
        obs, exp = self.voxel_map.get_2d_map()
        with self.visualization_lock:
            plt.subplot(2, 1, 1)
            plt.imshow(obs.detach().cpu().numpy())
            plt.title("obstacles")
            plt.axis("off")
            plt.subplot(2, 1, 2)
            plt.imshow(exp.detach().cpu().numpy())
            plt.title("explored")
            plt.axis("off")
            plt.savefig(self.log + '/debug' + str(self.obs_count) + '.jpg')
            plt.cla()

if __name__ == "__main__":
    imageProcessor = ImageProcessor(pcd_path = None)
    # imageProcessor = ImageProcessor(pcd_path = 'debug_2024-06-02_18-20-46/memory.pt', navigation_only = True)  
    # for text in ['red cup', 'red bowl', 'green bowl', 'blue whiteboard care bottle', 'white table', 'coffee machine', 'sink', 'microwave', 'orange tape', 'black chair', 'pink spray', 'purple moov body spray']:
    #     print(text)
        # imageProcessor.visualize_res(text = text) 
        # imageProcessor.visualize_hist(text = text)
        # imageProcessor.visualize_cs(text = text)
        # imageProcessor.test_DBSCAN(text = text)
    try:  
        while True:
            imageProcessor.recv_text()
    except KeyboardInterrupt:
        if not imageProcessor.voxel_map_localizer.voxel_pcd._points is None:
            print('Stop streaming images and write memory data, might take a while, please wait')
            torch.save(imageProcessor.voxel_map_localizer.voxel_pcd, imageProcessor.log + '/memory.pt')
            points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
            points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
            pcd = numpy_to_pcd(points, rgb / 255)
            o3d.io.write_point_cloud(imageProcessor.log + '/debug.pcd', pcd)
            print('finished')
