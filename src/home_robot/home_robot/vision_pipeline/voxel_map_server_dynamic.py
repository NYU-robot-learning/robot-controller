import zmq

from scannet import CLASS_LABELS_200

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from segment_anything import sam_model_registry, SamPredictor
# from transformers import AutoProcessor, OwlViTForObjectDetection
from ultralytics import YOLO, SAM, YOLOWorld

import clip
from torchvision import transforms

import os
import wget
import time

import open3d as o3d

from matplotlib import pyplot as plt
# This VoxelizedPointCloud is exactly the same thing as that in home_robot.util.voxel, rewrite here just for easy debugging
from voxel import VoxelizedPointcloud
from home_robot.utils.voxel import VoxelizedPointcloud
from voxel_map_localizer import VoxelMapLocalizer
from home_robot.agent.multitask import get_parameters
from home_robot.mapping.voxel import (
    SparseVoxelMapVoxel as SparseVoxelMap,
    SparseVoxelMapNavigationSpaceVoxelDynamic as SparseVoxelMapNavigationSpace,
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
import hydra

from transformers import AutoProcessor, AutoModel
import rerun as rr


from io import BytesIO
from PIL import Image

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

def send_rgb_img(socket, img):
    img = img.astype(np.uint8) 
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    socket.send(img_encoded.tobytes())

def recv_rgb_img(socket):
    img = socket.recv()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def send_depth_img(socket, depth_img):
    depth_img = (depth_img * 1000).astype(np.uint16)
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]  # Compression level from 0 (no compression) to 9 (max compression)
    _, depth_img_encoded = cv2.imencode('.png', depth_img, encode_param)
    socket.send(depth_img_encoded.tobytes())

def recv_depth_img(socket):
    depth_img = socket.recv()
    depth_img = np.frombuffer(depth_img, dtype=np.uint8)
    depth_img = cv2.imdecode(depth_img, cv2.IMREAD_UNCHANGED)
    depth_img = (depth_img / 1000.)
    return depth_img

def send_everything(socket, rgb, depth, intrinsics, pose):
    send_rgb_img(socket, rgb)
    socket.recv_string()
    send_depth_img(socket, depth)
    socket.recv_string()
    send_array(socket, intrinsics)
    socket.recv_string()
    send_array(socket, pose)
    socket.recv_string()

def recv_everything(socket):
    rgb = recv_rgb_img(socket)
    socket.send_string('')
    depth = recv_depth_img(socket)
    socket.send_string('')
    intrinsics = recv_array(socket)
    socket.send_string('')
    pose = recv_array(socket)
    socket.send_string('')
    return rgb, depth, intrinsics, pose

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
        img_port = 5560,
        text_port = 5561,
        pcd_path: str = None,
        navigation_only = False,
        rerun = True
    ):
        self.siglip = siglip
        current_datetime = datetime.datetime.now()
        self.log = 'debug_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.rerun = rerun
        if self.rerun:
            rr.init(self.log, spawn = True)
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
        self.value_map = torch.zeros(self.voxel_map.grid_size)

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
            if not os.path.exists('sam_vit_b_01ec64.pth'):
                wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', out = 'sam_vit_b_01ec64.pth')
            sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
            self.mask_predictor = SamPredictor(sam)
            self.mask_predictor.model = self.mask_predictor.model.eval().to(self.device)
            # self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
            # self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval().to(self.device)
            # self.texts = [['a photo of ' + text for text in CLASS_LABELS_200]]
            self.yolo_model = YOLOWorld('yolov8l-worldv2.pt')
            self.texts = CLASS_LABELS_200
            self.yolo_model.set_classes(self.texts)
        # self.voxel_map_localizer = VoxelMapLocalizer(device = self.device)
        self.voxel_map_localizer = VoxelMapLocalizer(log = self.log, device = 'cpu', siglip = self.siglip)
        if self.pcd_path is not None:
            print('Loading old semantic memory')
            self.voxel_map_localizer.voxel_pcd = torch.load(self.pcd_path)
            print('Finish loading old semantic memory')

    def debug_text(self):
        text = input('Enter debug text: ')
        start_pose = np.array([0, 0, 0])

        # Do visual grounding
        if text != '':
            with self.voxel_map_lock:
                localized_point = self.voxel_map_localizer.localize_AonB(text)
                debug_point = self.voxel_map_localizer.localize_A_v2(text)
                centroids, extends, similarity_max_list, points, obs_id = self.voxel_map_localizer.find_clusters_for_A(text, return_obs_counts = True)
                print(centroids, similarity_max_list, obs_id)
                print(debug_point, localized_point)
                if self.rerun:
                    for idx, point in enumerate(points):
                        rr.log("Res/pointcloud_" + str(idx), rr.Points3D(point, colors=torch.Tensor([1, 0, 0]).repeat(len(point), 1), radii=0.03))

                print(self.voxel_map_localizer.find_alignment_over_model(text).cpu().max().item())

                print('\n', text, localized_point, '\n')
            obs_id = int(self.voxel_map_localizer.find_obs_id_for_A(text).detach().cpu().item())
            rgb = np.load(self.log + '/rgb' + str(obs_id) + '.npy')
            if not self.rerun:
                cv2.imwrite(self.log + '/debug_' + text + '.png', rgb[:, :, [2, 1, 0]])
            else:
                rr.log('Memory_image', rr.Image(rgb))
            print('Points are observed from the ', obs_id, 'th image')
        # Do Frontier based exploration
        else:
            localized_point = self.sample_frontier()
            print('\n', localized_point, '\n')
        
        if localized_point is None:
            print('Unable to find any target point, some exception might happen')
            return
        
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        point = self.sample_navigation(start_pose, localized_point)
        # if text != '':
        #     plt.savefig('debug_' + text + str(self.obs_count) + '.png')
        # else:
        #     plt.savefig('debug_exploration' + str(self.obs_count) + '.png')
        # plt.clf()

        if point is None:
            print('Unable to find any target point, some exception might happen')
        else:
            print('Target point is', point)
            res = self.planner.plan(start_pose, point)
            if res.success:
                waypoints = [pt.state for pt in res.trajectory]
                # If we are navigating to some object of interst, send (x, y, z) of 
                # the object so that we can make sure the robot looks at the object after navigation
                finished = len(waypoints) <= 7
                # print('Waypoints before pruning', waypoints)
                if not finished:
                    waypoints = waypoints[:7]
                # print('Waypoints after pruning', waypoints)
                traj = self.planner.clean_path_for_xy(waypoints)
                # print('If we clean waypoints', traj)
                # Remove the starting point
                traj = traj[1:]
                # traj = [waypoints[-1]]
                # print('If we keep the last waypoint', traj)
                if finished:
                    traj.append([np.nan, np.nan, np.nan])
                    if isinstance(localized_point, torch.Tensor):
                        localized_point = localized_point.tolist()
                    traj.append(localized_point)
                print('Planned trajectory:', traj)
            else:
                print('[FAILURE]', res.reason)

    def recv_text(self):
        text = self.text_socket.recv_string()
        self.text_socket.send_string('Text recevied, waiting for robot pose')
        start_pose = recv_array(self.text_socket)
        if self.rerun:
            rr.set_time_sequence("frame", self.obs_count)

        debug_text = ''
        mode = 'navigation'
        obs = None
        # Do visual grounding
        if text != '':
            with self.voxel_map_lock:
                localized_point, debug_text, obs, pointcloud = self.voxel_map_localizer.localize_A_v2(text, debug = True, return_debug = True)
            if localized_point is not None:
                rr.log("/object", rr.Points3D(localized_point, colors=torch.Tensor([1, 0, 0]), radii=0.15))
        # Do Frontier based exploration
        if text is None or text == '' or localized_point is None:
            debug_text += '## Navigation fails, so robot starts exploring environments.\n'
            localized_point = self.sample_frontier(start_pose, text)
            mode = 'exploration'
            rr.log("/object", rr.Points3D([0, 0, 0], colors=torch.Tensor([1, 0, 0]), radii=0))
            print('\n', localized_point, '\n')
        
        if localized_point is None:
            print('Unable to find any target point, some exception might happen')
            send_array(self.text_socket, [])
            return
        
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        point = self.sample_navigation(start_pose, localized_point)
        if mode == 'navigation' and np.min(np.linalg.norm(np.asarray(localized_point)[:2] - np.asarray(pointcloud)[:, :2], axis = -1)) > 1.0:
            localized_point = self.sample_frontier(start_pose, None)
            mode = 'exploration'
            point = self.sample_navigation(start_pose, localized_point)
            debug_text += '## All reachable points of robot are too far from the target object, explore to find new paths. \n'

        if self.rerun:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img = np.array(img)
            buf.close()
            rr.log('2d_map', rr.Image(img))
        else:
            if text != '':
                plt.savefig(self.log + '/debug_' + text + str(self.obs_count) + '.png')
            else:
                plt.savefig(self.log + '/debug_exploration' + str(self.obs_count) + '.png')
        plt.clf()

        if self.rerun:
            if text is not None and text != '':
                debug_text = '## The goal is to navigate to **' + text + '**.\n' + debug_text
            else:
                debug_text = '## The robot does not need to navigate to anything. It just looks around. \n'
            rr.log("explanation", rr.TextDocument(debug_text, media_type = rr.MediaType.MARKDOWN))

        if obs is not None and mode == 'navigation':
            rgb = np.load(self.log + '/rgb' + str(obs) + '.npy')
            if not self.rerun:
                cv2.imwrite(self.log + '/debug_' + text + '.png', rgb[:, :, [2, 1, 0]])
            else:
                rr.log('/Memory_image', rr.Image(rgb))
        else:
            if self.rerun:
                rr.log('/Memory_image', rr.Image(np.zeros((256, 256, 3))))

        waypoints = None
        if point is None:
            print('Unable to find any target point, some exception might happen')
            send_array(self.text_socket, [])
        else:
            print('Target point is', point)
            res = self.planner.plan(start_pose, point)
            if res.success:
                waypoints = [pt.state for pt in res.trajectory]
                # If we are navigating to some object of interst, send (x, y, z) of 
                # the object so that we can make sure the robot looks at the object after navigation
                print(waypoints[-1][:2], start_pose[:2])
                finished = len(waypoints) <= 10 and mode == 'navigation'
                if not finished:
                    waypoints = waypoints[:7]
                traj = self.planner.clean_path_for_xy(waypoints)
                # traj = traj[1:]
                if finished:
                    traj.append([np.nan, np.nan, np.nan])
                    if isinstance(localized_point, torch.Tensor):
                        localized_point = localized_point.tolist()
                    traj.append(localized_point)
                print('Planned trajectory:', traj)
                send_array(self.text_socket, traj)
            else:
                print('[FAILURE]', res.reason)
                send_array(self.text_socket, [])
            
        if waypoints is not None:
            rr.log("/direction", rr.Arrows3D(origins = [start_pose[0], start_pose[1], 1.5], \
                                            vectors = [waypoints[-1][0] - start_pose[0], waypoints[-1][1] - start_pose[1], 0]\
                                                , colors=torch.Tensor([0, 1, 0]), radii=0.05))
            rr.log("/start", rr.Points3D([start_pose[0], start_pose[1], 1.5], colors=torch.Tensor([0, 0, 1]), radii=0.1))

    def sample_navigation(self, start, point):
        plt.clf()
        obstacles, _ = self.voxel_map.get_2d_map()
        plt.imshow(obstacles)
        if point is None:
            start_pt = self.planner.to_pt(start)
            plt.scatter(start_pt[1], start_pt[0], s = 10)
            return None
        goal = self.space.sample_target_point(start, point, self.planner)
        print("point:", point, "goal:", goal)
        obstacles, explored = self.voxel_map.get_2d_map()
        start_pt = self.planner.to_pt(start)
        plt.scatter(start_pt[1], start_pt[0], s = 15, c = 'b')
        point_pt = self.planner.to_pt(point)
        plt.scatter(point_pt[1], point_pt[0], s = 15, c = 'g')
        if goal is not None:
            goal_pt = self.planner.to_pt(goal)
            plt.scatter(goal_pt[1], goal_pt[0], s = 10, c = 'r')
        return goal

    def sample_frontier(self, start_pose = [0, 0, 0], text = None):
        # for goal in self.space.sample_closest_frontier(
        #     [0, 0, 0], verbose=True, debug=False, expand_size=self.default_expand_frontier_size
        # ):
        #     if goal is None:
        #         return None
        #     goal = goal.cpu().numpy()
        #     print("Sampled Goal:", goal)
        #     # show_goal = np.zeros(3)
        #     # show_goal[:2] = goal[:2]
        #     # goal_is_valid = self.space.is_valid(goal)
        #     # print(" Goal is valid:", goal_is_valid)
        #     # if not goal_is_valid:
        #     #     print(" -> resample goal.")
        #     #     continue
        #     return goal
        with self.voxel_map_lock:
            if text is not None and text != '':
                index, time_heuristics, alignments_heuristics, total_heuristics = self.space.sample_exploration(start_pose, self.planner, self.voxel_map_localizer, text, debug = False)
            else:
                index, time_heuristics, _, total_heuristics = self.space.sample_exploration(start_pose, self.planner, None, None, debug = False)
                alignments_heuristics = time_heuristics
                
        obstacles, explored = self.voxel_map.get_2d_map()
        plt.clf()
        plt.imshow(obstacles * 0.5 + explored * 0.5)
        plt.scatter(index[1], index[0], s = 20, c = 'r')
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
            

    def _recv_image(self):
        while True:
            # data = recv_array(self.img_socket)
            rgb, depth, intrinsics, pose = recv_everything(self.img_socket)
            print('Image received')
            start_time = time.time()
            self.process_rgbd_images(rgb, depth, intrinsics, pose)
            process_time = time.time() - start_time
            print('Image processing takes', process_time, 'seconds')
            print('processing took ' + str(process_time) + ' seconds')

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
            # inputs = self.owl_processor(text=self.texts, images=rgb, return_tensors="pt")
            # for input in inputs:
            #     inputs[input] = inputs[input].to(self.device)
            # outputs = self.owl_model(**inputs)
            # target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
            # results = self.owl_processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
            # if len(results[0]['boxes']) == 0:
            #     return
            results = self.yolo_model.predict(rgb.permute(1,2,0)[:, :, [2, 1, 0]].numpy(), conf=0.15, verbose=False)
            xyxy_tensor = results[0].boxes.xyxy
            if len(xyxy_tensor) == 0:
                return

            self.mask_predictor.set_image(rgb.permute(1,2,0).numpy())
            # bounding_boxes = torch.stack(sorted(results[0]['boxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
            bounding_boxes = torch.stack(sorted(xyxy_tensor, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse = True), dim = 0)
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
            if not self.rerun:
                cv2.imwrite(self.log + "/seg" + str(self.obs_count) + ".jpg", image_vis)
            else:
                rr.log('Segmentation mask', rr.Image(image_vis[:, :, [2, 1, 0]]))
    
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
                                    weights = weights,
                                    obs_count = self.obs_count)

    def load(self, log, number, load_memory = False, texts = None):
        self.voxel_map_localizer.log = log
        if load_memory:
            print('Loading semantic memory')
            self.voxel_map_localizer.voxel_pcd = torch.load(log + '/memory.pt')
            print('Finish oading semantic memory')
        for i in range(1, number + 1):
            self.obs_count += 1
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
            valid_depth = (depth < self.max_depth) & (depth > self.min_depth)
            valid_depth = (
                valid_depth
                & (median_filter_error < 0.01).bool()
            )

            self.voxel_map_localizer.voxel_pcd.clear_points(depth, torch.from_numpy(intrinsics), torch.from_numpy(pose))
            self.voxel_map.voxel_pcd.clear_points(depth, torch.from_numpy(intrinsics), torch.from_numpy(pose))

            self.voxel_map.add(
                camera_pose = torch.Tensor(pose), 
                rgb = torch.Tensor(rgb).permute(1, 2, 0), 
                depth = torch.Tensor(depth), 
                camera_K = torch.Tensor(intrinsics)
            )
            obs, exp = self.voxel_map.get_2d_map()

            if not load_memory:
                if self.owl:
                    self.run_owl_sam_clip(rgb, ~valid_depth, world_xyz)
                else:
                    self.run_mask_clip(rgb, ~valid_depth, world_xyz)
            if self.rerun:
                rr.set_time_sequence("frame", self.obs_count)
                if self.voxel_map.voxel_pcd._points is not None:
                    rr.log("Obstalce_map/pointcloud", rr.Points3D(self.voxel_map.voxel_pcd._points, colors=self.voxel_map.voxel_pcd._rgb / 255., radii=0.03))
                if self.voxel_map_localizer.voxel_pcd._points is not None:
                    rr.log("Semantic_memory/pointcloud", rr.Points3D(self.voxel_map_localizer.voxel_pcd._points, colors=self.voxel_map_localizer.voxel_pcd._rgb / 255., radii=0.03))
                rr.log("Obstalce_map/2D_obs_map", rr.Image(obs.int() * 127 + exp.int() * 127))
            else:
                cv2.imwrite(self.log + '/debug_' + str(self.obs_count) + '.jpg', np.asarray(obs.int() * 127 + exp.int() * 127))
            
            # self.sample_frontier()
            # from io import BytesIO
            # from PIL import Image
            # buf = BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # img = Image.open(buf)
            # img = np.array(img)
            # buf.close()
            # if self.rerun:
            #     rr.log('explore', rr.Image(img))

            if texts is None:
                continue

            for text in texts:
                if text is not None and text != '':
                    index, time_heuristics, alignments_heuristics, total_heuristics = self.space.sample_exploration([0, 0, 0], self.planner, self.voxel_map_localizer, text, debug = False)
                else:
                    index, time_heuristics, _, total_heuristics = self.space.sample_exploration([0, 0, 0], self.planner, None, None, debug = False)
                    alignments_heuristics = time_heuristics
                
                obstacles, explored = self.voxel_map.get_2d_map()
                if self.rerun:
                    plt.subplot(221)
                    plt.title('obs')
                    plt.imshow(obstacles * 0.5 + explored * 0.5)
                    plt.scatter(index[1], index[0], s = 20, c = 'r')
                    plt.subplot(222)
                    plt.title('time')
                    plt.imshow(time_heuristics.data)
                    # plt.scatter(index[1], index[0], s = 20, c = 'r')
                    plt.subplot(223)
                    plt.title('alignment')
                    plt.imshow(alignments_heuristics.data)
                    # plt.scatter(index[1], index[0], s = 20, c = 'r')
                    plt.subplot(224)
                    plt.title('total')
                    plt.imshow(total_heuristics.data)
                    # plt.scatter(index[1], index[0], s = 20, c = 'r')
                    
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    # Read the image into a numpy array
                    img = Image.open(buf)
                    img = np.array(img)
                    # Close the buffer
                    buf.close()
                    plt.clf()
                    if self.rerun:
                        rr.log(str(text) + '/explore', rr.Image(img))
                if text is None:
                    continue

                # localized_point = self.voxel_map_localizer.localize_AonB(text)
                localized_point, debug_text = self.voxel_map_localizer.localize_A_v2(text, debug = True)
                centroids, extends, similarity_max_list, points, obs_ids = self.voxel_map_localizer.find_clusters_for_A(text, return_obs_counts = True)
                # print(text, centroids, similarity_max_list)
                if self.rerun:
                    debug_text = '# The goal is to navigate to ' + text + '.\n' + debug_text
                    rr.log(text, rr.TextDocument(debug_text, media_type = rr.MediaType.MARKDOWN))
                    for idx, point in enumerate(points):
                        rr.log(text + "/pointcloud_" + str(idx), rr.Points3D(point, colors=torch.Tensor([1, 0, 0]).repeat(len(point), 1), radii=0.03))
                # print(self.voxel_map_localizer.find_alignment_over_model(text).cpu().max().item())
                obs_id = int(self.voxel_map_localizer.find_obs_id_for_A(text).detach().cpu().item())
                # if len(obs_ids) == 0:
                #     obs_id = 0
                # else:
                #     obs_id = max(obs_ids)
                print(self.voxel_map_localizer.check_existence(text, obs_id))

                rgb = np.load(log + '/rgb' + str(int(self.voxel_map_localizer.find_obs_id_for_A(text).detach().cpu().item())) + '.npy')
                if not self.rerun:
                    cv2.imwrite(log + '/debug_' + text + '.png', rgb[:, :, [2, 1, 0]])
                else:
                    rr.log(text + '/Memory_image', rr.Image(rgb))

                self.sample_navigation([0, 0, 0], localized_point)
                # plt.scatter(explore_target[1], explore_target[0], s = 15)
                buf = BytesIO()
                plt.title(str(self.voxel_map_localizer.find_alignment_over_model(text).cpu().max().item()) + ' ' + str(self.voxel_map_localizer.check_existence(text, obs_id)) + ' ' + text)
                plt.savefig(buf, format='png')
                buf.seek(0)
                # Read the image into a numpy array
                img = Image.open(buf)
                img = np.array(img)
                # Close the buffer
                buf.close()
                plt.clf()
                if self.rerun:
                    rr.log(text + '/localize', rr.Image(img))

                print('Points are observed from the ', obs_id, 'th image')

    def process_rgbd_images(self, rgb, depth, intrinsics, pose):
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.obs_count += 1
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
        
        with self.voxel_map_lock:
            self.voxel_map_localizer.voxel_pcd.clear_points(depth, torch.from_numpy(intrinsics), torch.from_numpy(pose))
            self.voxel_map.voxel_pcd.clear_points(depth, torch.from_numpy(intrinsics), torch.from_numpy(pose))

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
        if self.rerun:
            rr.set_time_sequence("frame", self.obs_count)
            rr.log('robot_pov', rr.Image(rgb.permute(1, 2, 0)))
            if self.voxel_map.voxel_pcd._points is not None:
                rr.log("Obstalce_map/pointcloud", rr.Points3D(self.voxel_map.voxel_pcd._points.detach().cpu(), \
                                                              colors=self.voxel_map.voxel_pcd._rgb.detach().cpu() / 255., radii=0.03))
            if self.voxel_map_localizer.voxel_pcd._points is not None:
                rr.log("Semantic_memory/pointcloud", rr.Points3D(self.voxel_map_localizer.voxel_pcd._points.detach().cpu(), \
                                                                 colors=self.voxel_map_localizer.voxel_pcd._rgb.detach().cpu() / 255., radii=0.03))
            rr.log("Obstalce_map/2D_obs_map", rr.Image(obs.int() * 127 + exp.int() * 127))
        else:
            cv2.imwrite(self.log + '/debug_' + str(self.obs_count) + '.jpg', obs.int() * 127 + exp.int() * 127)

@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(cfg):
    torch.manual_seed(1)
    imageProcessor = ImageProcessor(rerun = cfg.rerun)
    if not cfg.load_folder is None:
        print('Loading ', cfg.load_number, ' images from ', cfg.load_folder)
        # ['schoolbag', 'toy drill', 'red bowl', 'green bowl', 'purple cup', 'red cup', 'white rag', 'red apple', 'yellow ball', 'green rag', 'orange sofa', 'orange tape']
        imageProcessor.load(cfg.load_folder, cfg.load_number, texts = ['red bowl', 'green rag', 'green bowl', 'red cup', 'orange tape', 'toy drill', 'red packaging', 'purple body spray', 'red ball', 'wood carving', 'blue cup'])
    if not cfg.open_communication:
        imageProcessor.log = cfg.load_folder
        while True:
            imageProcessor.debug_text()
    else:
        try:  
            while True:
                imageProcessor.recv_text()
        except KeyboardInterrupt:
            if not imageProcessor.voxel_map_localizer.voxel_pcd._points is None:
                print('Stop streaming images and write memory data, might take a while, please wait')
                points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
                points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
                pcd = numpy_to_pcd(points, rgb / 255)
                o3d.io.write_point_cloud(imageProcessor.log + '/debug.pcd', pcd)
                print('finished')

if __name__ == "__main__":
    main()
