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

import datetime

import threading

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
        device = 'cuda',
        min_depth = 0.25,
        max_depth = 2.0,
        img_port = 5555,
        text_port = 5556,
        pcd_path: str = None
    ):
        current_datetime = datetime.datetime.now()
        self.log = 'debug_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
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

        self.img_socket = load_socket(img_port)
        self.text_socket = load_socket(text_port)

        self.voxel_map_lock = threading.Lock()  # Create a lock for synchronizing access to `self.voxel_map_localizer`

        self.img_thread = threading.Thread(target=self._recv_image)
        self.img_thread.daemon = True
        self.img_thread.start()

        # self.text_thread = threading.Thread(target=self._recv_text)
        # self.text_thread.daemon = True
        # self.text_thread.start()

    def create_vision_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
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
        self.voxel_map_localizer = VoxelMapLocalizer(device = 'cpu')
        if self.pcd_path is not None:
            print('Loading old semantic memory')
            self.voxel_map_localizer.voxel_pcd = torch.load(self.pcd_path)
            print('Finish loading old semantic memory')


    def recv_text(self):
        text = self.text_socket.recv_string()
        with self.voxel_map_lock:
            point = self.voxel_map_localizer.localize_AonB(text)
        send_array(self.text_socket, point)

    def _recv_image(self):
        while True:
            data = recv_array(self.img_socket)
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
            for box in bounding_boxes:
                tl_x, tl_y, br_x, br_y = box
                crops.append(self.clip_preprocess(transforms.ToPILImage()(rgb[:, max(int(tl_y), 0): min(int(br_y), rgb.shape[1]), max(int(tl_x), 0): min(int(br_x), rgb.shape[2])])))
            features = self.clip_model.encode_image(torch.stack(crops, dim = 0).to(self.device))
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
    
    def add_to_voxel_pcd(self, valid_xyz, feature, valid_rgb, weights = None, threshold = 0.85):
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

    def process_rgbd_images(self, data):
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

        # with self.voxel_map_lock:
        #     self.voxel_map_localizer.voxel_pcd.clear_points(depth, intrinsics, pose)

        if self.owl:
            self.run_owl_sam_clip(rgb, torch.logical_or(depth > self.max_depth, depth < self.min_depth), world_xyz)
        else:
            self.run_mask_clip(rgb, torch.logical_or(depth > self.max_depth, depth < self.min_depth), world_xyz)

if __name__ == "__main__":
    # imageProcessor = ImageProcessor(pcd_path = 'memory.pt')
    imageProcessor = ImageProcessor(pcd_path = None)   
    try:  
        while True:
            imageProcessor.recv_text()
    except KeyboardInterrupt:
        if not imageProcessor.voxel_map_localizer.voxel_pcd._points is None:
            print('Stop streaming images and write memory data, might take a while, please wait')
            torch.save(imageProcessor.voxel_map_localizer.voxel_pcd, 'memory.pt')
            points, _, _, rgb = imageProcessor.voxel_map_localizer.voxel_pcd.get_pointcloud()
            points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
            pcd = numpy_to_pcd(points, rgb / 255)
            o3d.io.write_point_cloud(imageProcessor.log + 'debug.pcd', pcd)
            print('finished')
