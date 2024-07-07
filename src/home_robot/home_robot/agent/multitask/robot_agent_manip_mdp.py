# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import datetime
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from loguru import logger

from home_robot.agent.multitask import Parameters
from home_robot.core.robot import RobotClient
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

import zmq

from matplotlib import pyplot as plt

from home_robot.agent.multitask.ok_robot_hw.global_parameters import *
from home_robot.agent.multitask.ok_robot_hw.robot import HelloRobot as Manipulation_Wrapper
from home_robot.agent.multitask.ok_robot_hw.camera import RealSenseCamera
from home_robot.agent.multitask.ok_robot_hw.utils.grasper_utils import pickup, move_to_point, capture_and_process_image
from home_robot.agent.multitask.ok_robot_hw.utils.communication_utils import send_array, recv_array

import cv2

class RobotAgentMDP:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail = False

    def __init__(
        self,
        robot: RobotClient,
        parameters: Dict[str, Any],
        manip_port: int = 5557,
        re: int = 1,
        log_dir: str = 'debug'
    ):
        print('------------------------YOU ARE NOW RUNNING PEIQI CODES V3-----------------')
        self.re = re
        self.log_dir = log_dir
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        end_link = "link_straight_gripper"
        if re == 1:
            stretch_gripper_max = 0.3
        else:
            stretch_gripper_max = 0.64
        self.transform_node = end_link
        self.manip_wrapper = Manipulation_Wrapper(self.robot, stretch_gripper_max = stretch_gripper_max, end_link = end_link)
        self.robot.move_to_nav_posture()

        self.normalize_embeddings = True
        self.pos_err_threshold = 0.2
        self.rot_err_threshold = 0.4
        self.obs_count = 0
        self.obs_history = []
        self.guarantee_instance_is_reachable = (
            parameters.guarantee_instance_is_reachable
        )

        self.image_sender = ImageSender()

        self.look_around_times = []
        self.execute_times = []
        
        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    def get_navigation_space(self) -> ConfigurationSpace:
        """Returns reference to the navigation space."""
        return self.space

    def look_around(self, visualize: bool = False):
        logger.info("Look around to check")
        for pan in [0.5, -0.5, -1.5]:
            for tilt in [-0.55]:
                self.robot.head.set_pan_tilt(pan, tilt)
                time.sleep(0.3)
                self.update()

                if visualize:
                    self.voxel_map.show(
                        orig=np.zeros(3),
                        xyt=self.robot.get_base_pose(),
                        footprint=self.robot.get_robot_model().get_footprint(),
                    )

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = self.robot.get_observation()
        # self.image_sender.send_images(obs)
        self.obs_history.append(obs)
        self.obs_count += 1
        self.image_sender.send_images(obs)

    def execute_action(
        self,
        text: str,
    ):
        start_time = time.time()

        self.robot.head.look_front()
        self.look_around()
        self.robot.head.look_front()
        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_pose()
        print("       Start:", start)
        res = self.image_sender.query_text(text, start)  

        look_around_finish = time.time()
        look_around_take = look_around_finish - start_time
        print('Looking around takes ', look_around_take, ' seconds.')
        self.look_around_times.append(look_around_take)
        print(self.look_around_times)
        print(sum(self.look_around_times) / len(self.look_around_times))

        if len(res) > 0:
            print("Plan successful!")
            if len(res) > 2 and np.isnan(res[-2]).all():
                self.robot.execute_trajectory(
                    res[:-2],
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                )

                execution_finish = time.time()
                execution_take = execution_finish - look_around_finish
                print('Executing action takes ', execution_take, ' seconds.')
                self.execute_times.append(execution_take)
                print(self.execute_times)
                print(sum(self.execute_times) / len(self.execute_times))

                return True, res[-1]
            else:
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                )

                execution_finish = time.time()
                execution_take = execution_finish - look_around_finish
                print('Executing action takes ', execution_take, ' seconds.')
                self.execute_times.append(execution_take)
                print(self.execute_times)
                print(sum(self.execute_times) / len(self.execute_times))

                return False, None
        else:
            print("Failed. Try again!")
            return None, None

    def run_exploration(
        self
    ):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def navigate(self, text, max_step = 50):
        finished = False
        step = 0
        end_point = None
        while not finished and step < max_step:
            step += 1
            finished, end_point = self.execute_action(text) 
            if finished is None:
                print("Navigation failed! The path might be blocked!")
                return None
        return end_point

    def place(self, text, init_tilt = INIT_HEAD_TILT, base_node = TOP_CAMERA_NODE):
        '''
            An API for running placing. By calling this API, human will ask the robot to place whatever it holds
            onto objects specified by text queries A
            - hello_robot: a wrapper for home-robot StretchClient controller
            - socoket: we use this to communicate with workstation to get estimated gripper pose
            - text: queries specifying target object
            - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
            - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        '''
        self.robot.switch_to_manipulation_mode()
        self.robot.head.look_at_ee()
        self.manip_wrapper.move_to_position(
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt)
        camera = RealSenseCamera(self.robot)

        time.sleep(2)
        rotation, translation = capture_and_process_image(
            camera = camera,
            mode = 'place',
            obj = text,
            socket = self.image_sender.manip_socket, 
            hello_robot = self.manip_wrapper)

        if rotation is None:
            return False
        print(rotation)

        # lift arm to the top before the robot extends the arm, prepare the pre-placing gripper pose
        self.manip_wrapper.move_to_position(lift_pos=1.05)
        time.sleep(1)
        self.manip_wrapper.move_to_position(wrist_yaw=0,
                                 wrist_pitch=0)
        time.sleep(1)

        # Placing the object
        move_to_point(self.manip_wrapper, translation, base_node, self.transform_node, move_mode=0)
        time.sleep(1.5)
        self.manip_wrapper.move_to_position(gripper_pos=1)

        # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
        self.manip_wrapper.move_to_position(lift_pos = self.manip_wrapper.robot.manip.get_joint_positions()[1] + 0.3)
        self.manip_wrapper.move_to_position(wrist_roll = 2.5)
        time.sleep(1.5)
        self.manip_wrapper.move_to_position(wrist_roll = -2.5)
        time.sleep(1.5)

        # Wait for some time and shrink the arm back
        self.manip_wrapper.move_to_position(gripper_pos=1, 
                                lift_pos = 1.05,
                                arm_pos = 0)
        time.sleep(3)
        self.manip_wrapper.move_to_position(wrist_pitch=-1.57)

        # Shift the base back to the original point as we are certain that orginal point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(base_trans = -self.manip_wrapper.robot.manip.get_joint_positions()[0])
        return True

    def manipulate(self, text, init_tilt = INIT_HEAD_TILT, base_node = TOP_CAMERA_NODE):
        '''
            An API for running manipulation. By calling this API, human will ask the robot to pick up objects
            specified by text queries A
            - hello_robot: a wrapper for home-robot StretchClient controller
            - socoket: we use this to communicate with workstation to get estimated gripper pose
            - text: queries specifying target object
            - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
            - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        '''

        self.robot.switch_to_manipulation_mode()
        self.robot.head.look_at_ee()

        gripper_pos = 1

        self.manip_wrapper.move_to_position(arm_pos=INIT_ARM_POS,
                                head_pan=INIT_HEAD_PAN,
                                head_tilt=init_tilt,
                                gripper_pos = gripper_pos,
                                lift_pos=INIT_LIFT_POS,
                                wrist_pitch = INIT_WRIST_PITCH,
                                wrist_roll = INIT_WRIST_ROLL,
                                wrist_yaw = INIT_WRIST_YAW)

        camera = RealSenseCamera(self.robot)

        rotation, translation, depth, width = capture_and_process_image(
            camera = camera,
            mode = 'pick',
            obj = text,
            socket = self.image_sender.manip_socket, 
            hello_robot = self.manip_wrapper)

        print('Predicted width:', width)
    
        if rotation is None:
            return False
        
        if width < 0.045 and self.re == 3:
            gripper_width = 0.45
        if width < 0.075 and self.re == 3: 
            gripper_width = 0.6
        else:
            gripper_width = 1

        if input('Do you want to do this manipulation? Y or N ') != 'N':
            pickup(self.manip_wrapper, rotation, translation, base_node, self.transform_node, gripper_depth = depth, gripper_width = gripper_width)
    
        # Shift the base back to the original point as we are certain that orginal point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(base_trans = -self.manip_wrapper.robot.manip.get_joint_positions()[0])

        return True

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

class ImageSender:
    def __init__(self, 
        stop_and_photo = False, 
        ip = '172.24.71.227', 
        image_port = 5555,
        text_port = 5556,
        manip_port = 5557,
        color_name = "/camera/color",
        depth_name = "/camera/aligned_depth_to_color",
        camera_name = "/camera_pose",
        slop_time_seconds = 0.05,
        queue_size = 100,
    ):
        context = zmq.Context()
        self.img_socket = context.socket(zmq.REQ)
        self.img_socket.connect('tcp://' + str(ip) + ':' + str(image_port))
        self.text_socket = context.socket(zmq.REQ)
        self.text_socket.connect('tcp://' + str(ip) + ':' + str(text_port))
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect('tcp://' + str(ip) + ':' + str(manip_port))

    def query_text(self, text, start):
        self.text_socket.send_string(text)
        self.text_socket.recv_string()
        send_array(self.text_socket, start)
        return recv_array(self.text_socket)
        
    def send_images(self, obs):
        rgb = obs.rgb
        depth = obs.depth
        camer_K = obs.camera_K
        camera_pose = obs.camera_pose
        # data = np.concatenate((depth.shape, rgb.flatten(), depth.flatten(), camer_K.flatten(), camera_pose.flatten()))
        # send_array(self.img_socket, data)
        send_everything(self.img_socket, rgb, depth, camer_K, camera_pose)
        # self.img_socket.recv_string()
