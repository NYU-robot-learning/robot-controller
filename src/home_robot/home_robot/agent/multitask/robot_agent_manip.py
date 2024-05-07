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
)

import zmq

from matplotlib import pyplot as plt

from home_robot.agent.multitask.ok_robot_hw.global_parameters import *
from home_robot.agent.multitask.ok_robot_hw.robot import HelloRobot as Manipulation_Wrapper
from home_robot.agent.multitask.ok_robot_hw.args import get_args
from home_robot.agent.multitask.ok_robot_hw.camera import RealSenseCamera
from home_robot.agent.multitask.ok_robot_hw.utils.grasper_utils import pickup, move_to_point, capture_and_process_image
from home_robot.agent.multitask.ok_robot_hw.utils.communication_utils import send_array, recv_array

class RobotAgentManip:
    """Basic demo code. Collects everything that we need to make this work."""

    _retry_on_fail = False

    def __init__(
        self,
        robot: RobotClient,
        parameters: Dict[str, Any],
        voxel_map: Optional[SparseVoxelMap] = None,
        manip_port: int = 5557,
    ):
        print('------------------------YOU ARE NOW RUNNING PEIQI CODES V3-----------------')
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.manip_wrapper = Manipulation_Wrapper(self.robot, end_link = GRIPPER_MID_NODE)
        self.robot.move_to_nav_posture()

        self.normalize_embeddings = True
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]
        self.obs_count = 0
        self.obs_history = []
        self.guarantee_instance_is_reachable = (
            parameters.guarantee_instance_is_reachable
        )

        self.image_sender = ImageSender()

        # Expanding frontier - how close to frontier are we allowed to go?
        self.default_expand_frontier_size = parameters["default_expand_frontier_size"]

        if voxel_map is not None:
            self.voxel_map = voxel_map
        else:
            self.voxel_map = SparseVoxelMap(
                resolution=parameters["voxel_size"],
                local_radius=parameters["local_radius"],
                obs_min_height=parameters["obs_min_height"],
                obs_max_height=parameters["obs_max_height"],
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

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot.get_robot_model(),
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Create a simple motion planner
        self.planner = RRTConnect(self.space, self.space.is_valid)
        if parameters["motion_planner"]["shortcut_plans"]:
            self.planner = Shortcut(
                self.planner, parameters["motion_planner"]["shortcut_iter"]
            )
        if parameters["motion_planner"]["simplify_plans"]:
            self.planner = SimplifyXYT(
                self.planner, min_step=0.05, max_step=1.0, num_steps=8
            )

        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    def get_navigation_space(self) -> ConfigurationSpace:
        """Returns reference to the navigation space."""
        return self.space

    def look_around(self, visualize: bool = False):
        logger.info("Look around to check")
        time.sleep(1)
        for pan in [0.25, 0.75, -0.25, -0.75, -1.25, -1.75, -2.25]:
            for tilt in [-0.3, -0.45, -0.6]:
                self.robot.head.set_pan_tilt(pan, tilt)
                time.sleep(0.5)
                self.update()

            if visualize:
                self.voxel_map.show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

    def rotate_in_place(self, steps: int = 6, visualize: bool = False) -> bool:
        """Simple helper function to make the robot rotate in place. Do a 360 degree turn to get some observations (this helps debug the robot and create a nice map).

        Returns:
            executed(bool): false if we did not actually do any rotations"""
        logger.info("Rotate in place")
        # self.robot.move_to_nav_posture()
        if steps <= 0:
            return False
        step_size = 2 * np.pi / steps
        i = 0
        while i < steps:
            print('-' * 20, 'step', i, '-'*20)
            self.robot.navigate_to([0, 0, step_size], relative=True, blocking=True)
            # TODO remove debug code
            # print(i, self.robot.get_base_pose())
            self.robot.head.set_pan_tilt(pan = 0, tilt = np.random.uniform(-0.7, -0.4))
            self.update()
            if self.robot.last_motion_failed():
                # We have a problem!
                self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                i = 0
            else:
                i += 1

            if visualize:
                self.voxel_map.show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

        return True

    def get_command(self):
        if (
            "command" in self.parameters.data.keys()
        ):  # TODO: this was breaking. Should this be a class method
            return self.parameters["command"]
        else:
            return self.ask("please type any task you want the robot to do: ")

    def __del__(self):
        """Clean up at the end if possible"""
        print("... Done.")

    def update(self):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = self.robot.get_observation()
        # self.image_sender.send_images(obs)
        self.obs_history.append(obs)
        self.obs_count += 1
        self.voxel_map.add_obs(obs)    

        # Add observation - helper function will unpack it
        self.voxel_map.get_2d_map(debug=True)
        if not os.path.exists('debug'):
            os.mkdir('debug')
        plt.savefig('debug/debug' + str(self.obs_count) + '.jpg')

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial."""
        print("Go back to (0, 0, 0) to finish...")
        print("- change posture and switch to navigation mode")
        self.robot.move_to_nav_posture()
        # self.robot.head.look_close(blocking=False)
        self.robot.switch_to_navigation_mode()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        print(
            f"- Current pose is valid: {self.space.is_valid(self.robot.get_base_pose())}"
        )
        print(f"-   start pose is valid: {self.space.is_valid(start)}")
        print(f"-    goal pose is valid: {self.space.is_valid(goal)}")
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            print("Done!")
        else:
            print("Can't go home; planning failed!")

    def run_exploration(
        self,
        rate: int = 10,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
        go_to_start_pose: bool = False,
        show_goal: bool = False,
    ):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.robot.move_to_nav_posture()

        if go_to_start_pose:
            print("Go to (0, 0, 0) to start with...")
            self.robot.navigate_to([0, 0, 0])

        all_starts = []
        all_goals = []

        # Explore some number of times
        no_success_explore = True
        for i in range(explore_iter):
            self.robot.move_to_nav_posture()
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start, verbose=True)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")

                # TODO: debug here -- why start is not valid?
                self.update()
                # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
                print(f"robot base pose: {self.robot.get_base_pose()}")

                print("--- STARTS ---")
                for a_start, a_goal in zip(all_starts, all_goals):
                    print(
                        "start =",
                        a_start,
                        self.space.is_valid(a_start),
                        "goal =",
                        a_goal,
                        self.space.is_valid(a_goal),
                    )

                self.robot.navigate_to([-0.1, 0, 0], relative=True)
                continue

            print("       Start:", start)
            # sample a goal
            res = plan_to_frontier(
                start,
                self.planner,
                self.space,
                self.voxel_map,
                try_to_plan_iter=try_to_plan_iter,
                visualize=False,  # visualize,
                expand_frontier_size=self.default_expand_frontier_size,
            )

            # if it succeeds, execute a trajectory to this position
            if res.success:
                no_success_explore = False
                print("Plan successful!")
                for i, pt in enumerate(res.trajectory):
                    print(i, pt.state)
                all_starts.append(start)
                all_goals.append(res.trajectory[-1].state)
                if visualize:
                    print("Showing goal location:")
                    robot_center = np.zeros(3)
                    robot_center[:2] = self.robot.get_base_pose()[:2]
                    self.voxel_map.show(
                        orig=robot_center,
                        xyt=res.trajectory[-1].state,
                        footprint=self.robot.get_robot_model().get_footprint(),
                    )
                if not dry_run:
                    self.robot.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )
            else:
                if self._retry_on_fail:
                    print("Failed. Try again!")
                    continue
                else:
                    print("Failed. Quitting!")
                    break

            if self.robot.last_motion_failed():
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("ROBOT IS STUCK! Move back!")

                # help with debug TODO: remove
                self.update()
                # self.save_svm(".")
                print(f"robot base pose: {self.robot.get_base_pose()}")

                r = np.random.randint(3)
                if r == 0:
                    self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                elif r == 1:
                    self.robot.navigate_to(
                        [0, 0, np.pi / 4], relative=True, blocking=True
                    )
                elif r == 2:
                    self.robot.navigate_to(
                        [0, 0, -np.pi / 4], relative=True, blocking=True
                    )
            self.look_around()
            # Append latest observations
            # self.update()
            # self.rotate_in_place()
            # self.save_svm("", filename=f"debug_svm_{i:03d}.pkl")
            if visualize:
                # After doing everything - show where we will move to
                robot_center = np.zeros(3)
                robot_center[:2] = self.robot.get_base_pose()[:2]
                self.voxel_map.show(
                    orig=robot_center,
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                )

            if manual_wait:
                input("... press enter ...")

        # if it fails to find any frontier in the given iteration, simply quit in sim
        # if no_success_explore:
        #     print("The robot did not explore at all, force quit in sim")
        #     self.robot.force_quit = True

        # if go_home_at_end:
        #     # Finally - plan back to (0,0,0)
        #     print("Go back to (0, 0, 0) to finish...")
        #     start = self.robot.get_base_pose()
        #     goal = np.array([0, 0, 0])
        #     res = self.planner.plan(start, goal)
        #     # if it fails, skip; else, execute a trajectory to this position
        #     if res.success:
        #         print("Full plan to home:")
        #         for i, pt in enumerate(res.trajectory):
        #             print("-", i, pt.state)
        #         if not dry_run:
        #             self.robot.execute_trajectory([pt.state for pt in res.trajectory])
        #     else:
        #         print("WARNING: planning to home failed!")

    def navigate(self, point, max_tries = 1000, radius_m = 0.7, visualize = False, verbose = True):
        print(point)
        self.robot.switch_to_navigation_mode()
        self.robot.head.look_front()
        target_grid = self.voxel_map.xy_to_grid_coords(point[:2]).int()
        obstacles, explored = self.voxel_map.get_2d_map()
        point_mask = torch.zeros_like(explored)
        point_mask[target_grid[0]: target_grid[0] + 2, target_grid[1]: target_grid[1] + 2] = True
        res = None
        try_count = 0
        for goal in self.space.sample_near_mask(point_mask, radius_m=radius_m):
            start = self.robot.get_base_pose()
            goal = goal.cpu().numpy()
            print("       Start:", start)
            print("Sampled Goal:", goal)
            start_is_valid = self.space.is_valid(start, verbose=True)
            goal_is_valid = self.space.is_valid(goal, verbose=False)
            if verbose:
                print("Start is valid:", start_is_valid)
                print(" Goal is valid:", goal_is_valid)
            if not goal_is_valid:
                print(" -> resample goal.")
                continue

            res = self.planner.plan(start, goal, verbose=False)
            if verbose:
                print("Found plan:", res.success)
            try_count += 1
            if res.success or try_count > max_tries:
                break

        all_starts, all_goals = [], []
        if res and res.success:
            for i, pt in enumerate(res.trajectory):
                print(i, pt.state)
                all_starts.append(start)
                all_goals.append(res.trajectory[-1].state)
            if visualize:
                print("Showing goal location:")
                robot_center = np.zeros(3)
                robot_center[:2] = self.robot.get_base_pose()[:2]
                self.voxel_map.show(
                    orig=robot_center,
                    xyt=res.trajectory[-1].state,
                    footprint=self.robot.get_robot_model().get_footprint(),
                )
            self.robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
            )
            return True
        else:
            print('Navigation Failure!')
            return false
        # self.look_ahead()

    def place(self, text, init_tilt = INIT_HEAD_TILT, transform_node = GRIPPER_MID_NODE, base_node = TOP_CAMERA_NODE):
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
        rotation, translation, _ = capture_and_process_image(
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
        time.sleep(2)

        # Placing the object
        move_to_point(self.manip_wrapper, translation, base_node, transform_node, move_mode=0)
        self.manip_wrapper.move_to_position(gripper_pos=1)

        # Lift the arm a little bit, and rotate the wrist roll of the robot in case the object attached on the gripper
        self.manip_wrapper.move_to_position(lift_pos = self.manip_wrapper.robot.manip.get_joint_positions()[1] + 0.3)
        self.manip_wrapper.move_to_position(wrist_roll = 3)
        time.sleep(1)
        self.manip_wrapper.move_to_position(wrist_roll = -3)

        # Wait for some time and shrink the arm back
        time.sleep(4)
        self.manip_wrapper.move_to_position(gripper_pos=1, 
                                lift_pos = 1.05,
                                arm_pos = 0)
        time.sleep(4)
        self.manip_wrapper.move_to_position(wrist_pitch=-1.57)
        time.sleep(1)

        # Shift the base back to the original point as we are certain that orginal point is navigable in navigation obstacle map
        self.manip_wrapper.move_to_position(base_trans = -self.manip_wrapper.robot.manip.get_joint_positions()[0])
        return True

    def manipulate(self, text, init_tilt = INIT_HEAD_TILT, transform_node = GRIPPER_MID_NODE, base_node = TOP_CAMERA_NODE):
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

        rotation, translation, depth = capture_and_process_image(
            camera = camera,
            mode = 'pick',
            obj = text,
            socket = self.image_sender.manip_socket, 
            hello_robot = self.manip_wrapper)
    
        if rotation is None:
            return False
        
        if input('Do you want to do this manipulation? Y or N ') != 'N':
            pickup(self.manip_wrapper, rotation, translation, base_node, transform_node, gripper_depth = depth)
    
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

class ImageSender:
    def __init__(self, 
        stop_and_photo = False, 
        ip = '100.107.224.62', 
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

    def query_text(self, text):
        self.text_socket.send_string(text)
        return recv_array(self.text_socket)
        
    def send_images(self, obs):
        rgb = obs.rgb
        depth = obs.depth
        camer_K = obs.camera_K
        camera_pose = obs.camera_pose
        data = np.concatenate((depth.shape, rgb.flatten(), depth.flatten(), camer_K.flatten(), camera_pose.flatten()))
        send_array(self.img_socket, data)
        self.img_socket.recv_string()