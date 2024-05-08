# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rclpy
import torch
from PIL import Image

# Mapping and perception
from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask import RobotAgentManip as RobotAgent

# Chat and UI tools
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from robot_hw_python.remote import StretchClient

import cv2
import threading

def compute_tilt(camera_xyz, target_xyz):
    '''
        a util function for computing robot head tilts so the robot can look at the target object after navigation
        - camera_xyz: estimated (x, y, z) coordinates of camera
        - target_xyz: estimated (x, y, z) coordinates of the target object
    '''
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--navigate-home", default=False, is_flag=True)
@click.option("--force-explore", default=False, is_flag=True)
def main(
    rate,
    # visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = False,
    show_intermediate_maps: bool = False,
    explore_iter: int = 10,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        show_intermediate_maps(bool): show maps as we explore
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    rclpy.init()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pcd_filename = output_filename + "_" + formatted_datetime + ".pcd"
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()
    # robot.nav.navigate_to([0, 0, 0])

    print("- Load parameters")
    parameters = get_parameters("src/home_robot_hw/configs/default.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()

    print("- Start robot agent with data collection")
    demo = RobotAgent(
        robot, parameters
    )

    # def send_image():
    #     while True:
    #         if robot.nav._is_enabled:
    #             obs = robot.get_observation()
    #             demo.image_sender.send_images(obs)

    # img_thread = threading.Thread(target=send_image)
    # img_thread.daemon = True
    # img_thread.start()

    
    while True:
        # mode = input('select mode? E/N/S')
        mode = 'N'
        if mode == 'S':
            break
        if mode == 'E':
            for _ in range(1):
                demo.look_around()
                demo.run_exploration(
                    rate,
                    manual_wait,
                    explore_iter=15,
                    task_goal=object_to_find,
                    go_home_at_end=navigate_home,
                    visualize=show_intermediate_maps,
                )
        else:
            # text = input('Enter object name: ')
            # point = demo.image_sender.query_text(text)
            # demo.navigate(point)
            # cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])
            # robot.switch_to_navigation_mode()
            # xyt = robot.nav.get_base_pose()
            # xyt[2] = xyt[2] + np.pi / 2
            # robot.nav.navigate_to(xyt)

            # if input('You want to run manipulation: y/n') == 'n':
            #     continue
            text = 'white bottle'
            camera_xyz = robot.head.get_pose()[:3, 3]
            # theta = compute_tilt(camera_xyz, point)
            theta = -0.6
            demo.manipulate(text, theta)
            robot.switch_to_navigation_mode()
        
            # if input('You want to run placing: y/n') == 'n':
            #     continue
            # text = input('Enter receptacle name: ')
            # point = demo.image_sender.query_text(text)
            # demo.navigate(point)
            # cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])
            # robot.switch_to_navigation_mode()
            # xyt = robot.nav.get_base_pose()
            # xyt[2] = xyt[2] + np.pi / 2
            # robot.nav.navigate_to(xyt)
        
            if input('You want to run placing: y/n') == 'n':
                continue
            text = 'table'
            camera_xyz = robot.head.get_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.place(text, theta)
            robot.switch_to_navigation_mode()
            xyt = robot.nav.get_base_pose()
            xyt[2] = xyt[2] - np.pi / 2
            robot.nav.navigate_to(xyt)

    pc_xyz, pc_rgb = demo.voxel_map.get_xyz_rgb()
    torch.save(demo.voxel_map.voxel_pcd, 'memory_chris.pt')
    if len(output_pcd_filename) > 0:
        print(f"Write pcd to {output_pcd_filename}...")
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(output_pcd_filename, pcd)
    if len(output_pkl_filename) > 0:
        print(f"Write pkl to {output_pkl_filename}...")
        demo.voxel_map.write_to_pickle(output_pkl_filename)


if __name__ == "__main__":
    main()
