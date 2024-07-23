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
import torch
from PIL import Image

# Mapping and perception
from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask import RobotAgentMDP as RobotAgent

# Chat and UI tools
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from home_robot_hw.remote import StretchClient

import cv2
import threading

import os

def compute_tilt(camera_xyz, target_xyz):
    '''
        a util function for computing robot head tilts so the robot can look at the target object after navigation
        - camera_xyz: estimated (x, y, z) coordinates of camera
        - target_xyz: estimated (x, y, z) coordinates of the target object
    '''
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

@click.command()
@click.option("--ip", default='100.108.67.79', type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--re", default=1, type=int)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
def main(
    ip,
    manual_wait,
    output_filename,
    navigate_home: bool = False,
    show_intermediate_maps: bool = False,
    explore_iter: int = 10,
    re: int = 1,
    input_path: str = None,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        show_intermediate_maps(bool): show maps as we explore
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pcd_filename = output_filename + "_" + formatted_datetime + ".pcd"
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()
    # robot.nav.navigate_to([0, 0, 0])

    print("- Load parameters")
    parameters = get_parameters("src/robot_hw_python/configs/default.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.nav.set_velocity(v = 15., w = 8.)
    robot.move_to_nav_posture()

    print("- Start robot agent with data collection")
    demo = RobotAgent(
        robot, parameters, ip = ip, re = re, log_dir = 'debug' + '_' + formatted_datetime
    )

    if input_path:
        print('start reading from old pickle file')
        demo.voxel_map.read_from_pickle(filename = input_path)
        print('finish reading from old pickle file')

    while True:
        mode = input('select mode? E/N/S')
        if mode == 'S':
            break
        if mode == 'E':
            robot.switch_to_navigation_mode()
            for epoch in range(10):
                print('\n', 'Exploration epoch ', epoch, '\n')
                if not demo.run_exploration():
                    print('Exploration failed! Quitting!')
                    break
        else:
            robot.move_to_nav_posture()
            robot.switch_to_navigation_mode()
            text = input('Enter object name: ')
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure!')
                continue
            robot.switch_to_navigation_mode()
            xyt = robot.nav.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.nav.navigate_to(xyt)
            cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])

            if input('You want to run manipulation: y/n') == 'n':
                continue
            camera_xyz = robot.head.get_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.manipulate(text, theta)
            
            robot.switch_to_navigation_mode()
            if input('You want to run placing: y/n') == 'n':
                continue
            text = input('Enter receptacle name: ')
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure')
                continue
            robot.switch_to_navigation_mode()
            xyt = robot.nav.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.nav.navigate_to(xyt)
            cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])
        
            if input('You want to run placing: y/n') == 'n':
                continue
            camera_xyz = robot.head.get_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.place(text, theta)


if __name__ == "__main__":
    main()
