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
import os
import requests

def compute_tilt(camera_xyz, target_xyz):
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

def update_step(name, percentage):
    url = 'http://10.19.247.197:5000/update_step'
    data = {'name': name, 'percentage': percentage}
    requests.post(url, json=data)

def update_place_step(name, percentage):
    url = 'http://10.19.247.197:5000/update_place_step'
    data = {'name': name, 'percentage': percentage}
    requests.post(url, json=data)

def update_mode(mode):
    url = 'http://10.19.247.197:5000/update_mode'
    data = {'mode': mode}
    requests.post(url, json=data)

def update_task(task):
    url = 'http://10.19.247.197:5000/update_task'
    data = {'task': task}
    requests.post(url, json=data)

def clear_mode_display():
    url = 'http://10.19.247.197:5000/clear_mode_display'
    requests.post(url)

def clear_task_display():
    url = 'http://10.19.247.197:5000/clear_task_display'
    requests.post(url)

@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
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
    rate,
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

    rclpy.init()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pcd_filename = output_filename + "_" + formatted_datetime + ".pcd"
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()

    print("- Load parameters")
    parameters = get_parameters("src/robot_hw_python/configs/default.yaml")
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()

    print("- Start robot agent with data collection")
    demo = RobotAgent(
        robot, parameters, re=re, log_dir='debug' + '_' + formatted_datetime
    )

    if input_path:
        print('start reading from old pickle file')
        demo.voxel_map.read_from_pickle(filename=input_path)
        print('finish reading from old pickle file')

    def send_image():
        while True:
            if robot.manip.get_joint_positions()[1] <= 0.5:
                obs = robot.get_observation()
                demo.image_sender.send_images(obs)

    img_thread = threading.Thread(target=send_image)
    img_thread.daemon = True
    img_thread.start()

    while True:
        update_mode("Select Mode")
        mode = input('Select mode? E/N/S: ').strip().upper()
        if mode == 'S':
            break
        if mode == 'E':
            update_mode("Exploration Mode Activated")
            robot.switch_to_navigation_mode()
            demo.run_exploration(
                rate,
                manual_wait,
                explore_iter=2,
                task_goal=object_to_find,
                go_home_at_end=navigate_home,
                visualize=show_intermediate_maps,
            )
            clear_mode_display()
            if not os.path.exists(demo.log_dir):
                os.mkdir(demo.log_dir)
            pc_xyz, pc_rgb = demo.voxel_map.get_xyz_rgb()
            torch.save(demo.voxel_map.voxel_pcd, demo.log_dir + '/memory.pt')
            if len(output_pcd_filename) > 0:
                print(f"Write pcd to {output_pcd_filename}...")
                pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
                open3d.io.write_point_cloud(demo.log_dir + '/' + output_pcd_filename, pcd)
            if len(output_pkl_filename) > 0:
                print(f"Write pkl to {output_pkl_filename}...")
                demo.voxel_map.write_to_pickle(demo.log_dir + '/' + output_pkl_filename)
        elif mode == 'N':
            update_mode("Navigation Mode Activated")
            while True:
                task = input('Pick or Place? ').strip().lower()
                if task == 'pick':
                    update_task("Pickup Mode Activated")
                    text = input('Enter object name: ').strip()
                    update_step("Object Specified", 5)
                    clear_task_display()
                    if input('You want to run manipulation: y/n').strip().lower() == 'n':
                        break
                    update_step("Manipulation Begins", 20)
                    theta = -0.6
                    demo.manipulate(text, theta)
                elif task == 'place':
                    update_task("Place Mode Activated")
                    text = input('Enter receptacle name: ').strip()
                    update_place_step("Receptacle Specified", 5)
                    clear_task_display()
                    if input('You want to run placing: y/n').strip().lower() == 'n':
                        break
                    camera_xyz = robot.head.get_pose()[:3, 3]
                    theta = -0.6
                    update_place_step("Place Operation Begins", 20)
                    demo.place(text, theta)
                else:
                    print("Invalid task. Please choose 'Pick' or 'Place'.")
            if input('Do you want to select a different mode? y/n: ').strip().lower() == 'n':
                break

if __name__ == "__main__":
    main()