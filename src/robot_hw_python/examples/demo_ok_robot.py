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

from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask import RobotAgentManip as RobotAgent

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
    url = 'http://localhost:5000/update_step'
    data = {'name': name, 'percentage': percentage}
    requests.post(url, json=data)

def update_place_step(name, percentage):
    url = 'http://localhost:5000/update_place_step'
    data = {'name': name, 'percentage': percentage}
    requests.post(url, json=data)

def update_mode(mode):
    url = 'http://localhost:5000/update_mode'
    data = {'mode': mode}
    requests.post(url, json=data)

def update_task(task):
    url = 'http://localhost:5000/update_task'
    data = {'task': task}
    requests.post(url, json=data)

def clear_mode_display():
    url = 'http://localhost:5000/clear_mode_display'
    requests.post(url)

def clear_task_display():
    url = 'http://localhost:5000/clear_task_display'
    requests.post(url)

import threading

def update_mode_with_clear(mode):
    update_mode(mode)
    threading.Timer(5, clear_mode_display).start()

def update_task_with_clear(task):
    update_task(task)
    threading.Timer(5, clear_task_display).start()

def clear_task_display_after_delay(delay):
    threading.Timer(delay, clear_task_display).start()

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
            update_mode_with_clear("Stopped")
            break
        if mode == 'E':
            update_mode_with_clear("Exploration Mode Activated")
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
        elif mode == 'N':
            update_mode_with_clear("Navigation Mode Activated")
            while True:
                task = input('Pick or Place? ').strip().lower()
                if task == 'pick':
                    update_task("Pickup Mode Activated")
                    clear_task_display_after_delay(5)  # Delay before clearing text
                    text = input('Enter object name: ').strip()
                    update_step("Object Specified", 5)
                    if input('You want to run manipulation: y/n').strip().lower() == 'n':
                        break
                    update_step("Manipulation Begins", 20)
                    theta = -0.6
                    demo.manipulate(text, theta)
                elif task == 'place':
                    update_task("Place Mode Activated")
                    clear_task_display_after_delay(5)  # Delay before clearing text
                    text = input('Enter receptacle name: ').strip()
                    update_place_step("Receptacle Specified", 5)
                    if input('You want to run placing: y/n').strip().lower() == 'n':
                        break
                    camera_xyz = robot.head.get_pose()[:3, 3]
                    theta = -0.6
                    update_place_step("Place Operation Begins", 20)
                    demo.place(text, theta)
                else:
                    print("Invalid task. Please choose 'Pick' or 'Place'.")
            if input('Do you want to select a different mode? y/n: ').strip().lower() == 'n':
                update_mode_with_clear("Stopped")
                break

if __name__ == "__main__":
    main()
