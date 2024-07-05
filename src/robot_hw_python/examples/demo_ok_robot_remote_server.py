import datetime
import cv2
import numpy as np
import click
import rclpy
import requests

from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask import RemoteRobotAgentManip as RobotAgent
from robot_hw_python.remote import StretchClient

def update_ui_step(step_name):
    url = "http://10.19.214.56:5000/update_step"
    data = {"step": step_name}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"Failed to update UI step: {e}")

def update_ui_task(task_name):
    url = "http://10.19.214.56:5000/update_task"
    data = {"task": task_name}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"Failed to update UI task: {e}")

def compute_tilt(camera_xyz, target_xyz):
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

@click.command()
@click.option("--rate", default=5, type=int)
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

    rclpy.init()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pcd_filename = output_filename + "_" + formatted_datetime + ".pcd"
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

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

    while True:
        mode = input('select mode? E/N/S')
        if mode == 'S':
            break
        if mode == 'E':
            update_ui_task("Exploration Mode Activated")
            update_ui_step("Switching the robot to navigation mode")
            robot.switch_to_navigation_mode()
            update_ui_step("Running the exploration routine")
            demo.run_exploration(
                rate,
                explore_iter=2
            )
        else:
            update_ui_task("Pickup Mode Activated")
            robot.move_to_nav_posture()
            robot.head.look_front()
            robot.switch_to_navigation_mode()
            text = input('Enter object name: ')
            update_ui_step(f"Navigating to {text}")
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure')
                continue
            cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])
            update_ui_step(f"Captured image of {text}")
            robot.switch_to_navigation_mode()
            xyt = robot.nav.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.nav.navigate_to(xyt)
            update_ui_step(f"Arrived at {text}")

            if input('You want to run manipulation: y/n') == 'n':
                continue
            update_ui_step(f"Preparing to manipulate {text}")
            camera_xyz = robot.head.get_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.manipulate(text, theta)
            update_ui_step(f"Manipulated {text}")
            
            robot.head.look_front()
            robot.switch_to_navigation_mode()
            if input('You want to run placing: y/n') == 'n':
                continue
            update_ui_task("Place Mode Activated")
            text = input('Enter receptacle name: ')
            update_ui_step(f"Navigating to {text}")
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure')
                continue
            cv2.imwrite(text + '.jpg', demo.robot.get_observation().rgb[:, :, [2, 1, 0]])
            update_ui_step(f"Captured image of {text}")
            robot.switch_to_navigation_mode()
            xyt = robot.nav.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.nav.navigate_to(xyt)
            update_ui_step(f"Arrived at {text}")
        
            if input('You want to run placing: y/n') == 'n':
                continue
            update_ui_step(f"Preparing to place {text}")
            camera_xyz = robot.head.get_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.place(text, theta)
            update_ui_step(f"Placed {text}")

if __name__ == "__main__":
    main()
