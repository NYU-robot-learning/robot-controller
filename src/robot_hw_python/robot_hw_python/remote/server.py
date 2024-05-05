# (c) 2024 chris paxton for Hello Robot, under MIT license

import threading
import time
import timeit
from typing import Optional

import click
import cv2
import numpy as np
import rclpy
import zmq

from home_robot.core.interfaces import ContinuousNavigationAction
from robot_hw_python.remote import StretchClient


class ZmqServer:
    def __init__(
        self,
        send_port: int = 4401,
        recv_port: int = 4402,
        use_remote_computer: bool = True,
        desktop_ip: Optional[str] = "192.168.1.10",
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.client = StretchClient()
        self.context = zmq.Context()

        # Set up the publisher socket using ZMQ
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)

        # Set up the receiver/subscriber using ZMQ
        self.recv_socket = self.context.socket(zmq.SUB)
        self.recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.recv_socket.setsockopt(zmq.SNDHWM, 1)
        self.recv_socket.setsockopt(zmq.RCVHWM, 1)
        self.recv_socket.setsockopt(zmq.CONFLATE, 1)
        self._last_step = -1

        # Make connections
        if use_remote_computer:
            address = "tcp://*:" + str(send_port)
            assert desktop_ip is not None, "must provide a valid IP address for remote"
        else:
            desktop_ip = "127.0.0.1"
            address = f"tcp://{desktop_ip}:" + str(send_port)
        self.recv_address = f"tcp://{desktop_ip}:{recv_port}"
        print(f"Publishing on {address}...")
        self.send_socket.bind(address)
        print(f"Waiting for actions on {self.recv_address}...")
        self.recv_socket.connect(self.recv_address)
        print("Done!")

        # for the threads
        self.control_mode = "none"
        self._done = False

    def spin_send(self):

        # Create a stretch client to get information
        sum_time: float = 0
        steps = 0
        t0 = timeit.default_timer()
        while rclpy.ok() and not self._done:
            # get information
            obs = self.client.get_observation()
            rgb, depth = obs.rgb, obs.depth
            width, height = rgb.shape[:2]

            # Convert depth into int format
            depth = (depth * 1000).astype(np.uint16)

            # Make both into jpegs
            _, rgb = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, depth = cv2.imencode(
                ".jp2", depth, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 800]
            )

            if self.client.in_manipulation_mode():
                control_mode = "manipulation"
            elif self.client.in_navigation_mode():
                control_mode = "navigation"
            else:
                control_mode = "none"

            # Get the other fields from an observation
            data = {
                "rgb": rgb,
                "depth": depth,
                "camera_K": obs.camera_K.cpu().numpy(),
                "camera_pose": obs.camera_pose,
                "joint": obs.joint,
                "gps": obs.gps,
                "compass": obs.compass,
                "rgb_width": width,
                "rgb_height": height,
                "control_mode": control_mode,
                "last_motion_failed": self.client.last_motion_failed(),
                "recv_address": self.recv_address,
                "step": self._last_step,
                "at_goal": self.client.at_goal(),
            }

            self.send_socket.send_pyobj(data)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose:
                print(f"[SEND] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(0.1)
            t0 = timeit.default_timer()

    def spin_recv(self):
        sum_time: float = 0
        steps = 0
        t0 = timeit.default_timer()
        while rclpy.ok() and not self._done:
            try:
                action = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                if self.verbose:
                    print(" - no action received")
                action = None
            if self.verbose:
                print(f" - {self.control_mode=}")
                print(f" - prev action step: {self._last_step}")
            if action is not None:
                if self.verbose:
                    print(f" - Action received: {action}")
                self._last_step = action.get("step", -1)
                if self.verbose:
                    print(f" - last action step: {self._last_step}")
                if "posture" in action:
                    if action["posture"] == "manipulation":
                        self.client.move_to_manip_posture()
                        self.client.switch_to_manipulation_mode()
                    elif action["posture"] == "navigation":
                        self.client.move_to_nav_posture()
                        self.client.switch_to_navigation_mode()
                    else:
                        print(
                            " - posture",
                            action["posture"],
                            "not recognized or supported.",
                        )
                if "control_mode" in action:
                    if action["control_mode"] == "manipulation":
                        self.client.switch_to_manipulation_mode()
                        self.control_mode = "manipulation"
                    elif action["control_mode"] == "navigation":
                        self.client.switch_to_navigation_mode()
                        self.control_mode = "navigation"
                    else:
                        print(
                            " - control mode",
                            action["control_mode"],
                            "not recognized or supported.",
                        )
                if "xyt" in action:
                    if self.verbose:
                        print(
                            "Is robot in navigation mode?",
                            self.client.in_navigation_mode(),
                        )
                        print(
                            f"{action['xyt']} {action['nav_relative']} {action['nav_blocking']}"
                        )
                    self.client.navigate_to(
                        action["xyt"],
                        relative=action["nav_relative"],
                        # TODO: should we actually block here? Probably not, right?
                        # blocking=action["nav_blocking"],
                    )

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose:
                print(f"[RECV] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(0.1)
            t0 = timeit.default_timer()

    def start(self):
        """Starts both threads spinning separately for efficiency."""
        self._send_thread = threading.Thread(target=self.spin_send)
        self._recv_thread = threading.Thread(target=self.spin_recv)
        self._done = False
        self._send_thread.start()
        self._recv_thread.start()

    def __del__(self):
        self._done = True
        self.recv_socket.close()
        self.send_socket.close()
        self.context.term()
        self._send_thread.join()
        self._recv_thread.join()


@click.command()
@click.option("--send_port", default=4401, help="Port to send observations to")
@click.option("--recv_port", default=4402, help="Port to receive actions from")
@click.option("--desktop_ip", default="192.168.1.9", help="IP address of desktop")
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
def main(
    send_port: int = 4401,
    recv_port: int = 4402,
    desktop_ip: str = "192.168.1.10",
    local: bool = False,
):
    rclpy.init()
    server = ZmqServer(
        send_port=send_port,
        recv_port=recv_port,
        desktop_ip=desktop_ip,
        use_remote_computer=(not local),
    )
    server.start()


if __name__ == "__main__":
    main()
