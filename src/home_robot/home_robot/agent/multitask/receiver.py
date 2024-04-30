# (c) 2024 chris paxton under MIT license

import time
import timeit

import cv2
import numpy as np
import rclpy
import zmq

from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import show_point_cloud


class HomeRobotZmqClient:
    def __init__(
        self,
        robot_ip: str = "192.168.1.15",
        port: int = 4401,
        use_remote_computer: bool = True,
    ):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)

        # Use remote computer or whatever
        if use_remote_computer:
            self.address = "tcp://" + robot_ip + ":" + str(self.port)
        else:
            self.address = "tcp://" + "127.0.0.1" + ":" + str(self.port)

        print("Connecting to address...")
        self.socket.connect(self.address)
        print("...connected.")

    def blocking_spin(self):
        """this is just for testing"""
        sum_time = 0
        steps = 0
        t0 = timeit.default_timer()
        camera = None
        while True:
            output = self.socket.recv_pyobj()
            output["rgb"] = cv2.imdecode(output["rgb"], cv2.IMREAD_COLOR)
            compressed_depth = output["depth"]
            depth = cv2.imdecode(compressed_depth, cv2.IMREAD_UNCHANGED)
            output["depth"] = depth / 1000.0

            if camera is None:
                camera = Camera.from_K(
                    output["camera_K"], output["rgb_height"], output["rgb_width"]
                )

            output["xyz"] = camera.depth_to_xyz(output["depth"])
            # show_point_cloud(output["xyz"], output["rgb"] / 255., orig=np.zeros(3))

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            print(
                f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
            )
            t0 = timeit.default_timer()


if __name__ == "__main__":
    client = HomeRobotZmqClient(
        robot_ip="192.168.1.15",
        port=4401,
        use_remote_computer=True,
    )
    client.blocking_spin()
