# (c) 2024 chris paxton under MIT license

import time
import timeit

import rclpy
import zmq

from robot_hw_python.remote import StretchClient


def main(port=4401, use_remote_computer: bool = True):
    rclpy.init()
    client = StretchClient()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    if use_remote_computer:
        address = "tcp://*:" + str(port)
    else:
        address = "tcp://127.0.0.1:" + str(port)
    socket.bind(address)

    # Create a stretch client to get information
    sum_time: float = 0
    steps = 0
    t0 = timeit.default_timer()
    while rclpy.ok():
        # get information
        obs = client.get_observation()
        rgb, depth = obs.rgb, obs.depth

        data = {
            "rgb": rgb,
            "depth": depth,
        }

        socket.send_pyobj(data)
        t1 = timeit.default_timer()
        dt = t1 - t0
        sum_time += dt
        steps += 1
        t0 = t1
        print(f"time taken = {dt} avg = {sum_time/steps}")

        time.sleep(0.1)
        t0 = timeit.default_timer()


if __name__ == "__main__":
    main()
