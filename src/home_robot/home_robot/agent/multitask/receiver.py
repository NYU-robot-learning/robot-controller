# (c) 2024 chris paxton under MIT license

import time
import timeit

import rclpy
import zmq


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
        while True:
            output = self.socket.recv_pyobj()
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
