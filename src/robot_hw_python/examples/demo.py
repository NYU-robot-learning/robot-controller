#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rclpy

from home_robot.agent.multitask.demo import demo_main
from robot_hw_python.remote import StretchClient

if __name__ == "__main__":
    rclpy.init()
    robot = StretchClient()
    demo_main(robot)
