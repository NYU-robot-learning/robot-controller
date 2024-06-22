import cv2
import time
import zmq
import sys

from robot_hw_python.remote import StretchClient
import rclpy
rclpy.init()
robot = StretchClient()
robot.switch_to_manipulation_mode()
state = robot.manip.get_joint_positions()
robot.manip.goto_joint_positions(state)
robot.head
robot.head.get_pan_tilt()
robot.head.get_images()
rgb, depth = robot.head.get_images()
rgb.shape


def start_zmq_client():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    try:
        socket.connect("tcp://10.19.214.56:5555")
    except zmq.ZMQError as e:
        if e.errno == zmq.EADDRINUSE:
            print("Address already in use. Please ensure the server is running.")
            sys.exit(1)
    return context, socket

context, socket = start_zmq_client()

def capture_and_send_image():
    rgb, depth = robot.head.get_images()
    
    rgb = rgb[:, :, [2, 1, 0]]
    
    _, buffer = cv2.imencode('.jpg', rgb)
    image_data = buffer.tobytes()

    socket.send(image_data)

while True:
    try:
        capture_and_send_image()
    except Exception as e:
        print(f"An error occurred: {e}")
    time.sleep(0.0001)
