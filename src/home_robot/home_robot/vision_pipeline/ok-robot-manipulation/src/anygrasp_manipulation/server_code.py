import zmq
import time

# Path to the PCD file
POINT_CLOUD_PCD_PATH = "/data/mohit/src/anygrasp_manipulation/point_clouds/3d_point_cloud.pcd"

LOCAL_COMPUTER_IP = "localhost"
LOCAL_COMPUTER_PORT = 5560

def send_point_cloud(ip, port):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{ip}:{port}")

    while True:
        try:
            with open(POINT_CLOUD_PCD_PATH, 'rb') as f:
                data = f.read()
                socket.send(data)
        except Exception as e:
            print(f"Error reading or sending PCD data: {e}")
        time.sleep(1)

if __name__ == "__main__":
    send_point_cloud(LOCAL_COMPUTER_IP, LOCAL_COMPUTER_PORT)

