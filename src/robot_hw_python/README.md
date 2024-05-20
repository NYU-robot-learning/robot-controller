### HomeRobot ROS2 Installation

```sh
# Make sure ROS can find python properly
sudo apt install python-is-python3 pybind11-dev

# Clone the repository
git clone https://github.com/NYU-robot-learning/robot-controller
git checkout cpaxton/ros2-migration
HOME_ROBOT_ROOT=$(realpath robot-controller)

# Install requirements
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -r requirements.txt

# Install the core home_robot package
pip install -e .

# Set up the python package for ROS
ln -s $HOME_ROBOT_ROOT/src/robot_hw_python $HOME/ament_ws/src/robot_hw_python

# Rebuild ROS2 packages to make sure paths are correct
cd $HOME/ament_ws
colcon build
```

### Running

In one terminal build robot_hw_python package after any changes has been made:
```
cd ~/ament_ws
colcon build --symlink-install --packages-select robot_hw_python
```
Using the `--symlink-install` flag means that you only need to run this once if you make any edits.


Then, in another terminal run the launch file
```
cd ~/ament_ws
sudo ./install/setup.bash
ros2 launch robot_hw_python startup_stretch_hector_slam.launch.py
```

#### Option 2

```
cd ~/ament_ws
sudo ./install/setup.bash
ros2 launch robot_hw_python start_server.py
```

In another terminal open python shell and run following commands to operate the robot using stretch_user_client node
```
>>> import rclpy
>>> rclpy.init()
>>> from robot_hw_python.remote import StretchClient
>>> r = StretchClient(urdf_path="/home/hello-robot/yaswanth/robot-controller/src/home_robot_hw/assets/hab_stretch/urdf")

# Manipulation
>>> r.switch_to_manipulation_mode()
>>> state = r.manip.get_joint_positions()
>>> state[1] = 0.8 # For lifting
>>> r.manip.goto_joint_positions(state)

# Navigation
>>> r.switch_to_navigation_mode()
>>> r.nav.navigate_to([0.5, 0, 0]) # For Navigation
```
