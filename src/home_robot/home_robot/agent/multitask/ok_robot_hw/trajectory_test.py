import copy

from home_robot_hw.remote import StretchClient

if __name__ == "__main__":
    #robot = StretchClient(urdf_path = "hab_stretch/urdf/stretch_manip_mode.urdf")
    robot = StretchClient(urdf_path = "hab_stretch/urdf")
    robot.switch_to_manipulation_mode()

    # current state
    state = robot.manip.get_joint_positions()
    eps = 0.00001
    for i, val in enumerate(state):
        state[i] = state[i] + eps

    # pull the arm back
    state[2] = 0.05
    p1_state = copy.deepcopy(state)
    
    # move the arm down
    state[4] = -1.5
    p2_state = copy.deepcopy(state)

    # lift the arm up
    state[1] = 1
    p3_state = copy.deepcopy(state)

    # rotate the gripper
    state[4] = -0.9
    p4_state = copy.deepcopy(state)

    # Move down and forward
    state[1] = 0.85
    state[2] = 0.7
    p5_state = copy.deepcopy(state)

    joint_positions = [p1_state, p2_state, p3_state, p4_state, p5_state]
    robot.manip.goto_joint_positions(joint_positions)
