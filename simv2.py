# sim_multiple_arms.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from robotic_arm import RoboticArm

def forward_kinematics(angles, lengths, base_position=(0, 0), base_angle=np.pi/2):
    x, y = base_position
    positions = [(x, y)]
    current_angle = base_angle
    for length, angle in zip(lengths, angles):
        current_angle += angle - np.pi / 2
        x += length * np.cos(current_angle)
        y += length * np.sin(current_angle)
        positions.append((x, y))
    return positions  # Return list of positions

def objective_function(angles, lengths1, lengths2, base_positions, target_point):
    n_joints1 = len(lengths1)
    n_joints2 = len(lengths2)
    angles1 = angles[:n_joints1]
    angles2 = angles[n_joints1:]

    positions1 = forward_kinematics(angles1, lengths1, base_positions[0])
    positions2 = forward_kinematics(angles2, lengths2, base_positions[1])

    x1, y1 = positions1[-1]
    x2, y2 = positions2[-1]
    xt, yt = target_point

    # Sum of squared distances to target point
    error1 = (x1 - xt)**2 + (y1 - yt)**2
    error2 = (x2 - xt)**2 + (y2 - yt)**2

    return error1 + error2

def simulate_two_arms_meeting(arm1, arm2, target_point):
    lengths1 = arm1.lengths
    lengths2 = arm2.lengths
    base_positions = [(0, 0), (2.0, 0)]  # Base positions of arm1 and arm2

    initial_angles = arm1.angles + arm2.angles
    bounds = [(0, np.pi)] * (len(arm1.angles) + len(arm2.angles))

    result = minimize(
        objective_function,
        initial_angles,
        args=(lengths1, lengths2, base_positions, target_point),
        method='SLSQP',
        bounds=bounds
    )

    if result.success:
        optimized_angles = result.x
        n1 = len(arm1.angles)
        angles1 = optimized_angles[:n1]
        angles2 = optimized_angles[n1:]

        simulate_movement_two_arms(arm1, arm2, angles1, angles2, base_positions)
    else:
        print("Optimization failed:", result.message)

def simulate_movement_two_arms(arm1, arm2, angles1_target, angles2_target, base_positions, steps=100):
    angles1_initial = np.array(arm1.angles)
    angles2_initial = np.array(arm2.angles)

    step1 = (angles1_target - angles1_initial) / steps
    step2 = (angles2_target - angles2_initial) / steps
    plt.pause(3.02)
    for i in range(steps):
        angles1_current = angles1_initial + step1 * i
        angles2_current = angles2_initial + step2 * i

        arm1.angles = angles1_current
        arm2.angles = angles2_current

        arm1.update_plot(base_position=base_positions[0])
        arm2.update_plot(base_position=base_positions[1])

        plt.pause(0.02)

    print("Simulation completed.")

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Define lengths and initial angles for each arm
    l1 = [1, 1]        # Lengths for arm1
    l2 = [1, 1, 1]     # Lengths for arm2

    a1 = [np.pi / 2, np.pi / 2]       # Initial angles for arm1
    a2 = [np.pi / 4, np.pi / 4, np.pi / 4]  # Initial angles for arm2

    # Create two arms with different numbers of joints
    arm1 = RoboticArm(lengths=l1, angles=a1, fig=fig, ax=ax, include_ui=False)
    arm2 = RoboticArm(lengths=l2, angles=a2, fig=fig, ax=ax, include_ui=False)

    # Define the target meeting point
    target_point = (1.0, 1.5)

    # Start the simulation
    simulate_two_arms_meeting(arm1, arm2, target_point)

    plt.show()
