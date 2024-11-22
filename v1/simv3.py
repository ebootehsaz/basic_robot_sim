# simv3.py

import numpy as np
import matplotlib.pyplot as plt
from robotic_arm import RoboticArm
from scipy.optimize import minimize, NonlinearConstraint

def forward_kinematics(angles, lengths, base_position=(0, 0), base_angle=np.pi/2):
    """
    Compute the positions of each joint and the end effector given joint angles and link lengths.
    """
    x, y = base_position
    positions = [(x, y)]
    current_angle = base_angle
    for length, angle in zip(lengths, angles):
        current_angle += angle - np.pi / 2
        x += length * np.cos(current_angle)
        y += length * np.sin(current_angle)
        positions.append((x, y))
    return positions

def segment_distance(p1, p2, q1, q2):
    """
    Calculate the minimum distance between two line segments in 2D.
    """
    # Convert points to numpy arrays
    p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))
    # Direction vectors
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b

    SMALL_NUM = 1e-8

    sc = sN = sD = D
    tc = tN = tD = D

    if D < SMALL_NUM:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
        elif sN > sD:
            sN = sD

    if tN < 0.0:
        tN = 0.0
    elif tN > tD:
        tN = tD

    if abs(sD) > SMALL_NUM:
        sc = sN / sD
    else:
        sc = 0.0

    if abs(tD) > SMALL_NUM:
        tc = tN / tD
    else:
        tc = 0.0

    dP = w + (sc * u) - (tc * v)
    distance = np.linalg.norm(dP)
    return distance

def collision_constraints(angles, lengths1, lengths2, base_positions, min_distance_threshold):
    """
    Compute the minimum distance between any segments of the two arms and return the constraint value.
    The constraint is satisfied when this value is non-negative.
    """
    n1 = len(lengths1)
    angles1 = angles[:n1]
    angles2 = angles[n1:]

    positions1 = forward_kinematics(angles1, lengths1, base_positions[0])
    positions2 = forward_kinematics(angles2, lengths2, base_positions[1])

    min_distance = np.inf
    for i in range(len(positions1) - 1):
        for j in range(len(positions2) - 1):
            dist = segment_distance(positions1[i], positions1[i + 1], positions2[j], positions2[j + 1])
            if dist < min_distance:
                min_distance = dist

    return min_distance - min_distance_threshold

def objective_function(angles, lengths1, lengths2, base_positions, target_point):
    """
    Objective function to minimize the total distance of the arms' end effectors to the target point.
    """
    n1 = len(lengths1)
    angles1 = angles[:n1]
    angles2 = angles[n1:]

    positions1 = forward_kinematics(angles1, lengths1, base_positions[0])
    positions2 = forward_kinematics(angles2, lengths2, base_positions[1])

    x1, y1 = positions1[-1]
    x2, y2 = positions2[-1]
    xt, yt = target_point

    error1 = (x1 - xt) ** 2 + (y1 - yt) ** 2
    error2 = (x2 - xt) ** 2 + (y2 - yt) ** 2

    total_error = error1 + error2
    return total_error

if __name__ == "__main__":
    # Set up the figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)
    ax.set_aspect('equal', 'box')
    ax.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

    # Define lengths and initial angles for each arm
    l1 = [1, 1]
    l2 = [1, 1, 1]

    a1_initial = [np.pi / 4, np.pi / 2]
    a2_initial = [np.pi / 4] * 3

    # Base positions of the arms
    base_position1 = (0, 0)
    base_position2 = (2.0, 0)

    # Create two arms
    arm1 = RoboticArm(lengths=l1, angles=a1_initial, fig=fig, ax=ax, include_ui=False)
    arm2 = RoboticArm(lengths=l2, angles=a2_initial, fig=fig, ax=ax, include_ui=False)

    # Define the target meeting point
    target_point = (1.0, 1.5)

    # Set up the optimization
    lengths1 = arm1.lengths
    lengths2 = arm2.lengths

    initial_angles = arm1.angles + arm2.angles
    bounds = [(0, np.pi)] * len(initial_angles)

    base_positions = [base_position1, base_position2]
    min_distance_threshold = 0.5  # Increase the threshold for more separation

    # Define the collision avoidance constraint
    collision_constraint = NonlinearConstraint(
        lambda angles: collision_constraints(angles, lengths1, lengths2, base_positions, min_distance_threshold),
        lb=0.0,
        ub=np.inf
    )

    # Run the optimizer
    result = minimize(
        objective_function,
        initial_angles,
        args=(lengths1, lengths2, base_positions, target_point),
        method='SLSQP',
        bounds=bounds,
        constraints=[collision_constraint],
        options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000}
    )

    if result.success:
        optimized_angles = result.x
        n1 = len(lengths1)
        angles1 = optimized_angles[:n1]
        angles2 = optimized_angles[n1:]

        # Simulate the movement
        steps = 200
        angles1_initial = np.array(arm1.angles)
        angles2_initial = np.array(arm2.angles)
        angles1_target = np.array(angles1)
        angles2_target = np.array(angles2)

        angles1 = angles1_initial.copy()
        angles2 = angles2_initial.copy()
        step_angles1 = (angles1_target - angles1_initial) / steps
        step_angles2 = (angles2_target - angles2_initial) / steps

        plt.pause(1.0)

        # Simulation loop
        for step in range(steps):
            # Update angles incrementally
            angles1 = angles1_initial + step_angles1 * (step + 1)
            angles2 = angles2_initial + step_angles2 * (step + 1)

            # Update the arms
            arm1.angles = angles1
            arm1.update_plot(base_position=base_position1)

            arm2.angles = angles2
            arm2.update_plot(base_position=base_position2)

            # Redraw the plot
            plt.pause(0.02)

        print("Simulation completed.")
        plt.show()
    else:
        print("Optimization failed:", result.message)
