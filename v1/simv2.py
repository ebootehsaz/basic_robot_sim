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

def collision_constraint_pair(angles, lengths1, lengths2, base_positions, min_distance_threshold, i, j):
    n_joints1 = len(lengths1)
    n_joints2 = len(lengths2)
    angles1 = angles[:n_joints1]
    angles2 = angles[n_joints1:]

    positions1 = forward_kinematics(angles1, lengths1, base_positions[0])
    positions2 = forward_kinematics(angles2, lengths2, base_positions[1])

    p1_start = positions1[i]
    p1_end = positions1[i + 1]
    p2_start = positions2[j]
    p2_end = positions2[j + 1]
    dist = segment_distance(p1_start, p1_end, p2_start, p2_end)

    # The constraint is that the distance minus the threshold should be non-negative
    return dist - min_distance_threshold


def segment_distance(p1, p2, q1, q2):
    """
    Calculate the minimum distance between two line segments (p1-p2) and (q1-q2).
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

    sc, sN, sD = 0.0, D, D  # sc = sN / sD
    tc, tN, tD = 0.0, D, D  # tc = tN / tD

    SMALL_NUM = 1e-8

    if D < SMALL_NUM:  # The lines are almost parallel
        sN = 0.0        # Force using tN later
        sD = 1.0        # To prevent division by zero
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

    if abs(sD) < SMALL_NUM:
        sc = 0.0
    else:
        sc = sN / sD

    if abs(tD) < SMALL_NUM:
        tc = 0.0
    else:
        tc = tN / tD

    dP = w + (sc * u) - (tc * v)
    distance = np.linalg.norm(dP)
    return distance

def objective_function(angles, lengths1, lengths2, base_positions, target_point):
    n1 = len(lengths1)
    angles1 = angles[:n1]
    angles2 = angles[n1:]

    positions1 = forward_kinematics(angles1, lengths1, base_positions[0])
    positions2 = forward_kinematics(angles2, lengths2, base_positions[1])

    x1, y1 = positions1[-1]
    x2, y2 = positions2[-1]
    xt, yt = target_point

    # Sum of squared distances to the target point
    error1 = (x1 - xt)**2 + (y1 - yt)**2
    error2 = (x2 - xt)**2 + (y2 - yt)**2

    return error1 + error2


def simulate_two_arms_meeting(arm1, arm2, target_point):
    lengths1 = arm1.lengths
    lengths2 = arm2.lengths
    base_positions = [(0, 0), (2.0, 0)]  # Positions of arm bases

    initial_angles = arm1.angles + arm2.angles
    bounds = [(0, np.pi)] * (len(initial_angles))

    min_distance_threshold = 0.2  # Adjust as needed for minimum distance

    # Define the collision avoidance constraints
    constraints = []
    for i in range(len(lengths1)):
        for j in range(len(lengths2)):
            constraints.append({
                'type': 'ineq',
                'fun': lambda angles, i=i, j=j: collision_constraint_pair(
                    angles, lengths1, lengths2, base_positions, min_distance_threshold, i, j)
            })

    # Run the optimizer with the constraints
    result = minimize(
        objective_function,
        initial_angles,
        args=(lengths1, lengths2, base_positions, target_point),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        optimized_angles = result.x
        n1 = len(lengths1)
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
    plt.pause(1.0)
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
    a2 = [np.pi / 1, np.pi / 5, np.pi / 1]  # Initial angles for arm2

    # Create two arms with different numbers of joints
    arm1 = RoboticArm(lengths=l1, angles=a1, fig=fig, ax=ax, include_ui=False)
    arm2 = RoboticArm(lengths=l2, angles=a2, fig=fig, ax=ax, include_ui=False)

    # Define the target meeting point
    target_point = (1.0, 1.5)

    # Start the simulation
    simulate_two_arms_meeting(arm1, arm2, target_point)

    plt.show()
