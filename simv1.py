# sim.py
from robotic_arm import RoboticArm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def forward_kinematics(angles, lengths):
    """
    Compute the (x, y) position of the end effector given joint angles and link lengths.
    """
    x, y = 0, 0
    current_angle = np.pi / 2  # Start from vertical upwards
    for length, angle in zip(lengths, angles):
        current_angle += angle - np.pi / 2  # Adjust current angle
        x += length * np.cos(current_angle)
        y += length * np.sin(current_angle)
    return x, y

def objective(angles, lengths):
    """
    Objective function to maximize the x-coordinate of the end effector.
    """
    x, _ = forward_kinematics(angles, lengths)
    return -x  # Negative because we minimize in optimization

def y_constraint(angles, lengths, ye_initial):
    """
    Constraint function to keep the y-coordinate constant.
    """
    _, y = forward_kinematics(angles, lengths)
    return y - ye_initial

def move_arm_to_max_x_same_y(arm):
    """
    Move the arm to the position where the x-coordinate is maximized while keeping y constant.
    """
    lengths = arm.lengths
    ye_initial = forward_kinematics(arm.angles, lengths)[1]
    print("Initial y-coordinate:", ye_initial)
    initial_guess = np.array(arm.angles)
    bounds = [(0, np.pi)] * len(initial_guess)
    cons = {'type': 'eq', 'fun': y_constraint, 'args': (lengths, ye_initial)}
    
    result = minimize(
        objective,
        initial_guess,
        args=(lengths,),
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
    
    if result.success:
        target_angles = result.x
        simulate_movement(arm, target_angles)
        final_x, final_y = forward_kinematics(target_angles, lengths)
        print(f"Final position: x = {final_x}, y = {final_y}")
    else:
        print("Optimization failed:", result.message)

def simulate_movement(arm, target_angles, steps=100):
    current_angles = np.array(arm.angles)
    target_angles = np.array(target_angles)
    step = (target_angles - current_angles) / steps
    
    input("Press Enter to start the simulation...")
    plt.pause(1.02)
    for i in range(steps):
        current_angles += step
        arm.update_angles(current_angles)
        plt.pause(0.02)  # Adjust the pause to control the simulation speed
        print(f"Step {i+1}/{steps}: Angles = {current_angles}")
    
    print(f"Final angles: {current_angles}")

if __name__ == "__main__":
    angles = [1.2, 0]
    arm = RoboticArm(angles=angles)
    plt.show(block=False)
    move_arm_to_max_x_same_y(arm)
    plt.show()  # Keep the plot open after the simulation
