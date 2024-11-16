from robotic_arm import RoboticArm
import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_movement(arm, target_angles, steps=100):
    current_angles = np.array(arm.angles)
    target_angles = np.array(target_angles)
    step = (target_angles - current_angles) / steps
        
    for i in range(steps):
        current_angles += step
        arm.update_angles(current_angles)
        plt.pause(0.05)  # Adjust the pause to control the simulation speed
        print(f"Step {i+1}/{steps}: Angles = {current_angles}")

    print(f"Final angles: {current_angles}")

if __name__ == "__main__":
    l = [1.0, 1.0]
    a = [np.pi / 2, 0]
    arm = RoboticArm(l, a)
    plt.show(block=False)
    # Define target angles for simulation
    target_angles = [0, -np.pi / 2]  # Example target angles for two joints
    simulate_movement(arm, target_angles)
    plt.show()  # Keep the plot open after the simulation
