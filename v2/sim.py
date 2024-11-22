# sim.py

import numpy as np
import matplotlib.pyplot as plt
from robotic_armv2 import RoboticArm, Module
from kinematics import Kinematics
from path_planner import PathPlanner  # Assuming you have a path_planner module

def main():
    # Prompt user for target coordinates
    target_x = float(input("Enter the target x coordinate: "))
    target_y = float(input("Enter the target y coordinate: "))
    target_point = (target_x, target_y)

    # Prompt user for number of arms
    n_arms = int(input("Enter the number of robotic arms: "))

    # Create figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Initialize lists to hold arms and kinematics instances
    arms = []
    kinematics_list = []

    # For consistent plotting, share the same axes among all arms
    for i in range(n_arms):
        print(f"\nInitializing Arm {i+1} with default configuration...")
        # Create default modules for each arm
        modules = [Module() for _ in range(2)]  # Default to 2 modules per arm
        # Initialize the robotic arm
        arm = RoboticArm(modules=modules, fig=fig, ax=ax, include_ui=False)
        arms.append(arm)
        # Initialize kinematics
        kinematics = Kinematics(arm)
        kinematics_list.append(kinematics)

    # Set base positions for each arm
    base_positions = []
    for i in range(n_arms):
        base_x = float(input(f"Enter base x position for Arm {i+1}: "))
        base_y = float(input(f"Enter base y position for Arm {i+1}: "))
        base_positions.append((base_x, base_y))

    # Forward Kinematics
    print("\nPerforming Forward Kinematics for each arm...")
    for i, (arm, kinematics) in enumerate(zip(arms, kinematics_list)):
        print(f"\nArm {i+1}:")
        joint_angles = kinematics.get_joint_angles()
        print(f"Initial Joint Angles: {joint_angles}")
        positions = kinematics.forward_kinematics(joint_angles)
        end_effector_pos = positions[-1]
        print(f"End Effector Position: {end_effector_pos}")
        # Update plot
        arm.update_plot(base_position=base_positions[i])
        plt.pause(0.1)

    input("\nPress Enter to continue to Inverse Kinematics...")

    # Inverse Kinematics
    print("\nPerforming Inverse Kinematics to reach the target point...")
    for i, (arm, kinematics) in enumerate(zip(arms, kinematics_list)):
        print(f"\nArm {i+1}:")
        success = kinematics.inverse_kinematics(target_point)
        if success:
            print("IK Successful")
            joint_angles = kinematics.get_joint_angles()
            print(f"New Joint Angles: {joint_angles}")
            positions = kinematics.forward_kinematics(joint_angles)
            end_effector_pos = positions[-1]
            print(f"End Effector Position: {end_effector_pos}")
            # Update plot
            arm.update_plot(base_position=base_positions[i])
            plt.pause(0.1)
        else:
            print("IK Failed to reach the target")

    input("\nPress Enter to continue to Path Planning with RRT...")

    # Path Planning using RRT
    print("\nPerforming Path Planning using RRT for each arm...")
    for i, (arm, kinematics) in enumerate(zip(arms, kinematics_list)):
        print(f"\nArm {i+1}:")
        planner = PathPlanner(arm)
        start_angles = kinematics.get_joint_angles()
        # For the goal angles, we can use the angles found from IK
        goal_angles = start_angles.copy()
        # Reset the arm to initial configuration for simulation
        kinematics.set_joint_angles([np.pi] * len(start_angles))
        print("Resetting arm to initial configuration for path planning...")
        arm.update_plot(base_position=base_positions[i])
        plt.pause(0.1)
        path = planner.plan_path(start_angles, goal_angles)
        if path:
            print("Path found. Simulating movement...")
            for angles in path:
                kinematics.set_joint_angles(angles)
                arm.update_plot(base_position=base_positions[i])
                plt.pause(0.05)
            print("Path planning simulation completed.")
        else:
            print("Path planning failed")

    print("\nSimulation complete.")
    plt.show()

if __name__ == "__main__":
    main()
