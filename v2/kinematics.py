import numpy as np
import random

class Kinematics:
    """
    A class to perform forward and inverse kinematics on a RoboticArm instance.
    """
    def __init__(self, arm):
        """
        Initialize the Kinematics instance.

        Parameters:
            arm (RoboticArm): The robotic arm instance to perform kinematics on.
        """
        self.arm = arm

    def get_joint_angles(self):
        """
        Get all joint angles in the robotic arm.

        Returns:
            angles (list): List of joint angles in radians.
        """
        angles = []
        for module in self.arm.modules:
            joint_angles = module.get_joint_angles()
            angles.extend(joint_angles)
        return angles

    def set_joint_angles(self, angles):
        """
        Set all joint angles in the robotic arm.

        Parameters:
            angles (list): List of joint angles in radians.
        """
        index = 0
        for module in self.arm.modules:
            module.joint1.set_angle(angles[index])
            module.joint2.set_angle(angles[index + 1])
            index += 2

    def forward_kinematics(self, angles=None):
        """
        Compute the positions of each joint and the end effector.

        Parameters:
            angles (list, optional): List of joint angles in radians.
                If None, use the current angles in the arm.

        Returns:
            positions (list of tuples): A list of (x, y) positions of each joint and the end effector.
        """
        if angles is not None:
            self.set_joint_angles(angles)
        x, y = (0, 0)
        positions = [(x, y)]  # List to store joint positions
        current_angle = np.pi / 2  # Start from vertical upwards position

        # Calculate positions of each joint
        for module in self.arm.modules:
            joint_angles = module.get_joint_angles()
            segment_lengths = [module.offset, module.main_length]

            for angle, length in zip(joint_angles, segment_lengths):
                # Adjust current angle for the joint
                current_angle += angle - (np.pi / 2)

                # Compute new x, y positions based on the current angle and length
                x += length * np.cos(current_angle)
                y += length * np.sin(current_angle)

                # Append new position to the list
                positions.append((x, y))

        return positions

    def inverse_kinematics(self, target, max_iterations=1000, tolerance=1e-3, learning_rate=0.1):
        """
        Perform inverse kinematics to reach a target position.

        Parameters:
            target (tuple): The target (x, y) position for the end effector.
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Acceptable distance from the target.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            success (bool): Whether the target was reached within the tolerance.
        """
        # Initialize joint angles
        angles = np.array(self.get_joint_angles())

        for iteration in range(max_iterations):
            # Compute current end effector position
            positions = self.forward_kinematics(angles)
            end_effector = positions[-1]
            error_vector = np.array(target) - np.array(end_effector)
            error = np.linalg.norm(error_vector)

            if error < tolerance:
                self.set_joint_angles(angles)
                return True  # Target reached

            # Compute Jacobian
            J = self.compute_jacobian(angles)

            # Compute change in joint angles using Jacobian transpose method
            dtheta = learning_rate * J.T @ error_vector

            # Update joint angles
            angles += dtheta

            # Enforce joint limits
            self.set_joint_angles(angles)
            angles = np.array(self.get_joint_angles())

        return False  # Failed to reach target within max_iterations

    def compute_jacobian(self, angles):
        """
        Compute the Jacobian matrix for the current joint configuration.

        Parameters:
            angles (numpy array): Array of joint angles in radians.

        Returns:
            J (numpy array): The Jacobian matrix (2 x n).
        """
        num_joints = len(angles)
        J = np.zeros((2, num_joints))

        # Compute positions and cumulative angles
        positions = [(0, 0)]
        cumulative_angles = [np.pi / 2]
        x, y = 0, 0
        for i, angle in enumerate(angles):
            cumulative_angle = cumulative_angles[-1] + angle - (np.pi / 2)
            cumulative_angles.append(cumulative_angle)

            length = self.get_segment_length(i)
            x += length * np.cos(cumulative_angle)
            y += length * np.sin(cumulative_angle)
            positions.append((x, y))

        end_effector = np.array(positions[-1])

        # Compute Jacobian
        for i in range(num_joints):
            J_i = np.array([0.0, 0.0])
            for j in range(i, num_joints):
                cumulative_angle = cumulative_angles[j + 1]
                length = self.get_segment_length(j)
                J_i[0] -= length * np.sin(cumulative_angle)
                J_i[1] += length * np.cos(cumulative_angle)
            J[:, i] = J_i

        return J

    def get_segment_length(self, joint_index):
        """
        Get the length of the segment corresponding to the given joint index.

        Parameters:
            joint_index (int): The index of the joint.

        Returns:
            length (float): The length of the segment.
        """
        module_index = joint_index // 2
        module = self.arm.modules[module_index]
        if joint_index % 2 == 0:
            return module.offset
        else:
            return module.main_length
