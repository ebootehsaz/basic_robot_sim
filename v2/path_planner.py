from kinematics import Kinematics
import numpy as np
import random


class PathPlanner:
    """
    A class to perform path planning using Rapidly-exploring Random Trees (RRT).
    """
    def __init__(self, arm):
        self.arm = arm
        self.kinematics = Kinematics(arm)

    def plan_path(self, start_angles, goal_angles, max_iterations=1000, step_size=0.1):
        """
        Plan a path from start_angles to goal_angles using RRT.

        Parameters:
            start_angles (list): Starting joint angles.
            goal_angles (list): Goal joint angles.
            max_iterations (int): Maximum number of iterations.
            step_size (float): Step size for advancing towards random samples.

        Returns:
            path (list): A list of joint angle configurations from start to goal.
        """
        num_joints = len(start_angles)
        tree = [np.array(start_angles)]
        parent = {tuple(start_angles): None}

        for iteration in range(max_iterations):
            # Randomly sample a configuration
            random_sample = np.array([random.uniform(0.0, 2 * np.pi) for _ in range(num_joints)])

            # Find nearest node in the tree
            nearest_node = min(tree, key=lambda node: np.linalg.norm(node - random_sample))

            # Move from nearest node towards random sample
            direction = random_sample - nearest_node
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            direction = direction / length
            new_node = nearest_node + step_size * direction

            # Enforce joint limits
            new_node = np.clip(new_node, 0.0, 2 * np.pi)

            # Optionally, check for collisions here (not implemented)

            # Add new node to tree
            tree.append(new_node)
            parent[tuple(new_node)] = tuple(nearest_node)

            # Check if goal is reached
            if np.linalg.norm(new_node - goal_angles) < step_size:
                # Build path
                path = [goal_angles]
                node = tuple(new_node)
                while node is not None:
                    path.append(list(node))
                    node = parent.get(node)
                path.reverse()
                return path

        # Failed to find a path
        return None
