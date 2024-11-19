import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class RoboticArm:
    """
    A class to represent a 2D planar robotic arm with multiple joints.

    This class allows for the simulation and visualization of a robotic arm in 2D space,
    where the lengths and angles of each joint can be interactively adjusted using sliders.
    The arm can also have joints added or removed dynamically.

    Attributes:
        lengths (list of float): Lengths of each joint segment.
        angles (list of float): Angles of each joint in radians.
        fig (matplotlib.figure.Figure): Matplotlib figure object for plotting.
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting.
        line (matplotlib.lines.Line2D): Line object representing the robotic arm.
        sliders (list): List of slider widgets for lengths and angles.

    Methods:
        __init__(self, lengths=None, angles=None): Initialize the robotic arm.
        constrain_angle(self, angle): Constrain angle within acceptable range.
        init_plot(self): Initialize the plot.
        init_ui(self): Initialize the user interface components (sliders, buttons).
        clear_sliders(self): Remove all slider widgets.
        update_length(self, val, index): Update the length of a joint.
        update_angle(self, val, index): Update the angle of a joint.
        update_angles(self, new_angles): Update all joint angles.
        update_plot(self): Update the plot of the robotic arm.
        add_joint(self, event): Add a new joint to the arm.
        remove_joint(self, event): Remove the last joint from the arm.
    """

    def __init__(self, lengths=None, angles=None):
        """
        Initialize the RoboticArm instance with given lengths and angles.

        Parameters:
            lengths (list of float, optional): Lengths of each joint segment.
                Defaults to [1.0, 0.5].
            angles (list of float, optional): Angles of each joint in radians.
                Defaults to [π/2, π/2].

        Raises:
            ValueError: If lengths and angles are not of the same length.
        """
        if lengths is None:
            self.lengths = [1.0, 0.5]  # Default lengths of each joint
        else:
            self.lengths = lengths

        if angles is None:
            self.angles = [np.pi / 2, np.pi / 2]  # Set all angles to pi/2 by default
        else:
            # Constrain all angles within acceptable range
            self.angles = [self.constrain_angle(angle) for angle in angles]

        # Ensure lengths and angles lists are of the same length
        if len(self.lengths) != len(self.angles):
            raise ValueError("Lengths and angles must be of the same length.")

        # Initialize the figure and axes for plotting
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.3)  # Adjust plot to make space for sliders

        # Initialize the line object that will represent the robotic arm
        self.line, = self.ax.plot([], [], 'o-', lw=2)

        # List to hold slider widgets
        self.sliders = []

        # Initialize the plot and user interface
        self.init_plot()
        self.init_ui()

    def constrain_angle(self, angle):
        """
        Constrain an angle to be within the range [0, π].

        Parameters:
            angle (float): The angle to constrain.

        Returns:
            float: The constrained angle.
        """
        return np.clip(angle, 0, np.pi)

    def init_plot(self):
        """
        Initialize the plot by updating it with the initial positions.
        """
        self.update_plot()

    def init_ui(self):
        """
        Initialize the user interface components including sliders and buttons.
        """
        # Clear any existing sliders
        self.clear_sliders()

        # Define slider appearance
        axcolor = 'lightgoldenrodyellow'  # Color for the slider background

        base_y = 0.02  # Starting y-position for the sliders
        slider_height = 0.03  # Height of each slider
        slider_spacing = 0.04  # Spacing between sliders

        # Create sliders for each joint's length and angle
        for i in range(len(self.lengths)):
            # Create axes for length slider
            ax_length = plt.axes([0.1, base_y, 0.65, slider_height], facecolor=axcolor)
            # Create length slider
            s_length = Slider(ax_length, f'Length {i+1}', 0.1, 2.0, valinit=self.lengths[i])
            # Set the update function for length slider
            # Use default argument idx=i to capture the current index in lambda
            s_length.on_changed(lambda val, idx=i: self.update_length(val, idx))

            # Create axes for angle slider
            ax_angle = plt.axes([0.1, base_y + slider_spacing, 0.65, slider_height], facecolor=axcolor)
            # Create angle slider
            s_angle = Slider(ax_angle, f'Angle {i+1}', 0, np.pi, valinit=self.angles[i])
            # Set the update function for angle slider
            # Use default argument idx=i to capture the current index in lambda
            s_angle.on_changed(lambda val, idx=i: self.update_angle(val, idx))

            # Append sliders to the list
            self.sliders.append((s_length, s_angle))

            # Update base_y for the next set of sliders
            # Each joint requires space for two sliders (length and angle)
            base_y += 2 * (slider_height + slider_spacing)

        # Create 'Add Joint' button
        self.add_button = Button(
            plt.axes([0.8, 0.05, 0.1, 0.04]),
            'Add Joint',
            color=axcolor,
            hovercolor='0.975'
        )
        self.add_button.on_clicked(self.add_joint)

        # Create 'Remove Joint' button
        self.remove_button = Button(
            plt.axes([0.8, 0.01, 0.1, 0.04]),
            'Remove Joint',
            color=axcolor,
            hovercolor='0.975'
        )
        self.remove_button.on_clicked(self.remove_joint)

    def clear_sliders(self):
        """
        Remove all existing slider widgets from the figure.
        """
        for s_length, s_angle in self.sliders:
            s_length.ax.remove()
            s_angle.ax.remove()
        self.sliders.clear()

    def update_length(self, val, index):
        """
        Update the length of a joint and refresh the plot.

        Parameters:
            val (float): The new length value.
            index (int): The index of the joint to update.
        """
        self.lengths[index] = val
        self.update_plot()

    def update_angle(self, val, index):
        """
        Update the angle of a joint and refresh the plot.

        Parameters:
            val (float): The new angle value in radians.
            index (int): The index of the joint to update.
        """
        self.angles[index] = self.constrain_angle(val)
        self.update_plot()

    def update_angles(self, new_angles):
        """
        Update all joint angles and refresh the plot.

        Parameters:
            new_angles (list of float): List of new angle values in radians.

        Raises:
            ValueError: If the length of new_angles does not match the number of joints.
        """
        if len(new_angles) != len(self.angles):
            raise ValueError("New angles list must be the same length as current angles.")

        # Constrain and update angles
        self.angles = [self.constrain_angle(angle) for angle in new_angles]

        # Update slider positions to reflect new angles
        for i, (s_length, s_angle) in enumerate(self.sliders):
            s_angle.set_val(self.angles[i])

        self.update_plot()

    def update_plot(self):
        """
        Update the plot of the robotic arm based on current lengths and angles.
        """
        # Starting position (origin)
        x, y = 0, 0
        positions = [(x, y)]  # List to store joint positions

        current_angle = np.pi / 2  # Start from vertical upwards position

        # Calculate positions of each joint
        for length, angle in zip(self.lengths, self.angles):
            # Adjust current angle for the joint
            # Subtract π/2 because the initial angle is set to vertical (π/2)
            # This ensures that angle sliders correspond to the angle between segments
            current_angle += angle - np.pi / 2  # Adjust current angle

            # Compute new x, y positions based on the current angle and length
            x += length * np.cos(current_angle)
            y += length * np.sin(current_angle)

            # Append new position to the list
            positions.append((x, y))

        # Unzip positions into x and y coordinates
        xs, ys = zip(*positions)

        # Update line data
        self.line.set_data(xs, ys)

        # Adjust plot limits to ensure the entire arm is visible
        max_length = sum(self.lengths) * 1.5  # Extend limits beyond arm's reach for better visibility
        self.ax.set_xlim(-max_length, max_length)
        self.ax.set_ylim(-max_length, max_length)

        # Ensure equal scaling on both axes
        self.ax.set_aspect('equal', 'box')

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def add_joint(self, event):
        """
        Add a new joint to the robotic arm with default length and angle.

        Parameters:
            event (matplotlib.backend_bases.Event): The triggering event (unused).
        """
        # Append default length and angle
        self.lengths.append(1.0)  # Default length for new joint
        self.angles.append(np.pi / 2)  # Default angle of π/2 (vertical upwards)

        # Reinitialize UI to add new sliders
        self.init_ui()

        # Update the plot to reflect the new joint
        self.update_plot()

    def remove_joint(self, event):
        """
        Remove the last joint from the robotic arm.

        Parameters:
            event (matplotlib.backend_bases.Event): The triggering event (unused).
        """
        if len(self.lengths) > 1:
            # Remove last length and angle
            self.lengths.pop()
            self.angles.pop()

            # Reinitialize UI to remove sliders
            self.init_ui()

            # Update the plot to reflect removal
            self.update_plot()

if __name__ == '__main__':
    # Create an instance of RoboticArm and display the plot
    arm = RoboticArm()
    plt.show()
