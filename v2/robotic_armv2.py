import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import MultipleLocator
from functools import partial

class Joint:
    """
    Represents a single joint in the robotic arm.
    """
    def __init__(self, angle=0.0, min_angle=0.0, max_angle=np.pi):
        """
        Initialize the Joint instance.

        Parameters:
            angle (float): The initial angle of the joint in radians.
            min_angle (float): The minimum allowed angle.
            max_angle (float): The maximum allowed angle.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.set_angle(angle)

    def set_angle(self, angle):
        """
        Set the angle of the joint, constrained within min and max angles.

        Parameters:
            angle (float): The new angle in radians.
        """
        self.angle = np.clip(angle, self.min_angle, self.max_angle)

    def get_angle(self):
        """
        Get the current angle of the joint.

        Returns:
            float: The current angle in radians.
        """
        return self.angle

class Module:
    """
    Represents a module consisting of two joints separated by a fixed offset.
    """
    def __init__(self, angle=0.0, main_length=1.0, offset=0.15):
        """
        Initialize the Module instance.

        Parameters:
            angle (float): The initial overall angle of the module in radians.
            main_length (float): The length of the main segment in the module.
            offset (float): The fixed length between the two joints.
        """
        self.main_length = main_length
        self.offset = offset
        self.joint1 = Joint(angle=0.0)
        self.joint2 = Joint(angle=0.0)
        self.set_angle(angle)

    def set_angle(self, theta):
        """
        Set the overall angle of the module by setting both joints to theta / 2.

        Parameters:
            theta (float): The overall angle of the module in radians.
        """
        half_theta = theta / 2
        self.joint1.set_angle(half_theta)
        self.joint2.set_angle(half_theta)

    def get_angle(self):
        """
        Get the overall angle of the module.

        Returns:
            float: The overall angle in radians.
        """
        return self.joint1.get_angle() + self.joint2.get_angle()

    def get_joint_angles(self):
        """
        Get the angles of the two joints in the module.

        Returns:
            list of float: Angles of joint1 and joint2 in radians.
        """
        return [self.joint1.get_angle(), self.joint2.get_angle()]

class RoboticArm:
    """
    Represents a 2D planar robotic arm composed of modules.
    """
    def __init__(self, modules=None, fig=None, ax=None, include_ui=True):
        """
        Initialize the RoboticArm instance with given modules.

        Parameters:
            modules (list of Module, optional): List of modules to include in the arm.
                Defaults to one module with default parameters.
            fig (matplotlib.figure.Figure, optional): Existing figure to plot on.
            ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
            include_ui (bool, optional): Whether to include sliders and buttons.
        """
        if modules is None:
            self.modules = [Module(angle=np.pi)]  # Default to one module with theta=pi (joint angles pi/2 each)
        else:
            self.modules = modules

        # Use provided figure and axes or create new ones
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.35)
        else:
            self.fig = fig
            self.ax = ax

        # Initialize the line object that will represent the robotic arm
        self.line, = self.ax.plot([], [], 'o-', lw=2)

        # List to hold slider widgets
        self.sliders = []
        self.include_ui = include_ui

        # Initialize the plot and user interface
        self.init_plot()
        if self.include_ui:
            self.init_ui()

    def init_plot(self):
        """
        Initialize the plot by updating it with the initial positions.
        """
        self.update_plot()

    def init_ui(self):
        """
        Initialize the user interface components including sliders and buttons.
        """
        # Clear existing sliders
        self.clear_sliders()

        # Slider appearance
        axcolor = 'lightgoldenrodyellow'
        base_y = 0.02
        slider_height = 0.03
        slider_spacing = 0.04

        # Create sliders for each module
        for i, module in enumerate(self.modules):
            # Create axes for angle slider
            ax_angle = plt.axes([0.1, base_y, 0.65, slider_height], facecolor=axcolor)
            # Initialize slider with module's current overall angle
            s_angle = Slider(
                ax_angle,
                f'Module {i+1} Angle',
                0.0,
                2 * np.pi,
                valinit=module.get_angle(),
                valstep=0.01
            )

            # Bind the slider to the module's update method
            s_angle.on_changed(partial(self.update_module_angle, index=i))

            self.sliders.append(s_angle)

            base_y += slider_height + slider_spacing

        # Create 'Add Module' button
        self.add_button = Button(
            plt.axes([0.8, 0.05, 0.15, 0.04]),
            'Add Module',
            color=axcolor,
            hovercolor='0.975'
        )
        self.add_button.on_clicked(self.add_module)

        # Create 'Remove Module' button
        self.remove_button = Button(
            plt.axes([0.8, 0.01, 0.15, 0.04]),
            'Remove Module',
            color=axcolor,
            hovercolor='0.975'
        )
        self.remove_button.on_clicked(self.remove_module)

    def clear_sliders(self):
        """
        Remove all existing slider widgets from the figure.
        """
        for slider in self.sliders:
            slider.ax.remove()
        self.sliders.clear()

    def update_module_angle(self, val, index):
        """
        Update the angle of a module and refresh the plot.

        Parameters:
            val (float): The new angle value in radians.
            index (int): The index of the module to update.
        """
        self.modules[index].set_angle(val)
        self.update_plot()

    def update_plot(self, base_position=(0, 0)):
        """
        Update the plot of the robotic arm based on current module angles.

        Parameters:
            base_position (tuple of float, optional): The starting position (x, y).
        """
        x, y = base_position
        positions = [(x, y)]  # List to store joint positions
        current_angle = np.pi / 2  # Start from vertical upwards position

        # Calculate positions of each joint
        for module in self.modules:
            joint_angles = module.get_joint_angles()
            segment_lengths = [module.offset, module.main_length]

            for angle, length in zip(joint_angles, segment_lengths):
                # Adjust current angle for the joint
                # Subtract π/2 because the initial angle is set to vertical (π/2)
                # This ensures that angle sliders correspond to the angle between segments
                current_angle += angle - (np.pi / 2)

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
        total_length = sum([m.offset + m.main_length for m in self.modules])
        max_length = total_length * 1.1  # Extend limits for better visibility
        self.ax.set_xlim(-max_length, max_length)
        self.ax.set_ylim(-max_length, max_length)

        # Set gridline spacing
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))

        # Ensure equal scaling on both axes
        self.ax.set_aspect('equal', 'box')

        # Enable grid lines
        self.ax.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5)

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def add_module(self, event):
        """
        Add a new module to the robotic arm with default parameters.

        Parameters:
            event (matplotlib.backend_bases.Event): The triggering event (unused).
        """
        # Create a new module with default angle pi (joint angles pi/2 each)
        new_module = Module(angle=np.pi)
        self.modules.append(new_module)
        self.init_ui()
        self.update_plot()

    def remove_module(self, event):
        """
        Remove the last module from the robotic arm.

        Parameters:
            event (matplotlib.backend_bases.Event): The triggering event (unused).
        """
        if len(self.modules) > 1:
            self.modules.pop()
            self.init_ui()
            self.update_plot()

if __name__ == '__main__':
    arm = RoboticArm()
    plt.show()
