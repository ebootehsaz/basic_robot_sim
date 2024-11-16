import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class RoboticArm:
    def __init__(self, lengths=None, angles=None):
        # Set default lengths and angles if not provided
        if lengths is None:
            self.lengths = [1.0, 0.5]
        else:
            self.lengths = lengths
        if angles is None:
            self.angles = [np.pi / 2, 0]
        else:
            self.angles = angles

        # Ensure lengths and angles lists are the same length
        if len(self.lengths) != len(self.angles):
            raise ValueError("Lengths and angles must be of the same length.")

        # Initialize figure and axes
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.5)  # Adjust for sliders
        self.line, = self.ax.plot([], [], 'o-', lw=2)

        self.sliders = []
        self.init_plot()
        self.init_ui()

    def init_plot(self):
        self.update_plot()

    def init_ui(self):
        self.clear_sliders()
        axcolor = 'lightgoldenrodyellow'
        base_y = 0.02
        slider_height = 0.02
        slider_spacing = 0.04

        # Create sliders for each joint
        for i in range(len(self.lengths)):
            # Length Slider
            ax_length = plt.axes([0.1, base_y, 0.65, slider_height], facecolor=axcolor)
            s_length = Slider(ax_length, f'Length {i+1}', 0.1, 2.0, valinit=self.lengths[i])
            s_length.on_changed(self.make_length_update(i))
            # Angle Slider
            ax_angle = plt.axes([0.1, base_y + slider_spacing, 0.65, slider_height], facecolor=axcolor)
            angle_range = (-np.pi, np.pi)
            s_angle = Slider(ax_angle, f'Angle {i+1}', angle_range[0], angle_range[1], valinit=self.angles[i])
            s_angle.on_changed(self.make_angle_update(i))
            # Store sliders
            self.sliders.append((s_length, s_angle))
            # Update position for next set of sliders
            base_y += (slider_height + slider_spacing + 0.02)

        # Add buttons for adding/removing joints
        self.add_button = Button(plt.axes([0.8, 0.05, 0.1, 0.04]), 'Add Joint', color=axcolor, hovercolor='0.975')
        self.add_button.on_clicked(self.add_joint)
        self.remove_button = Button(plt.axes([0.8, 0.01, 0.1, 0.04]), 'Remove Joint', color=axcolor, hovercolor='0.975')
        self.remove_button.on_clicked(self.remove_joint)

    def clear_sliders(self):
        # Remove existing sliders from the figure
        for s_length, s_angle in self.sliders:
            s_length.ax.remove()
            s_angle.ax.remove()
        self.sliders.clear()

    def make_length_update(self, index):
        # Helper function to capture index in slider callback
        def update(val):
            self.update_length(val, index)
        return update

    def make_angle_update(self, index):
        # Helper function to capture index in slider callback
        def update(val):
            self.update_angle(val, index)
        return update

    def add_joint(self, event):
        self.lengths.append(1.0)
        self.angles.append(0)
        self.init_ui()
        self.update_plot()

    def remove_joint(self, event):
        if len(self.lengths) > 1:
            self.lengths.pop()
            self.angles.pop()
            self.init_ui()
            self.update_plot()

    def update_length(self, val, index):
        self.lengths[index] = val
        self.update_plot()

    def update_angle(self, val, index):
        self.angles[index] = val
        self.update_plot()

    def update_angles(self, new_angles):
        if len(new_angles) != len(self.angles):
            raise ValueError("New angles list must be the same length as current angles.")
        self.angles = new_angles
        # Update slider positions
        for i, (s_length, s_angle) in enumerate(self.sliders):
            s_angle.set_val(self.angles[i])
        self.update_plot()

    def update_plot(self):
        x, y = 0, 0
        positions = [(x, y)]
        current_angle = 0  # Base reference angle
        for index, (length, angle) in enumerate(zip(self.lengths, self.angles)):
            current_angle += angle  # Cumulative angle
            x += length * np.cos(current_angle)
            y += length * np.sin(current_angle)
            positions.append((x, y))
        xs, ys = zip(*positions)
        self.line.set_data(xs, ys)
        # Adjust plot limits dynamically
        max_length = sum(self.lengths) * 1.5
        self.ax.set_xlim(-max_length, max_length)
        self.ax.set_ylim(-max_length, max_length)
        self.ax.set_aspect('equal', 'box')
        self.fig.canvas.draw_idle()
