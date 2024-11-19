import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class RoboticArm:
    def __init__(self, lengths=None, angles=None):
        if lengths is None:
            self.lengths = [1.0, 0.5]  # Default lengths of each joint
        else:
            self.lengths = lengths

        if angles is None:
            self.angles = [np.pi / 2, np.pi / 2]  # Set all angles to pi/2 by default
        else:
            self.angles = [self.constrain_angle(angle) for angle in angles]

        if len(self.lengths) != len(self.angles):
            raise ValueError("Lengths and angles must be of the same length.")

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.3)
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.sliders = []
        self.init_plot()
        self.init_ui()

    def constrain_angle(self, angle):
        """Constrain angle within acceptable range [0, pi]."""
        return np.clip(angle, 0, np.pi)

    def init_plot(self):
        self.update_plot()

    def init_ui(self):
        self.clear_sliders()
        axcolor = 'lightgoldenrodyellow'
        base_y = 0.02
        slider_height = 0.03
        slider_spacing = 0.04

        for i in range(len(self.lengths)):
            ax_length = plt.axes([0.1, base_y, 0.65, slider_height], facecolor=axcolor)
            s_length = Slider(ax_length, f'Length {i+1}', 0.1, 2.0, valinit=self.lengths[i])
            s_length.on_changed(lambda val, idx=i: self.update_length(val, idx))
            
            ax_angle = plt.axes([0.1, base_y + slider_spacing, 0.65, slider_height], facecolor=axcolor)
            s_angle = Slider(ax_angle, f'Angle {i+1}', 0, np.pi, valinit=self.angles[i])
            s_angle.on_changed(lambda val, idx=i: self.update_angle(val, idx))
            
            self.sliders.append((s_length, s_angle))
            base_y += 2 * (slider_height + slider_spacing)

        self.add_button = Button(plt.axes([0.8, 0.05, 0.1, 0.04]), 'Add Joint', color=axcolor, hovercolor='0.975')
        self.add_button.on_clicked(self.add_joint)
        self.remove_button = Button(plt.axes([0.8, 0.01, 0.1, 0.04]), 'Remove Joint', color=axcolor, hovercolor='0.975')
        self.remove_button.on_clicked(self.remove_joint)

    def clear_sliders(self):
        for s_length, s_angle in self.sliders:
            s_length.ax.remove()
            s_angle.ax.remove()
        self.sliders.clear()

    def update_length(self, val, index):
        self.lengths[index] = val
        self.update_plot()

    def update_angle(self, val, index):
        self.angles[index] = self.constrain_angle(val)
        self.update_plot()

    def update_angles(self, new_angles):
        if len(new_angles) != len(self.angles):
            raise ValueError("New angles list must be the same length as current angles.")
        self.angles = [self.constrain_angle(angle) for angle in new_angles]
        # Update slider positions
        for i, (s_length, s_angle) in enumerate(self.sliders):
            s_angle.set_val(self.angles[i])
        self.update_plot()

    def update_plot(self):
        x, y = 0, 0
        positions = [(x, y)]
        current_angle = np.pi / 2  # Start from vertical up position
        for length, angle in zip(self.lengths, self.angles):
            current_angle += angle - np.pi / 2  # Adjust each angle by subtracting pi/2
            x += length * np.cos(current_angle)
            y += length * np.sin(current_angle)
            positions.append((x, y))
        xs, ys = zip(*positions)
        self.line.set_data(xs, ys)
        max_length = sum(self.lengths) * 1.5
        self.ax.set_xlim(-max_length, max_length)
        self.ax.set_ylim(-max_length, max_length)
        self.ax.set_aspect('equal', 'box')
        self.fig.canvas.draw_idle()

    def add_joint(self, event):
        self.lengths.append(1.0)
        self.angles.append(np.pi / 2)  # Default new joint to pi/2
        self.init_ui()
        self.update_plot()

    def remove_joint(self, event):
        if len(self.lengths) > 1:
            self.lengths.pop()
            self.angles.pop()
            self.init_ui()
            self.update_plot()

if __name__ == '__main__':
    arm = RoboticArm()
    plt.show()
