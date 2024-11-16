import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class RoboticArm:
    def __init__(self):
        self.lengths = [1.0, 0.5]  # Default lengths
        self.angles = [np.pi / 2, 0]  # Default angles, first at pi/2, second at 0
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.5)  # Adjust the bottom to give more space for sliders
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.sliders = []
        self.init_plot()
        self.init_ui()

    def init_plot(self):
        self.update_plot()  # Initial plot draw

    def init_ui(self):
        self.clear_sliders()
        axcolor = 'lightgoldenrodyellow'
        base_y = 0.02
        for i in range(len(self.lengths)):
            ax_length = plt.axes([0.1, base_y, 0.65, 0.02], facecolor=axcolor)
            if i == 0:
                ax_angle = plt.axes([0.1, base_y + 0.03, 0.65, 0.02], facecolor=axcolor)
                s_angle = Slider(ax_angle, f'Angle {i+1}', 0, np.pi, valinit=self.angles[i])
            else:
                ax_angle = plt.axes([0.1, base_y + 0.03, 0.65, 0.02], facecolor=axcolor)
                s_angle = Slider(ax_angle, f'Angle {i+1}', -np.pi/2, np.pi/2, valinit=self.angles[i])

            s_length = Slider(ax_length, f'Length {i+1}', 0.1, 2.0, valinit=self.lengths[i])
            s_length.on_changed(lambda val, i=i: self.update_length(val, i))
            s_angle.on_changed(lambda val, i=i: self.update_angle(val, i))
            self.sliders.append((s_length, s_angle))
            base_y += 0.07  # Increase spacing to prevent overlap

        self.add_button = Button(plt.axes([0.8, 0.05, 0.1, 0.04]), 'Add Joint', color=axcolor, hovercolor='0.975')
        self.add_button.on_clicked(self.add_joint)
        self.remove_button = Button(plt.axes([0.8, 0.01, 0.1, 0.04]), 'Remove Joint', color=axcolor, hovercolor='0.975')
        self.remove_button.on_clicked(self.remove_joint)

    def clear_sliders(self):
        for slider_pair in self.sliders:
            slider_pair[0].ax.remove()
            slider_pair[1].ax.remove()
        self.sliders.clear()

    def add_joint(self, event):
        self.lengths.append(1.0)
        # Default new joint angles to 0 with range -pi/2 to pi/2
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

    def update_plot(self):
        x, y = 0, 0
        positions = [(x, y)]
        current_angle = 0  # Start with a 0 angle for the base reference
        for index, (length, angle) in enumerate(zip(self.lengths, self.angles)):
            if index == 0:
                current_angle = angle  # Absolute angle for the first joint
            else:
                current_angle += angle  # Relative angles for subsequent joints
            x += length * np.cos(current_angle)
            y += length * np.sin(current_angle)
            positions.append((x, y))
        xs, ys = zip(*positions)
        self.line.set_data(xs, ys)
        self.ax.set_xlim(-sum(self.lengths)*1.5, sum(self.lengths)*1.5)
        self.ax.set_ylim(-sum(self.lengths)*1.5, sum(self.lengths)*1.5)
        self.ax.set_aspect('equal', 'box')
        self.fig.canvas.draw_idle()

arm = RoboticArm()
plt.show()
