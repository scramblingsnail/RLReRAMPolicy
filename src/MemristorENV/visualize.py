import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import io
import cv2


class MemShower:
    def __init__(self, envs_num: int, vg_low, vg_high, vds_pos, vds_neg, g_low, g_high,
                 program_g_low, program_g_high, writing_tolerance):
        self.envs_num = envs_num
        self.h = 1
        self.w = 3
        self.arrow_len = 1
        self.vds_arrow_len = 2
        self.g_color = 'purple'
        self.vg_low = vg_low
        self.vg_high = vg_high
        self.vds_pos = vds_pos
        self.vds_neg = vds_neg
        self.g_low = g_low
        self.g_high = g_high
        self.program_g_low = program_g_low
        self.program_g_high = program_g_high
        self.writing_tolerance = writing_tolerance
        self.fig_size = (3, 2)
        self.dpi=300

    def plot_background(self, ax, target_g_low, target_g_high):
        r"""
        plot the background, including the mosfet, the memristor, the target area.
        """
        target_low = (target_g_low - self.g_low) / (self.g_high - self.g_low)
        target_high = (target_g_high - self.g_low) / (self.g_high - self.g_low)
        mem_rect = patches.Rectangle((0, 0), self.w, 2 * self.h,
                                     fill=True, linewidth=2, facecolor='grey')
        ax.add_patch(mem_rect)
        ax.set_xlim(-8, self.w*3)
        ax.set_ylim(-1, self.h * 8)
        ax.set_xticks([])
        ax.set_yticks([])

        mos_lines = [((-3, self.h), (-2, self.h)), ((-2, self.h), (-2, self.h + 0.5)),
                     ((-2, self.h + 0.5), (-1, self.h + 0.5)), ((-1, self.h + 0.5), (-1, self.h)),
                     ((-1, self.h), (0, self.h))]

        mos_xys = [list(zip(line[0], line[1])) for line in mos_lines]
        for xy in mos_xys:
            ax.plot(xy[0], xy[1], color='black', linewidth=0.5)

        gate_rect = patches.Rectangle((-2, self.h + 1), 1, 1,
                                      fill=True, linewidth=0.1, facecolor='grey')
        ax.add_patch(gate_rect)

        # plot target
        ax.arrow(self.w * (1 - target_low), 2 * self.h + self.arrow_len, 0, -self.arrow_len, width=0.02, color='brown')
        ax.arrow(self.w * (1 - target_high), 2 * self.h + self.arrow_len, 0, -self.arrow_len, width=0.02, color='brown')

        # plot program range
        program_low_x = (1 - (self.program_g_low - self.g_low) / (self.g_high - self.g_low)) * self.w
        program_high_x = (1 - (self.program_g_high - self.g_low) / (self.g_high - self.g_low)) * self.w
        ax.plot((program_low_x, program_low_x), (0, 2 * self.h), linestyle='--', color='red')
        ax.plot((program_high_x, program_high_x), (0, 2 * self.h), linestyle='--', color='red')

    def plot_state(self, ax, g, g_target_low, g_target_high):
        r"""
        plot the state, including current g, zoom-in current g
        """
        g_x = (g - self.g_low) / (self.g_high - self.g_low)
        g_rect = patches.Rectangle((self.w * (1 - g_x), 0), self.w * g_x, 2 * self.h,
                                   fill=True, linewidth=0.1, facecolor=self.g_color)
        ax.add_patch(g_rect)

        # zoom in:
        show_sigma_times = 3
        unit_x = self.w / (2 * show_sigma_times)
        target_left_arrow = unit_x * (show_sigma_times - 1)
        target_right_arrow = unit_x * (show_sigma_times + 1)
        zoom_rect_y = 2 * self.h + self.arrow_len * 2
        zoom_in_rect = patches.Rectangle((0, zoom_rect_y), self.w, 2 * self.h,
                                         fill=True, linewidth=2, facecolor='red')
        ax.add_patch(zoom_in_rect)
        ax.arrow(target_left_arrow, zoom_rect_y + 2 * self.h + self.arrow_len, 0, -self.arrow_len, width=0.02, color='brown')
        ax.arrow(target_right_arrow, zoom_rect_y + 2 * self.h + self.arrow_len, 0, -self.arrow_len, width=0.02, color='brown')

        show_g_low = g_target_low - (show_sigma_times - 1) * (g_target_high - g_target_low) / 2
        show_g_high = g_target_high + (show_sigma_times - 1) * (g_target_high - g_target_low) / 2

        if show_g_low <= g <= show_g_high:
            zoom_g_x = (g - show_g_low) / (show_g_high - show_g_low)
            zoom_g_rect = patches.Rectangle((self.w * (1 - zoom_g_x), zoom_rect_y), self.w * zoom_g_x, 2 * self.h,
                                            fill=True, linewidth=0.1, facecolor=self.g_color)
            ax.add_patch(zoom_g_rect)

        ax.text(self.w / 2, self.h, '{:.2f}kOhm'.format(1 / g), fontsize=4, ha='center', va='bottom', color='black')
        ax.text(self.w / 2, self.h * 3, 'target: {:.2f}kOhm'.format(1 / g_target_low), fontsize=4, ha='center',
                va='bottom', color='black')

        if g_target_low <= g <= g_target_high:
            ax.text(self.w / 2, self.h * 5, 'DONE!', fontsize=8, ha='center', va='bottom', color='green')

        if g < self.program_g_low or g > self.program_g_high:
            ax.text(self.w / 2, self.h * 5, 'FAIL!', fontsize=8, ha='center', va='bottom', color='yellow')

    def plot_action(self, ax, vds, vg):
        r"""
        plot the actions, including the vg, vds.
        """
        gate_y = self.h + 1 + (vg - self.vg_low) / (self.vg_high - self.vg_low)
        ax.plot((-2, -1), (gate_y, gate_y), color='green', linewidth=1)
        ax.text(-1.5, self.h + 2, 'vg={:.2f}V'.format(vg), fontsize=6, ha='center')

        if vds > 0:
            vds_x = vds / self.vds_pos
            ax.arrow(self.w + self.vds_arrow_len, self.h, -vds_x, 0, width=0.2, color='red')
            ax.text(self.w + self.vds_arrow_len / 2, self.h, 'vds={:.2f}V'.format(vds), ha='left', va='bottom', fontsize=6)
        else:
            vds_x = vds / self.vds_neg
            ax.arrow(-3 - self.vds_arrow_len, self.h, vds_x, 0, width=0.2, color='blue')
            ax.text(-3 - self.vds_arrow_len / 2, self.h, 'vds={:.2f}V'.format(vds), ha='right', va='bottom', fontsize=6)

    def plot_reward(self, ax, reward):
        r"""
        show the reward.
        """
        ax.set_title('Reward: {:.2f}'.format(reward))

    def show_action_frame(self, actions, last_states):
        r"""
        plot the frame that shows the action.

        Inputs:
            actions(np.ndarray): the applied actions, size -- (envs_num, 2), in dim1 -- (vg, vr), unit: mV
            last_states(np.ndarray): states before applying actions,
                size -- (envs_num), in dim1 -- (current_g, target_g), unit: mS.
        """

        row_num = int(np.sqrt(self.envs_num))
        col_num = int(np.ceil(self.envs_num / row_num))
        figs, axes = plt.subplots(row_num, col_num, figsize=(self.fig_size[0] * col_num, self.fig_size[1] * row_num))
        if row_num == 1 and col_num == 1:
            axes = [[axes]]

        for env_idx in range(self.envs_num):
            row_idx = env_idx // row_num
            col_idx = env_idx % row_num
            target_g = last_states[env_idx, 1]
            tolerance = self.writing_tolerance
            g_target_low, g_target_high = target_g - tolerance, target_g + tolerance
            self.plot_background(ax=axes[row_idx][col_idx], target_g_low=g_target_low, target_g_high=g_target_high)
            self.plot_state(ax=axes[row_idx][col_idx], g=last_states[env_idx][0], g_target_low=g_target_low,
                            g_target_high=g_target_high)
            self.plot_action(ax=axes[row_idx][col_idx], vds=actions[env_idx][1], vg=actions[env_idx][0])

        frame = self.get_img_from_fig(fig=figs)
        plt.close(figs)
        return frame

    def get_img_from_fig(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show_result_state(self, states, rewards):
        r"""
        plot the result state.
        """
        row_num = int(np.sqrt(self.envs_num))
        col_num = int(np.ceil(self.envs_num / row_num))
        figs, axes = plt.subplots(row_num, col_num, figsize=(self.fig_size[0] * col_num, self.fig_size[1] * row_num))
        if row_num == 1 and col_num == 1:
            axes = [[axes]]

        for env_idx in range(self.envs_num):
            row_idx = env_idx // row_num
            col_idx = env_idx % row_num
            target_g = states[env_idx][1]
            tolerance = self.writing_tolerance
            g_target_low, g_target_high = target_g - tolerance, target_g + tolerance
            self.plot_background(ax=axes[row_idx][col_idx], target_g_low=g_target_low, target_g_high=g_target_high)
            self.plot_state(ax=axes[row_idx][col_idx], g=states[env_idx][0], g_target_low=g_target_low,
                            g_target_high=g_target_high)
            self.plot_reward(ax=axes[row_idx][col_idx], reward=rewards[env_idx])
        frame = self.get_img_from_fig(fig=figs)
        plt.close(figs)
        return frame
