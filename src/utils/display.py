import h5py
import matplotlib.pyplot as plt
import logging
import numpy as np
import os

from matplotlib import animation


def display_gif(frames, save_path, title=None, interval: float=0.03, dpi=300):
    r"""

    Args:
        frames: list of env.render(); render_mode = 'rgb_array' when make.
        save_path: save_path
        title:
        interval:
        dpi:

    Returns:
        None
    """
    plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    if title is not None:
        plt.suptitle(title)

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval * 1000)
    anim.save(save_path, writer='ffmpeg', dpi=dpi)


def saved_data_to_gif(evaluate_data_path, display_data_path, gif_dir, delete_display_data = True,
                      delete_eval_data = False, save_num = None, max_frames: int = 400, interval: float = 0.03):
    file_flag = evaluate_data_path.split('/')[-1].split('.')[0]

    evaluate_data_f = h5py.File(evaluate_data_path, 'r')
    count = 0


    with h5py.File(display_data_path, 'r') as file:
        name_list = list(file.keys())
        if save_num is not None and save_num < len(name_list):
            name_list = np.random.choice(name_list, save_num, replace=False)
        for name in name_list:
            step_prefix = name.split('_')[0]
            avg_reward_name = step_prefix + '_average_reward'
            avg_step_name = step_prefix + '_average_step'
            avg_reward = evaluate_data_f[avg_reward_name][0]
            avg_step = evaluate_data_f[avg_step_name][0]
            title = '{} avg reward: {:.4f}, avg step: {:.2f}'.format(step_prefix, avg_reward, avg_step)
            logging.info('Saving evaluate data to gif: {}\tdetails: {}'.format(name, title))
            each_pics = file[name][:]
            gif_path = '{}/{}_{}.gif'.format(gif_dir, file_flag, name)

            if not os.path.exists(gif_path) and each_pics.shape[0] < max_frames:
                print('\t||Converter: Generating {}'.format(gif_path))
                display_gif(each_pics, gif_path, title=title, interval=interval)
            count += 1
            if save_num is not None and count >= save_num:
                break

    evaluate_data_f.close()
    if delete_display_data:
        os.remove(display_data_path)
    if delete_eval_data:
        os.remove(evaluate_data_path)


def plot_reward_trajectory(evaluate_data_path, fig_dir):
    data_flag = evaluate_data_path.split('/')[-1].split('_')[0]
    evaluate_data_f = h5py.File(evaluate_data_path, 'r')
    print('\t||Plotter: plotting rewards and dones.')
    for name in evaluate_data_f.keys():
        step_prefix = name.split('_')[0]
        reward_name = step_prefix + '_eval_rewards'
        terminal_name = step_prefix + '_eval_terminals'
        rewards = evaluate_data_f[reward_name][:]
        terminals = evaluate_data_f[terminal_name][:]
        terminals = terminals * np.max(rewards)

        num = rewards.shape[0]
        fig_path = fig_dir + '/{}_{}.png'.format(data_flag, step_prefix)

        row_num = col_num = int(np.ceil(np.sqrt(num)))

        fig, axes = plt.subplots(row_num, col_num)
        if num > 1:
            for idx in range(num):
                row_idx = idx // row_num
                col_idx = idx % row_num
                axes[row_idx][col_idx].plot(rewards[idx], label='sampler_{}_rewards'.format(idx))
                axes[row_idx][col_idx].plot(terminals[idx], label='sampler_{}_done'.format(idx), alpha=0.2)
                axes[row_idx][col_idx].legend()
        else:
            for idx in range(num):
                axes.plot(rewards[idx], label='sampler_{}_rewards'.format(idx))
                axes.plot(terminals[idx], label='sampler_{}_done'.format(idx), alpha=0.2)
                axes.legend()
        fig.suptitle(step_prefix)
        plt.savefig(fig_path, dpi=200)
        plt.close()
