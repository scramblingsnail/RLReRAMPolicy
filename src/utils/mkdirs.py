import os
import os.path as path
import time


def make_dirs(env_flag: str):
    project_dir = __file__
    for idx in range(3):
        project_dir = path.dirname(project_dir)

    current_time = time.strftime('%m-%d-%H-%M')
    # data
    eval_data_dir = os.path.join(project_dir, 'eval_data/{}'.format(env_flag))
    display_data_dir = os.path.join(project_dir, 'display_data/{}'.format(env_flag))
    if not os.path.exists(eval_data_dir):
        os.makedirs(eval_data_dir)
    if not os.path.exists(display_data_dir):
        os.makedirs(display_data_dir)

    # checkpoint
    checkpoint_dir = os.path.join(project_dir, "checkpoints/{}/{}".format(env_flag, current_time))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # visualize
    root_eval_plot_dir = os.path.join(project_dir, 'figs/eval_plots/{}'.format(env_flag))
    root_gif_dir = os.path.join(project_dir, 'figs/gifs/{}'.format(env_flag))

    eval_plot_dir = os.path.join(root_eval_plot_dir, current_time)
    gif_dir = os.path.join(root_gif_dir, current_time)
    if not os.path.exists(eval_plot_dir):
        os.makedirs(eval_plot_dir)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    return eval_data_dir, display_data_dir, eval_plot_dir, gif_dir, checkpoint_dir, current_time