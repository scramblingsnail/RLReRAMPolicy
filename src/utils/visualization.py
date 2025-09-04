import matplotlib.pyplot as plt
from matplotlib import animation


def display_gif(frames, save_path):
    r"""
    :param frames: list of env.render(); render_mode = 'rgb_array' when make.
    :param save_path: save_path
    :return:
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
    anim.save(save_path, writer='ffmpeg', fps=30)
