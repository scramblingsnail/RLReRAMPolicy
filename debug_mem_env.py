import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

from src.MemristorLab import OneTOneRSimulator, show_waveform
from src.MemristorENV import OneTOneREnv, random_onetoner_cell
from src.MemristorLab.numerical_models.mosfet import NMOSFET
from src.utils.display import display_gif


# mem_sim = OneTOneRSimulator(k_reset=4.13e-33, k_set=-4.13e-33, alpha_reset=25, alpha_set=25, delta_set=0.6,
#                             i_reset=-1.8, i_set=0.2, x_off=1.8, x_on=1.2, w_c=0.127, r_on=1, r_off=500, init_r=50,
#                             k=0.013, v_th=475, lamb=1e-5, i_s=1e-7, max_iteration=100, eps=1e-9,
#                             max_dt=1e-9, probe_dt=2e-7, difference_scheme='second_runge_kutta')


mem_sim = OneTOneRSimulator(k_reset=6e-33, k_set=-2e-33, alpha_reset=25, alpha_set=25, delta_set=0.6,
                            i_reset=-1.8, i_set=0.2, x_off=1.8, x_on=1.2, w_c=0.127, r_on=1, r_off=500, init_r=50,
                            k=0.013, v_th=475, lamb=1e-5, i_s=1e-7, max_iteration=100, eps=1e-9,
                            max_dt=1e-9, probe_dt=2e-7, difference_scheme='second_runge_kutta')

mos_model = NMOSFET(k=0.013, v_th=475, lamb=1e-5, i_s=1e-7)



def double_staircase_waveform(v_start, v_end, steps_num, step_time):
    d_v = (v_end - v_start) / (steps_num - 1)
    durations = [step_time for idx in range(steps_num * 2 - 1)]
    v_values = [d_v * idx + v_start for idx in range(steps_num)]
    v_values += v_values[-2::-1]
    return durations, v_values



def pulse_process(init_r, pulse_voltages, pulse_widths, gate_voltages):
    mem_sim.set_memristance(init_r)

    for idx, pulse_v in enumerate(pulse_voltages):
        probe_time, probe_i = mem_sim.evolution(t=pulse_widths[idx], v_r=pulse_v, v_s=0, v_g=gate_voltages[idx])
        mem_r = mem_sim.memristor.r
        print(f"Pulse V: {pulse_v}; Pulse width: {pulse_widths[idx]}; Gate V: {gate_voltages[idx]}; After g: {1 / mem_r} mS; After r: {mem_r}")


def sweeping_process(init_r, sweep_steps, sweep_time_step, sweeping_voltages, save_dir='./simu_figs'):
    r""" sweep vr from 0 to vr_end. sweeping voltages: [(vg, vr_end), ] """
    mem_sim.set_memristance(init_r)
    fig, ax = plt.subplots()
    for operation_idx, operation in enumerate(sweeping_voltages):
        vg, vr_end = operation
        durations, vg_values = double_staircase_waveform(vg, vg, steps_num=sweep_steps, step_time=sweep_time_step)
        _, vr_values = double_staircase_waveform(0, vr_end, steps_num=sweep_steps, step_time=sweep_time_step)
        _, vs_values = double_staircase_waveform(0, 0, steps_num=sweep_steps, step_time=sweep_time_step)

        t_pool = []
        i_pool = []
        current_t = 0
        for idx, duration in enumerate(durations):
            probe_time, probe_i = mem_sim.evolution(t=duration, v_r=vr_values[idx], v_s=vs_values[idx], v_g=vg_values[idx])
            print(len(probe_time), mem_sim.probe_dt, duration)
            print('x: {}; r: {}\n'.format(mem_sim.memristor.x,
                                          mem_sim.memristor.state_to_memristance(mem_sim.memristor.x)))
            t_pool.append(np.array(probe_time) + current_t)
            i_pool.append(np.array(probe_i))
            current_t += duration

        t_pool = np.concatenate(t_pool, axis=0)
        i_pool = np.concatenate(i_pool, axis=0)
        ax.plot(np.array(vr_values)[:len(vr_values) // 2 + 1] / 1e3, i_pool[:len(vr_values) // 2 + 1] / 1e3,
                label='{}'.format(operation_idx))
        ax.plot(np.array(vr_values)[len(vr_values) // 2:] / 1e3, i_pool[len(vr_values) // 2:] / 1e3, color='red')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'init_r-{:.1f}-sweep_steps-{:d}-sweep_time_step-{:.3f}.png'.format(init_r,
                                                                                                          sweep_steps,
                                                                                                          sweep_time_step))

    ax.set_xlabel('V')
    ax.set_ylabel('mA')
    # plt.yscale('log')
    ax.legend()
    plt.savefig(save_path)



def set_pulses():
    init_r = 20
    pulse_time = 2e-8
    gate_voltages = [
        0.59e3,
        # 0.55e3, 0.56e3, 0.57e3, 0.58e3, 0.59e3, 0.59e3, 0.6e3, 0.61e3, 0.62e3, 0.63e3, 0.64e3, 0.65e3,
        # 0.592e3, 0.595e3, 0.597e3,
        # 0.6e3, 0.61e3, 0.62e3, 0.63e3, 0.64e3,
        # 0.65e3, 0.66e3, 0.67e3, 0.68e3, 0.69e3,
    ]
    pulse_widths = [pulse_time for i in range(len(gate_voltages))]
    pulse_voltages = [3e3 for i in range(len(gate_voltages))]
    pulse_process(
        init_r=init_r,
        pulse_voltages=pulse_voltages,
        pulse_widths=pulse_widths,
        gate_voltages=gate_voltages,
    )

    print("Single pulse:")

    for idx in range(len(pulse_voltages)):
        pulse_process(
            init_r=init_r,
            pulse_voltages=pulse_voltages[idx: idx+1],
            pulse_widths=pulse_widths[idx: idx+1],
            gate_voltages=gate_voltages[idx: idx+1],
        )


def reset_pulses():
    init_r = 2.5
    pulse_time = 2e-8
    pulse_voltages = [
        -0.8e3, -0.9e3, -1.0e3, -1.1e3, -1.2e3, -1.3e3, -1.4e3, -1.5e3, -1.6e3, -1.7e3, -1.8e3, -1.9e3, -2e3, -2.1e3, -2.2e3, -2.3e3,
        -2.4e3, -2.5e3, -2.6e3, -2.7e3, -2.8e3, -2.9e3, -3e3, -3.1e3, -3.2e3, -3.3e3, -3.4e3,
    ]
    pulse_widths = [pulse_time for i in range(len(pulse_voltages))]
    gate_voltages = [1e3 for i in range(len(pulse_voltages))]
    pulse_process(
        init_r=init_r,
        pulse_voltages=pulse_voltages,
        pulse_widths=pulse_widths,
        gate_voltages=gate_voltages,
    )

    print("Single pulse:")

    for idx in range(len(pulse_voltages)):
        pulse_process(
            init_r=init_r,
            pulse_voltages=pulse_voltages[idx: idx+1],
            pulse_widths=pulse_widths[idx: idx+1],
            gate_voltages=gate_voltages[idx: idx+1],
        )



def set_process_by_sweeping_vg():
    sweep_steps = 51
    sweep_time_step = 2e-8 #2e-7
    init_r = 50

    # # vg, vr_end
    # sweeping_voltages = [(0.57e3, 3e3), (0.571e3, 3e3), (0.572e3, 3e3), (0.5722e3, 3e3), (0.5724e3, 3e3),
    #                      (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3),
    #                      (0.5727e3, 3e3), (0.5729e3, 3e3), (0.5732e3, 3e3), (0.5735e3, 3e3), (0.5737e3, 3e3),
    #                      (0.5739e3, 3e3), (0.575e3, 3e3), (0.576e3, 3e3), (0.577e3, 3e3), (0.578e3, 3e3),
    #                      (0.579e3, 3e3), (0.58e3, 3e3), (0.581e3, 3e3), (0.583e3, 3e3), (0.585e3, 3e3),
    #                      (0.587e3, 3e3), (0.59e3, 3e3), (0.593e3, 3e3), (0.595e3, 3e3), (0.598e3, 3e3), (0.6e3, 3e3),
    #                      (0.602e3, 3e3), (0.605e3, 3e3), (0.607e3, 3e3), (0.609e3, 3e3), (0.612e3, 3e3), (0.615e3, 3e3),
    #                      (0.618e3, 3e3), (0.62e3, 3e3), (0.625e3, 3e3),
    #                      ]


    sweeping_voltages = [(0.57e3, 3e3), (0.571e3, 3e3), (0.572e3, 3e3), (0.5722e3, 3e3), (0.5724e3, 3e3),
                         (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3), (0.5724e3, 3e3),
                         (0.5727e3, 3e3), (0.5729e3, 3e3), (0.5732e3, 3e3), (0.5735e3, 3e3), (0.5737e3, 3e3),
                         (0.5739e3, 3e3), (0.575e3, 3e3), (0.576e3, 3e3), (0.577e3, 3e3), (0.578e3, 3e3),
                         (0.579e3, 3e3), (0.58e3, 3e3), (0.581e3, 3e3), (0.583e3, 3e3), (0.585e3, 3e3),
                         (0.587e3, 3e3), (0.59e3, 3e3), (0.593e3, 3e3), (0.595e3, 3e3), (0.598e3, 3e3), (0.6e3, 3e3),
                         (0.602e3, 3e3), (0.605e3, 3e3), (0.607e3, 3e3), (0.609e3, 3e3), (0.612e3, 3e3), (0.615e3, 3e3),
                         (0.618e3, 3e3), (0.62e3, 3e3), (0.625e3, 3e3),
                         ]

    # sweeping_voltages = [(0.568e3, 5.9e3)]
    sweeping_process(init_r=init_r, sweep_steps=sweep_steps, sweep_time_step=sweep_time_step,
                     sweeping_voltages=sweeping_voltages)


def reset_process_by_sweeping_vr():
    sweep_steps = 51
    sweep_time_step = 2e-8 #2e-7
    init_r = 2
    sweeping_voltages = [(2e3, -1.39e3), (2e3, -1.4e3), (2e3, -1.5e3), (2e3, -1.6e3), (2e3, -1.7e3),
                         (2e3, -1.8e3), (2e3, -1.9e3), (2e3, -2e3), (2e3, -2.1e3), (2e3, -2.2e3), (2e3, -2.3e3),
                         (2e3, -2.4e3), (2e3, -2.5e3), (2e3, -2.6e3), (2e3, -2.7e3), (2e3, -2.8e3), (2e3, -2.9e3),
                         (2e3, -3.0e3), (2e3, -3.1e3), (2e3, -3.2e3), (2e3, -3.3e3), (2e3, -3.4e3),
                         ]
    sweeping_process(init_r=init_r, sweep_steps=sweep_steps, sweep_time_step=sweep_time_step,
                     sweeping_voltages=sweeping_voltages)


def pulse_write(pulse_time, vg, vds, times):
    for idx in range(times):
        probe_time, probe_i = mem_sim.evolution(t=pulse_time, v_r=vds, v_s=0, v_g=vg)
        print('x: {}; r: {}\n'.format(mem_sim.memristor.x,
                                      mem_sim.memristor.state_to_memristance(mem_sim.memristor.x)))




if __name__ == '__main__':

    # mem_cfg_path = r'./src/MemristorENV/default_memristor_cfg.yaml'
    # transistor_cfg_path = r'./src/MemristorENV/default_transistor_cfg.yaml'
    # simulate_cfg_path = r'./src/MemristorENV/default_simulate_cfg.yaml'
    # array_cfg_path = r'./src/MemristorENV/default_array_cfg.yaml'
    #
    # with open(mem_cfg_path, 'r') as f:
    #     memristor_para_dict = yaml.safe_load(f)
    #
    # with open(transistor_cfg_path, 'r') as f:
    #     transistor_para_dict = yaml.safe_load(f)
    #
    # with open(simulate_cfg_path, 'r') as f:
    #     simulate_para_dict = yaml.safe_load(f)
    #
    # with open(array_cfg_path, 'r') as f:
    #     array_para_dict = yaml.safe_load(f)
    #
    #
    # cell = random_onetoner_cell(array_para_dict=array_para_dict, memristor_para_dict=memristor_para_dict,
    #                             transistor_para_dict=transistor_para_dict, simulate_para_dict=simulate_para_dict)
    # pulse_time = 2e-8
    #
    # import time
    # cell.memristor.set_r(84.33)
    # t1 = time.time()
    #
    # cell.evolution(t=pulse_time, v_s=0, v_r=6.61e3, v_g=580)
    # t2 = time.time()
    # print(cell.memristor.r)
    # print(t2 - t1)


    # mem_env = OneTOneREnv(array_para_dict=array_para_dict, memristor_para_dict=memristor_para_dict,
    #                       transistor_para_dict=transistor_para_dict, simulate_para_dict=simulate_para_dict)
    #
    # states, _ = mem_env.reset()
    # print(1 / states)
    # actions = np.array([[0.6, 4], [1, -2], [0.6, 3], [0.6, 3]])
    #
    # show_frames = []
    # for i in range(10):
    #     states, rewards, terminals, _, _ = mem_env.step(actions)
    #     print(rewards)
    #     frames = mem_env.render()
    #     show_frames += frames
    # print('after: ')
    # print(1 / states)
    # display_gif(frames=show_frames, save_path='./mem_test.gif', fps=1)

    # write(2e-8)

    set_process_by_sweeping_vg()
    # set_pulses()
    # reset_process_by_sweeping_vr()

    # set_pulses()
    # reset_pulses()


