import time
import numpy as np
from multiprocessing import Process, Queue
from typing import Union, Tuple, List

from .array import OneTOneRArray
from .visualize import MemShower


class ObservationSpace:
    def __init__(self, low: List[np.ndarray], high: List[np.ndarray]):
        r"""
        Observation spaces of multi envs

        Args:
            low: the low values of observations; list length: envs_num; array.shape: observation_dim.
            high: the high values of observations; list length: envs_num; array.shape: observation_dim.
        """
        self.low = low
        self.high = high


class ActionSpace:
    def __init__(self, low: List[np.ndarray], high: List[np.ndarray]):
        r"""
        Action space of multi envs

        Args:
            low: the low values of actions; list length: envs_num; array.shape: action_dim.
            high: the high values of observations; list length: envs_num; array.shape: action_dim.
        """
        self.low = low
        self.high = high


class OneTOneREnv(OneTOneRArray):
    r"""
    A 1T1R writing environment with variation and random noises.

    Args:
        array_para_dict (dict): The setting dict of an array.
            including parameters:

            - array_size (Union[Tuple, List]): (row_num, col_num)
            - sampler_num (int): number of processes for simulating.
            e.g. sampler_num = 8: write 8 random devices simultaneously.
            - memristor_relative_noise (float): the std of variance among memristors' parameters.
            - transistor_relative_noise (float): the std of variance among memristors' parameters.
            - program_conductance_range (Union[Tuple, List]): programmable conductance range: (lower, upper).
            unit: mS
            - bit_width (int): the bit width of conductance states;

                - the conductance interval is: delta_g = (upper - lower) / (2**bit_width);
                - the i_th conductance state is: lower + (0.5 + i) * delta_g;
            - max_write_num (int): maximum writing times for each single task
            - vg_range (Union[Tuple, List]): the range of applied gate voltage: (low, upper) unit: V
            - vds_range (Union[Tuple, List]): the range of applied drain-source voltage: (low, upper) unit: V
            - writing tolerance (float): tolerance to the difference between written conductance and target conductance.
            - pulse time (float): fixed pulse duration / s.

        memristor_para_dict (dict): the setting dict of memristor model.
            including parameters:

            - k_reset: nm/s; k_reset > 0
            - k_set: nm/s; k_set < 0
            - alpha_reset:
            - alpha_set:
            - delta_set:
            - i_reset: uA; i_reset < 0
            - i_set: uA; i_set > 0
            - x_off: nm
            - x_on: nm
            - w_c: nm
            - r_on: kOhm
            - r_off: kOhm

        transistor_para_dict (dict): the setting dict of transistor model.
            including parameters:

            - k: uA/(mV)**2
            - v_th: mV
            - lamb: 1/mV
            - i_s: uA

        simulate_para_dict (dict): the setting dict of simulation.
            including parameters:

            - max_dt: the max_dt for numerical calculation.
            - probe_dt: the probing time interval.
            - max_iteration: the maximum iteration for newton approach.
            - eps: a small value for calculation in newton approach.
            - difference_scheme: the difference scheme in newton approach.
            - relative_noise_std: the std of noise that added to the result conductance after each single step.
        ...
    """

    def __init__(self, array_para_dict: dict, memristor_para_dict: dict, transistor_para_dict: dict,
                 simulate_para_dict: dict):
        super().__init__(array_para_dict, memristor_para_dict, transistor_para_dict, simulate_para_dict)
        self.sampler_num = array_para_dict['sampler_num']
        self.num_envs = array_para_dict['sampler_num']
        self.observation_space = None
        self.action_space = None
        self.action_dim = 4
        self.incremental_writing_mode = array_para_dict['incremental_writing_mode']
        self.only_g_observation_mode = array_para_dict["only_g_observation_mode"]
        if self.only_g_observation_mode:
            self.observation_dim = 3
        else:
            self.observation_dim = 7 # g_current, g_target, g_change, vg_set, vr_set, vg_reset, vr_reset
        assert self.observation_dim in [3, 7]

        r_on, r_off = memristor_para_dict['r_on'], memristor_para_dict['r_off']
        assert 1 / r_on > array_para_dict['program_conductance_range'][1] > array_para_dict['program_conductance_range'][0] > 1 / r_off
        self.valid_conductance_range = array_para_dict["valid_conductance_range"]
        self.program_conductance_range = array_para_dict['program_conductance_range']
        self.bit_width = array_para_dict['bit_width']
        self.max_write_num = array_para_dict['max_write_num']
        self.delta_g = (self.program_conductance_range[1] - self.program_conductance_range[0]) / (2**self.bit_width - 1)
        self.vg_set_range = array_para_dict['vg_set_range']
        self.vds_set_range = array_para_dict['vds_set_range']
        self.vg_reset_range = array_para_dict['vg_reset_range']
        self.vds_reset_range = array_para_dict['vds_reset_range']

        self.actual_vg_set_range = array_para_dict['actual_vg_set_range']
        self.actual_vds_set_range = array_para_dict['actual_vds_set_range']
        self.actual_vg_reset_range = array_para_dict['actual_vg_reset_range']
        self.actual_vds_reset_range = array_para_dict['actual_vds_reset_range']

        self.writing_tolerance = self.delta_g / 2
        self.pulse_time = array_para_dict['pulse_time']
        self.vg_change_range = array_para_dict['vg_change_range']
        # self.vds_change_range = array_para_dict['vds_change_range']
        self.vds_set_change_range = array_para_dict['vds_set_change_range']
        self.vds_reset_change_range = array_para_dict["vds_reset_change_range"]

        # init actions
        self.init_vg_set = array_para_dict['init_vg_set']
        self.init_vg_reset = array_para_dict['init_vg_reset']
        self.init_vds_set = array_para_dict['init_vds_set']
        self.init_vds_reset = array_para_dict['init_vds_reset']
        # current actions
        self.vg_set = np.ones(self.num_envs) * self.init_vg_set
        self.vg_reset = np.ones(self.num_envs) * self.init_vg_reset
        self.vds_set = np.ones(self.num_envs) * self.init_vds_set
        self.vds_reset = np.ones(self.num_envs) * self.init_vds_reset

        self.sample_indices, self.sample_targets, self.sample_g_changes, _, self.sample_write_counts = self.init_samplers()
        # for plotting
        self.last_actions = np.zeros((self.sampler_num, self.action_dim))
        self.last_states = self.read_all_samplers()
        self.states = np.zeros((self.sampler_num, self.observation_dim))
        self.rewards = np.zeros((self.sampler_num,))
        self.init_spaces()
        # self.shower = MemShower(envs_num=self.num_envs, vg_low=self.vg_range[0], vg_high=self.vg_range[1],
        #                         vds_neg=self.vds_range[0], vds_pos=self.vds_range[1], g_low=1 / r_off, g_high=1 / r_on,
        #                         program_g_low=self.program_conductance_range[0],
        #                         program_g_high=self.program_conductance_range[1],
        #                         writing_tolerance=self.writing_tolerance)
        self.data_queue = Queue()
        self.operation_queue = Queue()

    def _only_g_observation_space(self):
        r""" current_g, target_g, g_change """
        g_range = self.valid_conductance_range[1] - self.valid_conductance_range[0]
        ob_low = np.array([
            self.valid_conductance_range[0],
            self.valid_conductance_range[0],
            -g_range,
        ],
        )
        ob_high = np.array([
            self.valid_conductance_range[1],
            self.valid_conductance_range[1],
            g_range,
        ])

        ob_lows = [ob_low for i in range(self.num_envs)]
        ob_highs = [ob_high for i in range(self.num_envs)]
        observation_space = ObservationSpace(low=ob_lows, high=ob_highs)
        return observation_space

    def _only_g_state(self, current_gs: np.ndarray, target_gs: np.ndarray, g_changes: np.ndarray):
        r""" current_g, target_g, g_change """
        states = np.zeros((self.sampler_num, 3))
        states[:, 0] = current_gs
        states[:, 1] = target_gs
        states[:, 2] = g_changes

        # r""" current_g, target_g """
        # states = np.zeros((self.sampler_num, 2))
        # states[:, 0] = current_gs
        # states[:, 1] = target_gs
        return states

    def _g_v_observation_space(self):
        r""" current_g, target_g, g_change, current_vg_set, current_vds_set, current_vg_reset, current_vds_reset """
        g_range = self.valid_conductance_range[1] - self.valid_conductance_range[0]
        ob_low = np.array([
            self.valid_conductance_range[0],
            self.valid_conductance_range[0],
            -g_range,
            self.vg_set_range[0],
            self.vds_set_range[0],
            self.vg_reset_range[0],
            self.vds_reset_range[0],
        ],
        )
        ob_high = np.array([
            self.valid_conductance_range[1],
            self.valid_conductance_range[1],
            g_range,
            self.vg_set_range[1],
            self.vds_set_range[1],
            self.vg_reset_range[1],
            self.vds_reset_range[1],
        ])
        ob_lows = [ob_low for i in range(self.num_envs)]
        ob_highs = [ob_high for i in range(self.num_envs)]
        observation_space = ObservationSpace(low=ob_lows, high=ob_highs)
        return observation_space

    def _g_v_state(self, current_gs: np.ndarray, target_gs: np.ndarray, g_changes: np.ndarray):
        r""" current_g, target_g, g_change, current_vg_set, current_vds_set, current_vg_reset, current_vds_reset """
        states = np.zeros((self.sampler_num, 7))
        states[:, 0] = current_gs
        states[:, 1] = target_gs
        states[:, 2] = g_changes
        states[:, 3] = self.vg_set
        states[:, 4] = self.vds_set
        states[:, 5] = self.vg_reset
        states[:, 6] = self.vds_reset
        return states

    def _incremental_action_space(self):
        r""" d_vg_set, d_vds_set, d_vg_reset, d_vds_reset """
        # assert self.vg_change_range[0] == -self.vg_change_range[1]
        # assert self.vds_change_range[0] == -self.vds_change_range[1]
        a_low = np.array(
            [
                self.vg_change_range[0],
                self.vds_set_change_range[0],
                self.vg_change_range[0],
                self.vds_reset_change_range[0],
            ]
        )
        a_high = np.array(
            [
                self.vg_change_range[1],
                self.vds_set_change_range[1],
                self.vg_change_range[1],
                self.vds_reset_change_range[1],
            ]
        )
        a_lows = [a_low for i in range(self.num_envs)]
        a_highs = [a_high for i in range(self.num_envs)]
        action_space = ActionSpace(low=a_lows, high=a_highs)
        return action_space

    def _direct_action_space(self):
        r""" d_vg_set, d_vds_set, d_vg_reset, d_vds_reset """
        a_low = np.array(
            [
                self.vg_set_range[0],
                self.vds_set_range[0],
                self.vg_reset_range[0],
                self.vds_reset_range[0],
            ]
        )
        a_high = np.array(
            [
                self.vg_set_range[1],
                self.vds_set_range[1],
                self.vg_reset_range[1],
                self.vds_reset_range[1],
            ]
        )
        a_lows = [a_low for i in range(self.num_envs)]
        a_highs = [a_high for i in range(self.num_envs)]
        action_space = ActionSpace(low=a_lows, high=a_highs)
        return action_space

    def init_spaces(self):
        if self.only_g_observation_mode:
            self.observation_space = self._only_g_observation_space()
        else:
            self.observation_space = self._g_v_observation_space()

        if self.incremental_writing_mode:
            self.action_space = self._incremental_action_space()
        else:
            self.action_space = self._direct_action_space()

    def writing_finished(self, conductance, target_conductance, write_count):
        r"""
        if difference is between writing tolerance
            or current conductance is beyond program_conductance_range,
            or write count is more than max_write_num,
        then finish.
        """
        finished = False
        truncated = False
        if np.abs(conductance - target_conductance) < self.writing_tolerance:
            finished = True
        if (conductance > self.valid_conductance_range[1]
            or conductance < self.valid_conductance_range[0]
            or write_count > self.max_write_num
        ):
            truncated = True
            finished = True
            # print("Truncated; Current conductance: ", conductance)
        return finished, truncated

    def init_samplers(self):
        r"""
        Randomly initialize the samplers, including the device indices, conductance targets.
        """
        device_indices = []
        device_targets = np.zeros(self.sampler_num)
        device_conductances = np.zeros(self.sampler_num)
        device_last_g_change = np.zeros(self.sampler_num)
        device_write_count = np.zeros(self.sampler_num).astype(int)
        for sample_idx in range(self.sampler_num):
            (row_idx, col_idx), target_conductance = self.random_sampler(device_indices)
            device_indices.append((row_idx, col_idx))
            device_targets[sample_idx] = target_conductance
            device_conductances[sample_idx] = 1 / self.array[row_idx][col_idx].memristor.r
        return device_indices, device_targets, device_last_g_change, device_conductances, device_write_count

    def reset(self):
        r"""
        Reset envs, specifically, reset all the samplers.

        Returns:
            Current_state(np.ndarray):
                size: (self.sampler_num, 2);

                - index0 in dim1: current_conductance
                - index1 in dim1: target_conductance
            Empty dict:
                for aligning with AsyncVectorEnv in gym.
        """
        self.sample_indices, self.sample_targets, self.sample_g_changes, device_conductances, self.sample_write_counts = self.init_samplers()

        for sample_idx in range(self.sampler_num):
            self.reset_operation(sample_idx=sample_idx)

        observations = self.read_all_samplers()
        return observations, {}

    def random_sampler(self, device_indices=None):
        r"""
        Randomly choose a new device, which is not in device_indices.
        And randomly reset the target conductance. The target conductance is randomly selected as discrete values.
        For example, the target conductance corresponding to target idx 0 is:

        ```program_g_lower + (idx + 0.5) * delta_g```

        here, `delta_g` is ```(program_upper - program_lower) / (2**bit_width)```

        Args:
            device_indices(list, *optional*):
                if not provided, use self.sample_indices.
        """
        if device_indices is None:
            device_indices = self.sample_indices
        row_idx = np.random.choice(self.array_size[0])
        col_idx = np.random.choice(self.array_size[1])
        while (row_idx, col_idx) in device_indices:
            row_idx = np.random.choice(self.array_size[0])
            col_idx = np.random.choice(self.array_size[1])
        # continuous target
        # unit: mS
        current_conductance = 1 / self.array[row_idx][col_idx].memristor.r
        target_idx = np.random.randint(0, int(2**self.bit_width))
        target_conductance = self.program_conductance_range[0] + self.delta_g * (target_idx + 0.5)
        while True in self.writing_finished(current_conductance, target_conductance, 0):
            target_idx = np.random.randint(0, int(2 ** self.bit_width))
            target_conductance = self.program_conductance_range[0] + self.delta_g * (target_idx + 0.5)
        return (row_idx, col_idx), target_conductance

    def random_reset_device(self, row_idx: int, col_idx: int):
        r_on, r_off = self.array[row_idx][col_idx].memristor.r_on, self.array[row_idx][col_idx].memristor.r_off
        init_r = r_on - 1
        while not r_on < init_r < r_off:
            init_g = np.random.uniform(low=self.program_conductance_range[0], high=self.program_conductance_range[1])
            init_r = 1 / init_g
        self.array[row_idx][col_idx].set_memristance(init_r)

    def read_sampler_state(self, sample_idx: int):
        target_conductance = self.sample_targets[sample_idx]
        row_idx, col_idx = self.sample_indices[sample_idx]
        device = self.array[row_idx][col_idx]
        current_conductance = 1 / device.memristor.r
        g_change = self.sample_g_changes[sample_idx]
        return current_conductance, target_conductance, g_change

    def read_all_samplers(self):
        current_gs = np.zeros((self.sampler_num,))
        target_gs = np.zeros((self.sampler_num, ))
        g_changes = np.zeros((self.sampler_num, ))

        for sampler_idx in range(self.sampler_num):
            current_conductance, target_conductance, delta_g = self.read_sampler_state(sampler_idx)
            current_gs[sampler_idx] = current_conductance
            target_gs[sampler_idx] = target_conductance
            g_changes[sampler_idx] = delta_g

        if self.only_g_observation_mode:
            all_observations = self._only_g_state(current_gs=current_gs, target_gs=target_gs, g_changes=g_changes)
        else:
            all_observations = self._g_v_state(current_gs=current_gs, target_gs=target_gs, g_changes=g_changes)
        return all_observations

    def update_samplers(self, sample_idx: int):
        # randomly reset r of current device (between program range and within r_on and r_off of the device)
        row_idx, col_idx = self.sample_indices[sample_idx]
        self.random_reset_device(row_idx, col_idx)
        # new device and target
        self.sample_indices[sample_idx], self.sample_targets[sample_idx] = self.random_sampler()
        # reset g change
        self.sample_g_changes[sample_idx] = 0
        # reset write count
        self.sample_write_counts[sample_idx] = 0
        # reset incremental action
        self.reset_operation(sample_idx=sample_idx)

    def reset_operation(self, sample_idx: int):
        self.vg_reset[sample_idx] = self.init_vg_reset
        self.vds_reset[sample_idx] = self.init_vds_reset
        self.vg_set[sample_idx] = self.init_vg_set
        self.vds_set[sample_idx] = self.init_vds_set

    def reward(self, before_conductance: float, current_conductance: float, target_conductance: float, write_count: int):
        r"""
        Calculate rewards based on states.

        Inputs:
            before_conductance (float): conductance before action.
            current conductance (float): conductance after action.
            target conductance (float): target

        Return:
            rewards(np.ndarray): the rewards.

        """
        # TODO: Calculate rewards.
        final_score = 50
        speed_score = 50
        write_count_threshold = 1
        speed_discount_step = 5

        # out of program range.
        fail_score = -50
        max_distance_r = -2

        # Rewards:
        # for distance
        before_dist = np.abs(target_conductance - before_conductance)
        current_dist = np.abs(target_conductance - current_conductance)

        g_range = self.valid_conductance_range[1] - self.valid_conductance_range[0]
        unit_distance_r = max_distance_r / (g_range / self.delta_g)

        if current_dist >= before_dist:
            current_dist = np.clip(current_dist, 0, g_range)
            distance_reward = unit_distance_r * current_dist / self.delta_g
        else:
            distance_reward = unit_distance_r / 2

        # for fail
        if (current_conductance < self.valid_conductance_range[0]
            or current_conductance > self.valid_conductance_range[1]
            or write_count > self.max_write_num
        ):
            fail_reward = fail_score
        else:
            fail_reward = 0

        # for finish
        if np.abs(target_conductance - current_conductance) < self.writing_tolerance:
            finish_reward = final_score
            speed_reward = speed_score - max((write_count - write_count_threshold) * 2, 0)
            speed_reward = max(speed_reward, 0)

            # speed_reward = speed_score / max(1., (write_count - write_count_threshold) / speed_discount_step + 1)
        else:
            finish_reward = 0
            speed_reward = 0
        # print(f"distance reward: {distance_reward} ", f"change_reward: {distance_reward} ", f"speed reward: {speed_reward}", f"no move reward: {no_move_reward}")
        return finish_reward + distance_reward + fail_reward + speed_reward

    def _clean_v_reset(self, sample_idx: int):
        # re-assign init operation for reset
        self.vg_reset[sample_idx] = self.init_vg_reset
        self.vds_reset[sample_idx] = self.init_vds_reset

    def _clean_v_set(self, sample_idx: int):
        # re-assign the init operation for set
        self.vg_set[sample_idx] = self.init_vg_set
        self.vds_set[sample_idx] = self.init_vds_set

    def _prior_operate_direction(self, sample_idx: int, current_g: float, target_g: float):
        if current_g < target_g:
            # need set
            self._clean_v_reset(sample_idx)
        else:
            # need reset
            self._clean_v_set(sample_idx)

    def incremental_actions(self, do_actions: np.ndarray):
        r"""
        In incremental mode, the `do_actions` means the plus or minus value applied to the corresponding action.

        (delta_vg_set, delta_vr_set, delta_vg_reset, delta_vr_reset),

        return: (vg, vr)
        """
        actual_actions = np.zeros((self.sampler_num, 2))
        for sample_idx in range(self.sampler_num):
            # here, the direction is prior knowledge.
            # set
            if self.states[sample_idx, 0] < self.states[sample_idx, 1]:
                # use the stored set operation
                self.vg_set[sample_idx] += do_actions[sample_idx, 0]
                self.vds_set[sample_idx] += do_actions[sample_idx, 1]
                # clip
                self.vg_set[sample_idx] = np.clip(
                    self.vg_set[sample_idx],
                    self.actual_vg_set_range[0],
                    self.actual_vg_set_range[1],
                )
                self.vds_set[sample_idx] = np.clip(
                    self.vds_set[sample_idx],
                    self.actual_vds_set_range[0],
                    self.actual_vds_set_range[1],
                )
                actual_actions[sample_idx, 0] = self.vg_set[sample_idx]
                actual_actions[sample_idx, 1] = self.vds_set[sample_idx]
            else:
                # use the stored reset operation
                # TODO: disable Vg_reset
                # self.vg_reset[sample_idx] += do_actions[sample_idx, 2]
                self.vds_reset[sample_idx] += do_actions[sample_idx, 3]
                # clip
                self.vg_reset[sample_idx] = np.clip(
                    self.vg_reset[sample_idx],
                    self.actual_vg_reset_range[0],
                    self.actual_vg_reset_range[1],
                )
                self.vds_reset[sample_idx] = np.clip(
                    self.vds_reset[sample_idx],
                    self.actual_vds_reset_range[0],
                    self.actual_vds_reset_range[1],
                )
                actual_actions[sample_idx, 0] = self.vg_reset[sample_idx]
                actual_actions[sample_idx, 1] = self.vds_reset[sample_idx]
                # print('actual reset actions:')
                # print(actual_actions)
        return actual_actions

    def direct_actions(self, do_actions: np.ndarray):
        r"""
        In direct mode, the `do_actions` means the actual applied voltages.

        (vg_set, vr_set, vg_reset, vr_reset),

        return: (vg, vr)
        """
        actual_actions = np.zeros((self.sampler_num, 2))
        for sample_idx in range(self.sampler_num):
            # here, the direction is prior knowledge.
            # set
            if self.states[sample_idx, 0] < self.states[sample_idx, 1]:
                self.vg_set[sample_idx] = do_actions[sample_idx, 0]
                self.vds_set[sample_idx] = do_actions[sample_idx, 1]
                # clip
                self.vg_set[sample_idx] = np.clip(
                    self.vg_set[sample_idx],
                    self.actual_vg_set_range[0],
                    self.actual_vg_set_range[1],
                )
                self.vds_set[sample_idx] = np.clip(
                    self.vds_set[sample_idx],
                    self.actual_vds_set_range[0],
                    self.actual_vds_set_range[1],
                )
                actual_actions[sample_idx, 0] = self.vg_set[sample_idx]
                actual_actions[sample_idx, 1] = self.vds_set[sample_idx]
            else:
                # TODO: disable Vg_reset
                # self.vg_reset[sample_idx] += do_actions[sample_idx, 2]
                self.vds_reset[sample_idx] = do_actions[sample_idx, 3]
                # clip
                self.vg_reset[sample_idx] = np.clip(
                    self.vg_reset[sample_idx],
                    self.actual_vg_reset_range[0],
                    self.actual_vg_reset_range[1],
                )
                self.vds_reset[sample_idx] = np.clip(
                    self.vds_reset[sample_idx],
                    self.actual_vds_reset_range[0],
                    self.actual_vds_reset_range[1],
                )
                actual_actions[sample_idx, 0] = self.vg_reset[sample_idx]
                actual_actions[sample_idx, 1] = self.vds_reset[sample_idx]
        return actual_actions

    def step(self, do_actions: np.ndarray):
        r"""
         do actions to change envs and get rewards.

         Args:
            do_actions (np.ndarray): action for all samplers
                if self.incremental_writing_mode:
                    (sampler_num, action_dim), [(delta_vg_set, delta_vr_set, delta_vg_reset, delta_vr_reset), ...]; unit: V.
                else:
                    (sampler_num, action_dim), [(vg_set, vr_set, vg_reset, vr_reset), ...]; unit: V.

         Returns:
             states (np.ndarray): the states after actions. (sampler_num, observation_dim)
             rewards (np.ndarray): the obtained rewards after actions. (sampler_num, )
             terminals (np.ndarray): If this action has reached the target or gamed over. (sampler_num, )
             truncated (np.ndarray): If game over. (sampler_num, )
             None: align with the outputs of AsyncVectorEnv in gym.
         """

        if self.incremental_writing_mode:
            applied_voltages = self.incremental_actions(do_actions)
        else:
            applied_voltages = self.direct_actions(do_actions)

        _, states, rewards, terminals, truncated = self.multi_write(
            do_actions=do_actions,
            applied_voltages=applied_voltages,
        )
        return states, rewards, terminals, truncated, None

    def _clipped_g_change(self, current_conductance, before_conductance):
        change_range = self.valid_conductance_range[1] - self.valid_conductance_range[0]
        return np.clip(current_conductance - before_conductance, -change_range, change_range)

    def write(self, sample_idx: int, vg: float, vr: float):
        r"""
        write the memristors according to the programming voltages.

        Args:
            sample_idx (int): the writing sampler idx.
            vg (float): the gate voltage, unit: V
            vr (float): the program voltage, unit: V

        Returns:
            current_conductance: conductance after writing.
            target_conductance: target conductance.
            reward: reward after writing.
            no_move_count: the count of no move.
            terminal: be written to target conductance or out of program conductance range.
            truncated: out of program conductance range.

        if writing finished, reset device and send back new conductance and target conductance.
        """
        row_idx, col_idx = self.sample_indices[sample_idx]
        target_conductance = self.sample_targets[sample_idx]
        device = self.array[row_idx][col_idx]
        vg, vr = vg * 1e3, vr * 1e3
        before_conductance = 1 / device.memristor.r
        _, _ = device.evolution(t=self.pulse_time, v_r=vr, v_s=0, v_g=vg)
        current_conductance = 1 / device.memristor.r

        self.sample_write_counts[sample_idx] += 1
        self.sample_g_changes[sample_idx] = self._clipped_g_change(current_conductance, before_conductance)

        # reduce observation space (Actually, the set & reset operation has been separated by doing so.)
        self._prior_operate_direction(sample_idx=sample_idx, current_g=current_conductance, target_g=target_conductance)

        # TODO: Calculate rewards.
        reward = self.reward(
            before_conductance,
            current_conductance,
            target_conductance,
            self.sample_write_counts[sample_idx],
        )
        terminal, truncated = self.writing_finished(
            current_conductance,
            target_conductance,
            self.sample_write_counts[sample_idx],
        )
        if terminal or truncated:
            self.update_samplers(sample_idx)
            current_conductance, target_conductance, g_change = self.read_sampler_state(sample_idx)
            terminal = True
        return current_conductance, target_conductance, reward, self.sample_g_changes[sample_idx], terminal, truncated

    def multi_write(self, do_actions: np.ndarray, applied_voltages: np.ndarray):
        r"""
        Write in all envs.

        Args:
            do_actions (np.ndarray): (sampler_num, action_dim), [(delta_vg_set, delta_vds_set, delta_vg_reset, delta_vds_reset), ...]
            applied_voltages(np.ndarray): actually applied voltages
                (sampler_num, 2), [(vg, vr), (vg, vr), ...]; unit: V

        Returns:
            actions(np.ndarray): (sampler_num, action_dim=(Vg, Vr)) ---> At
            states(np.ndarray): St+1 for all samplers
                (sampler_num, observation_dim=(current_g, target_g)) ---> St+1
            terminals(np.ndarray): (sampler_num, )
                terminal_t: if St+1 is a terminal.
            truncateds (np.ndarray): (sampler_num, )
                terminal_t: if St+1 truncate.
        """

        # TODO: deal with collected data in queue
        current_gs = np.zeros((self.sampler_num, ))
        target_gs = np.zeros((self.sampler_num, ))
        g_changes = np.zeros((self.sampler_num, ))

        terminals = np.zeros((self.sampler_num,), dtype=bool)
        rewards = np.zeros((self.sampler_num,))
        truncateds = np.zeros((self.sampler_num,), dtype=bool)

        for sample_idx in range(self.sampler_num):
            results = self.write(sample_idx, applied_voltages[sample_idx][0], applied_voltages[sample_idx][1])
            current_conductance, target_conductance, reward, delta_g, terminal, truncated = results
            current_gs[sample_idx] = current_conductance
            target_gs[sample_idx] = target_conductance
            g_changes[sample_idx] = delta_g
            rewards[sample_idx] = reward
            terminals[sample_idx] = terminal
            truncateds[sample_idx] = truncated

        if self.only_g_observation_mode:
            states = self._only_g_state(current_gs=current_gs, target_gs=target_gs, g_changes=g_changes)
        else:
            states = self._g_v_state(current_gs=current_gs, target_gs=target_gs, g_changes=g_changes)

        # record the action and result states
        self.last_actions = do_actions
        self.states = states
        self.rewards = rewards
        return do_actions, states, rewards, terminals, truncateds

    def render(self) -> list:
        r"""
        get the frames of a single step.

        Returns:
            frames(List(np.ndarray)):

                - the first frame: last states + last actions;
                - the second frame: current states.
        """
        # frame1 = self.shower.show_action_frame(actions=self.last_actions, last_states=self.last_states)
        # frame2 = self.shower.show_result_state(states=self.states, rewards=self.rewards)
        # # update last states
        # self.last_states = self.states
        # return [frame1, frame2]


