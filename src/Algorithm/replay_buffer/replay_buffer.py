import copy

import torch
import numpy as np

from typing import Tuple, Union


class ReplayBuffer:
    r"""
    This is a replay buffer storing transitions. states, actions, rewards, terminals are stored in sequence separately.
    Each buffer is corresponding to a sampler.

    Composition of generated states:
        The practical state used for Critic and Actor is a state sequence of length n, for the sake of compatibility
        with sequence models.
        Denote the state and action sampled from environment as m_t and a_t respectively.
        The practical state S_t is:
            S_t: [m_(t-n+1), a_(t-n+1), ... m_t]
            a_t: a_t
            r_t: r_t
            s_(t+1): [m_(t-n+2), a_(t-n+2), ... m_t, m_(t+1)]
            each m in S_t must not be a terminal, S_t is the longest sequence without terminal back from m_t, and then
            padded with (m_begin, a_padding) to meet length of max_length.

            E.g.
                When n=1:
                    S_t = m_t
                    S_(t+1) = m_(t+1)
                When n=2, and m_(t-1) is a terminal:
                    S_t = [m_t, a_padding, m_t]
                    S_(t+1) = [m_t, a_t, m_(t+1)]
    """
    def __init__(self, num_samplers, buffer_size, batch_size, observation_dim, action_dim,
                 padding_action, max_length, init_states: Union[np.ndarray, tuple, list], sample_dtype=torch.float32):
        self.num_samplers = num_samplers
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.sample_dtype = sample_dtype
        self.state_pool, self.action_pool, self.reward_pool, self.terminal_pool, self.current_pointer = self.init_buffer(init_states)
        self.padding_action = np.array(padding_action, dtype=self.action_pool.dtype)
        assert self.padding_action.shape[0] == self.action_dim
        self.full = False
        self.device = torch.device('cpu')
        self.add_data_num = None
        self.recent_add_data = None

    def init_buffer(self, init_states):
        init_states = np.array(init_states, dtype=float)
        print(init_states.shape)
        assert init_states.shape == (self.num_samplers, self.observation_dim)
        state_pool = np.zeros((self.num_samplers, self.buffer_size, self.observation_dim), dtype=float)
        state_pool[:, 0, :] = init_states
        action_pool = np.zeros((self.num_samplers, self.buffer_size, self.action_dim), dtype=float)
        reward_pool = np.zeros((self.num_samplers, self.buffer_size), dtype=float)
        terminal_pool = np.zeros((self.num_samplers, self.buffer_size), dtype=bool)
        terminal_pool[:, 0] = np.ones(self.num_samplers, dtype=bool)
        current_pointer = 1
        return state_pool, action_pool, reward_pool, terminal_pool, current_pointer

    @property
    def current_data_num(self):
        if self.full:
            return self.num_samplers * self.buffer_size
        else:
            return self.num_samplers * self.current_pointer

    def reset_buffer(self, init_states):
        self.state_pool, self.action_pool, self.reward_pool, self.terminal_pool, self.current_pointer = self.init_buffer(
            init_states)

    def _left_move(self, current_idx, move_num: int):
        return (current_idx + self.buffer_size - move_num) % self.buffer_size

    def _right_move(self, current_idx, move_num: int):
        return (current_idx + self.buffer_size + move_num) % self.buffer_size

    def put(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray):
        r"""
        (num_samplers, len, dim)
        putin data:
            (At, Rt, St+1, Tt+1)
            diagram:
            ----------------------------
                        current_p
            St  | St+1  | old_S | ...
            At  | old_A | ...
            Rt  | old_R | ...
            Tt  | Tt+1  | old_T | ...
            ----------------------------
        states: (num_samplers, len, observation_dim)
        actions: (num_samplers, len, action_dim)
        rewards: (num_samplers, len)
            r_t: reward of (s_t, a_t) -> s_t+1.
        terminals: (num_samplers, len)
            terminal_t: if s_t is a terminal.
            (terminal also means beginning, we do not record the final state, but record the beginning state of
            the next new trajectory.)
        current_pointer: means the starting pointer for recording new data.
        """
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == terminals.shape[0] == self.num_samplers
        assert states.shape[2] == self.observation_dim
        assert actions.shape[2] == self.action_dim

        putin_len = states.shape[1]
        # TODO: 是否训练次数会太多？
        self.add_data_num = putin_len * self.num_samplers
        self.recent_add_data = (states, actions, rewards, terminals)


        def update_buffer(buffer, new_data, p_start):
            p = p_start + putin_len
            if p < self.buffer_size:
                buffer[:, p_start:p] = new_data
            else:
                self.full = True
                rest_len = self.buffer_size - p_start
                buffer[:, p_start:] = new_data[:, :rest_len]
                buffer[:, :putin_len - rest_len] = new_data[:, rest_len:]
                p = p - self.buffer_size
            return p

        update_buffer(self.state_pool, states, self.current_pointer)
        update_buffer(self.action_pool, actions, self._left_move(self.current_pointer, 1))
        update_buffer(self.reward_pool, rewards, self._left_move(self.current_pointer, 1))
        self.current_pointer = update_buffer(self.terminal_pool, terminals, self.current_pointer)

    def acceptable_indices(self):
        r"""
        St could not be current_pointer - 1 (because it is the most recent data, followed by old data).
        return:
            [sampled_indices_of_sampler1(np.ndarray), sampled_indices_of_sampler2, ...]
        """
        if self.current_pointer < self.batch_size and not self.full:
            raise BufferError('Collected data is not sufficient.')

        upper = self.current_pointer - 1
        full_upper = self._left_move(self.current_pointer, 1)

        def random_indices(sample_num):
            if not self.full:
                indices = np.random.randint(low=0, high=upper, size=sample_num)
            else:
                num1 = int(full_upper / (self.buffer_size - 1) * sample_num)
                num2 = sample_num - num1
                # here, the random index can not be current_pointer - 1
                random1 = np.random.randint(low=0, high=full_upper, size=num1)
                random2 = np.random.randint(low=self.current_pointer, high=self.buffer_size, size=num2)
                indices = np.concatenate((random1, random2), axis=0)
            return indices

        # def acceptable_indices(sample_idx, sample_num):
        #     resample_num = sample_num
        #     indices_list = []
        #     while resample_num > 0:
        #         # resample
        #         re_indices = random_indices(resample_num)
        #         re_terminals = self.terminal_pool[sample_idx, re_indices]
        #         indices_list.append(re_indices[np.bitwise_not(re_terminals)])
        #         resample_num = re_terminals.sum()
        #     indices = np.concatenate(indices_list, axis=0)
        #     return indices

        def random_sample_nums():
            r""" sample num in each set of data that recorded by each sampler """
            num_list = []
            sample_bound = self.batch_size
            for sample_idx in range(self.num_samplers - 1):
                sample_num = np.random.randint(0, sample_bound)
                sample_bound -= sample_num
                num_list.append(sample_num)
            num_list.append(sample_bound)
            return num_list

        sample_num_list = random_sample_nums()
        sample_idx_list = []
        start, end = 0, 0
        for idx in range(self.num_samplers):
            num = sample_num_list[idx]
            end = start + sample_num_list[idx]
            if num > 0:
                # sample_indices = acceptable_indices(idx, num)
                sample_indices = random_indices(num)
            else:
                sample_indices = np.zeros(0)
            sample_idx_list.append(sample_indices)
            start = end
        return sample_idx_list

    def get_frame_data(self):
        r"""
        Get batch_size transitions (St, At, Rt, St+1) from buffer.
        terminals: if St+1 is a terminal
        St is just m_t.
        return:
            states: (batch_size, observation_dim), torch.Tensor
            actions: (batch_size, action_dim), torch.Tensor
            rewards: (batch_size, ), torch.Tensor
            terminals: (batch_size), torch.Tensor
            next_states: (batch_size, observation_dim), torch.Tensor
        """
        sample_idx_list = self.acceptable_indices()

        states = torch.zeros((self.batch_size, self.observation_dim), dtype=self.sample_dtype, device=self.device)
        actions = torch.zeros((self.batch_size, self.action_dim), dtype=self.sample_dtype, device=self.device)
        rewards = torch.zeros((self.batch_size, ), dtype=self.sample_dtype, device=self.device)
        terminals = torch.zeros((self.batch_size, ), dtype=self.sample_dtype, device=self.device)
        next_states = torch.zeros((self.batch_size, self.observation_dim), dtype=self.sample_dtype, device=self.device)

        start, end = 0, 0
        for idx, sample_indices in enumerate(sample_idx_list):
            num = sample_indices.shape[0]
            end = start + num
            if num > 0:
                next_indices = self._right_move(sample_indices, 1)
                states[start: end] = torch.tensor(self.state_pool[idx, sample_indices], dtype=self.sample_dtype, device=self.device)
                actions[start: end] = torch.tensor(self.action_pool[idx, sample_indices], dtype=self.sample_dtype, device=self.device)
                rewards[start: end] = torch.tensor(self.reward_pool[idx, sample_indices], dtype=self.sample_dtype, device=self.device)
                terminals[start: end] = torch.tensor(self.terminal_pool[idx, next_indices], dtype=self.sample_dtype, device=self.device)
                next_states[start: end] = torch.tensor(self.state_pool[idx, next_indices], dtype=self.sample_dtype, device=self.device)
            start = end
        return states, actions, rewards, terminals, next_states

    def pad_state_sequence(self, sampler_idx, data_idx):
        r"""
        pad_state_seq: (observation_dim, max_length + 1)
            (Mt-max_len, ... , Mt, Mt+1);
        pad_action_seq: (action_dim, max_length + 1)
            (At-max_len, ... , At-1, A_t, A_padding);
        St:
            (Mt-max_len, At-max_len, ... , At-1, Mt)
        St+1:
            (Mt-max_len+1, At-max_len+1 ... , At, Mt+1)
        (St, At, Rt, St+1)
        """
        # TODO: output torch.Tensor
        pad_state_seq = np.zeros((self.observation_dim, self.max_length + 1), dtype=self.state_pool.dtype)
        pad_action_seq = np.zeros((self.action_dim, self.max_length), dtype=self.action_pool.dtype)
        terminal = False
        state = self.state_pool[sampler_idx, data_idx]
        # (m_t+1)
        next_idx = self._right_move(data_idx, 1)
        pad_state_seq[:, -1] = self.state_pool[sampler_idx, next_idx]
        for idx in range(self.max_length):
            current_idx = self._left_move(data_idx, idx)
            if terminal:
                # for action: padding with padding_action.
                # for state: padding with most recent state.
                action = self.padding_action
            else:
                action = self.action_pool[sampler_idx, current_idx]
                state = self.state_pool[sampler_idx, current_idx]
            pad_state_seq[:, -idx-2] = state
            pad_action_seq[:, -idx-1] = action
            # If date back to the beginning of current trajectory.
            # or if it is going to hop from old data to recent data, stop at the current_pointer.
            if current_idx == self.current_pointer or self.terminal_pool[sampler_idx, current_idx]:
                terminal = True
        return pad_state_seq, pad_action_seq

    def get_sequence_data(self):
        r"""
        Get batch_size transitions (St, At, Rt, St+1) from buffer.
        St is a sequence with length of (self.max_length), and date back to the latest terminal in buffer.

        E.g.
            S_t = states[:, :, :-1]
            S_t+1 = states[:, :, 1:]

        return:
            states: (batch_size, observation_dim, max_length + 1)
                in length dimension: (Mt-max_len, ... , Mt, Mt+1);
            Actions: (batch_size, action_dim, max_length)
                in length dimension: (At-max_len, ... , At-1, A_t);
            Rewards: (batch_size, )
            terminals: (batch_size, )
                if Mt+1 is a terminal.
        """
        # TODO: output torch.Tensor
        sample_idx_list = self.acceptable_indices()

        states = np.zeros((self.batch_size, self.observation_dim, self.max_length + 1), dtype=self.state_pool.dtype)
        actions = np.zeros((self.batch_size, self.action_dim, self.max_length), dtype=self.action_pool.dtype)
        rewards = np.zeros((self.batch_size, ), dtype=self.reward_pool.dtype)
        terminals = np.zeros((self.batch_size, ), dtype=self.terminal_pool.dtype)
        start, end = 0, 0
        for sampler_idx, sample_indices in enumerate(sample_idx_list):
            num = sample_indices.shape[0]
            end = start + num
            next_indices = self._right_move(sample_indices, 1)
            if num > 0:
                for idx, data_idx in enumerate(sample_indices):
                    pad_state_seq, pad_action_seq = self.pad_state_sequence(sampler_idx=sampler_idx, data_idx=data_idx)
                    states[start+idx] = pad_state_seq
                    actions[start+idx] = pad_action_seq
                rewards[start: end] = self.reward_pool[sampler_idx, sample_indices]
                terminals[start: end] = self.terminal_pool[sampler_idx, next_indices]
        return states, actions, rewards, terminals

    def get(self):
        return self.get_sequence_data()


def init_replay_buffer(config, init_states):
    replay_buffer = ReplayBuffer(num_samplers=config['sampler_num'], buffer_size=config['replay_buffer_size'],
                                 batch_size=config['batch_size'], observation_dim=config['observation_dim'],
                                 action_dim=config['action_dim'], padding_action=config['padding_action'],
                                 max_length=config['state_max_length'], init_states=init_states)
    return replay_buffer


# sampler_num = 1
# state_dim = 1
# action_dim = 1
# rr = ReplayBuffer(num_samplers=sampler_num, buffer_size=10, batch_size=20, observation_dim=state_dim, action_dim=action_dim,
#                   padding_action=(0, ),
#                   max_length=2, init_states=np.zeros((sampler_num, state_dim)))
# print(rr.state_pool.shape)
#
# data_num = 6
# data_s = np.random.randint(0, 100, size=(sampler_num, data_num, state_dim))
# data_a = np.random.randint(-3, 4, size=(sampler_num, data_num, action_dim))
# data_r = np.random.uniform(0, 1, size=(sampler_num, data_num))
# data_t = np.random.choice([True, False, False, False, False, False], size=(sampler_num, data_num))
#
# data_s[:, -1, :] = -100
# data_a[:, -1, :] = -100
# rr.put(states=data_s, actions=data_a, rewards=data_r, terminals=data_t)
# rr.put(states=data_s-1, actions=data_a, rewards=data_r, terminals=data_t)
# print(data_r)
# print('state')
# print(rr.state_pool)
# print('actions')
# print(rr.action_pool)
# print('terminal pool:', rr.terminal_pool)
#
# states, actions, rewards, terminals = rr.get()
# print('current')
# print(states[:, :, :-1])
# print('next')
# print(states[:, :, 1:])
# print('terminal')
# print(terminals)
# print(actions)
# print(states)

# class TT(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = torch.nn.Linear(1, 3)
#         self.linear2 = copy.deepcopy(self.linear1)
#
#     def update(self):
#         for name in self.linear1._parameters.keys():
#             print("before: ", self.linear2._parameters[name].data)
#             self.linear2._parameters[name].data = self.linear1._parameters[name].data * 0.5 + self.linear2._parameters[name].data
#             print('after: ', self.linear2._parameters[name].data)

# aa = torch.nn.Linear(1, 3)
# print(aa._parameters.keys())
# tt = TT()
# tt.update()

# aa = torch.tensor([0, 1])
# bb = torch.tensor([1, 0])
#
# print(torch.minimum(aa, bb))
