import numpy as np
import os
import pickle as pkl
import copy
from segment_tree import SumSegmentTree, MinSegmentTree
import random


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class EpisodicMemory(object):

    def __init__(self,
                 buffer_size,
                 state_dim,
                 action_shape,
                 obs_space,
                 gamma=0.99,
                 alpha=0.6,
                 max_step=1000,
                 policy=None,
                 eta=None,
                 policy_noise=None,
                 noise_clip=None,
                 max_action=None):
        buffer_size = int(buffer_size)
        self.state_dim = state_dim
        self.capacity = buffer_size
        self.curr_capacity = 0
        self.pointer = 0
        self.obs_space = obs_space
        self.action_shape = action_shape
        self.max_step = max_step
        self.policy = policy

        self.query_buffer = np.zeros((buffer_size, state_dim))
        self._q_values = -np.inf * np.ones(buffer_size + 1)
        self.returns = -np.inf * np.ones(buffer_size + 1)
        self.replay_buffer = np.empty((buffer_size,) + obs_space.shape, np.float32)
        self.action_buffer = np.empty((buffer_size,) + action_shape.shape, np.float32)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.truly_done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.contra_count = np.ones((buffer_size,))
        self.lru = np.zeros(buffer_size)
        self.time = 0
        self.gamma = gamma
        # self.hashes = dict()
        self.reward_mean = None
        self.min_return = 0
        self.end_points = []
        assert alpha > 0
        self._alpha = alpha
        self.beta_set = [-1]
        self.beta_coef = [1.]
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def clean(self):
        buffer_size, state_dim, obs_space, action_shape = \
             self.capacity, self.state_dim, self.obs_space, self.action_shape
        self.curr_capacity = 0
        self.pointer = 0

        self.query_buffer = np.zeros((buffer_size, state_dim))
        self._q_values = -np.inf * np.ones(buffer_size + 1)
        self.returns = -np.inf * np.ones(buffer_size + 1)
        self.replay_buffer = np.empty((buffer_size,) + obs_space.shape, np.float32)
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.float32)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.truly_done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.contra_count = np.ones((buffer_size,))
        self.lru = np.zeros(buffer_size)
        self.time = 0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.end_points = []

    @property
    def q_values(self):
        return self._q_values

    def squeeze(self, obses):
        return np.array([(obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low) for obs in obses])

    def unsqueeze(self, obses):
        return np.array([obs * (self.obs_space.high - self.obs_space.low) + self.obs_space.low for obs in obses])

    def save(self, filedir):
        save_dict = {
            "query_buffer": self.query_buffer,
            "returns": self.returns,
            "replay_buffer": self.replay_buffer,
            "reward_buffer": self.reward_buffer,
            "truly_done_buffer": self.truly_done_buffer,
            "next_id": self.next_id,
            "prev_id": self.prev_id,
            "gamma": self.gamma,
            "_q_values": self._q_values,
            "done_buffer": self.done_buffer,
            "curr_capacity": self.curr_capacity,
            "capacity": self.capacity
        }
        with open(os.path.join(filedir, "episodic_memory.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)

    def add(self, obs, action, state, sampled_return, next_id=-1):

        index = self.pointer
        self.pointer = (self.pointer + 1) % self.capacity

        if self.curr_capacity >= self.capacity:
            # Clean up old entry
            if index in self.end_points:
                self.end_points.remove(index)
            self.prev_id[index] = []
            self.next_id[index] = -1
            self.q_values[index] = -np.inf
        else:
            self.curr_capacity = min(self.capacity, self.curr_capacity + 1)
        # Store new entry
        self.replay_buffer[index] = obs
        self.action_buffer[index] = action
        if state is not None:
            self.query_buffer[index] = state
        self.q_values[index] = sampled_return
        self.returns[index] = sampled_return
        self.lru[index] = self.time

        self._it_sum[index] = self._max_priority**self._alpha
        self._it_min[index] = self._max_priority**self._alpha
        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)
        self.time += 0.01

        return index

    def retrieve_trajectories(self):
        trajs = []
        for e in self.end_points:
            traj = []
            prev = e
            while prev is not None:
                traj.append(prev)
                try:
                    prev = self.prev_id[prev][0]
                    # print(e,prev)
                except IndexError:
                    prev = None
            # print(np.array(traj))
            trajs.append(np.array(traj))
        return trajs

    def update_sequence_with_qs(self, sequence):
        next_id = -1
        Rtd = 0
        for obs, a, z, q_t, r, truly_done, done in reversed(sequence):
            if truly_done:
                Rtd = r
            else:
                Rtd = self.gamma * Rtd + r
            current_id = self.add(obs, a, z, Rtd, next_id)

            if done:
                self.end_points.append(current_id)
            self.replay_buffer[current_id] = obs
            self.reward_buffer[current_id] = r
            self.truly_done_buffer[current_id] = truly_done
            self.done_buffer[current_id] = done
            next_id = int(current_id)
        return

    @staticmethod
    def switch_first_half(obs0, obs1, batch_size):
        tmp = copy.copy(obs0[:batch_size // 2, ...])
        obs0[:batch_size // 2, ...] = obs1[:batch_size // 2, ...]
        obs1[:batch_size // 2, ...] = tmp
        return obs0, obs1

    def sample(self, batch_size, mix=False, priority=False):
        # Draw such that we always have a proceeding element
        if self.curr_capacity < batch_size + len(self.end_points):
            return None
        batch_idxs = []
        batch_idxs_next = []
        count = 0
        while len(batch_idxs) < batch_size:
            if priority:
                mass = random.random() * self._it_sum.sum(0, self.curr_capacity)
                rnd_idx = self._it_sum.find_prefixsum_idx(mass)
            else:
                rnd_idx = np.random.randint(0, self.curr_capacity)
            count += 1
            assert count < 1e8
            if self.next_id[rnd_idx] == -1:
                continue
            else:
                batch_idxs_next.append(self.next_id[rnd_idx])
                batch_idxs.append(rnd_idx)

        batch_idxs = np.array(batch_idxs).astype(np.int)
        batch_idxs_next = np.array(batch_idxs_next).astype(np.int)
        # batch_idx_pre = [self.prev_id[id] for id in batch_idxs]

        obs0_batch = self.replay_buffer[batch_idxs]
        obs1_batch = self.replay_buffer[batch_idxs_next]
        # batch_idxs_neg, obs2_batch = self.sample_negative(batch_size, batch_idxs, batch_idxs_next, batch_idx_pre)
        action_batch = self.action_buffer[batch_idxs]
        action1_batch = self.action_buffer[batch_idxs_next]
        # action2_batch = self.action_buffer[batch_idxs_neg]
        reward_batch = self.reward_buffer[batch_idxs]
        terminal1_batch = self.done_buffer[batch_idxs]
        q_batch = self.q_values[batch_idxs]
        return_batch = self.returns[batch_idxs]

        if mix:
            obs0_batch, obs1_batch = self.switch_first_half(obs0_batch, obs1_batch, batch_size)
        if priority:
            self.contra_count[batch_idxs] += 1
            self.contra_count[batch_idxs_next] += 1

        result = {
            'index0': array_min2d(batch_idxs),
            'index1': array_min2d(batch_idxs_next),
            # 'index2': array_min2d(batch_idxs_neg),
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            # 'obs2': array_min2d(obs2_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'actions1': array_min2d(action1_batch),
            # 'actions2': array_min2d(action2_batch),
            'count': array_min2d(self.contra_count[batch_idxs] + self.contra_count[batch_idxs_next]),
            'terminals1': array_min2d(terminal1_batch),
            'return': array_min2d(q_batch),
            'true_return': array_min2d(return_batch),
        }
        return result
