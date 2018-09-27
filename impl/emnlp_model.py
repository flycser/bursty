#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : emnlp_model
# @Date : 09/08/2018 14:00:51
# @Poject : bursty
# @Author : FEI, hfut_jf@aliyun.com
# @Desc : EMNLP model

import numpy as np

from scipy.stats import poisson

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')


def int2bss(digit, length):
    """
    convert a integer to a binary state sequence
    :param z:
    :return:
    """

    return [int(z) for z in format(digit, '0{:d}b'.format(length))]

class SingleBursty(object):
    """
    Algorithm 1: model single stream
    """
    def __init__(self, local_len, scale, gamma_1, mean=None, num_state=2):
        self.mean = mean
        self.num_state = num_state # default 2
        self.local_len = local_len # local smoothness
        self.scale = scale
        self.gamma_1 = gamma_1
        self.logger = logging.getLogger('fei')

    def optimize(self, xs):
        len_xs = len(xs)
        size_local_sq = self.num_state ** self.local_len

        self.costs = np.zeros(shape=[len_xs+1, size_local_sq], dtype=np.float32) # the extra entries in first dimension are for initialization before index i=1
        self.costs.fill(np.inf) # fill in all entries of costs with infinity

        if self.mean == None:
            self.mean = np.mean(xs)

        self.logger.debug('Mean of input frequency sequency: {:.5f}'.format(self.mean))

        if self.scale == None:
            means = np.linspace(self.mean, np.max(xs), num=self.num_state).tolist()
        else:
            assert self.scale > 1., 'Parameter scale should be larger than 1.'
            means = [self.mean * self.scale ** i for i in range(self.num_state)]

        self.logger.debug('Mean values at different state: {}'.format(means))

        gen_cost_mat = self.get_gen_cost_mat(xs, means)  # store costs in different states, different time slots, start from 0
        smooth_cost_list = self.get_smooth_cost_mat() # smooth costs for different state sequences, indexed by integer converted by the binary sequence

        self.costs[0, :] = 0

        # bottom up, dynamic program
        for i in range(1, len_xs+1): # from 1 to T
            for s in range(self.num_state): # 0, 1
                for prev_sqidx in range(size_local_sq): # from 0 to 2^(L)-1, previous L time slots
                    cur_int_bss = (prev_sqidx << 1 | s) & (size_local_sq - 1) # convert bss to integer, start from 0

                    cur_smooth_cost = smooth_cost_list[cur_int_bss] if i > 2 else self.smooth_cost(int2bss(cur_int_bss, self.local_len)[:i])
                    tmp_cost = self.costs[i - 1, prev_sqidx] + gen_cost_mat[s, i - 1] + self.gamma_1 * cur_smooth_cost
                    self.logger.debug('cureent int={:0{width}}, current bss={:0>{width}}, prev int={:0{width}}, prev bss={:0>{width}}, smooth cost={:.4f}, tmp cost={:.4f}, current gen cost={:.4f}, prev cost={:4f}.'.format(cur_int_bss, str(bin(cur_int_bss)[2:]), prev_sqidx, str(bin(prev_sqidx)[2:]), cur_smooth_cost, tmp_cost, gen_cost_mat[s, i - 1], self.costs[i - 1, prev_sqidx], width=self.local_len))
                    if tmp_cost < self.costs[i, cur_int_bss]:
                        self.costs[i, cur_int_bss] = tmp_cost

        # print(gen_cost_mat)
        state_seq = []
        prev_sqidx = np.argmin(self.costs[-1, :])
        # print(prev_sqidx)
        state_seq = int2bss(prev_sqidx, self.local_len) + state_seq
        # backtrack to find optimal path
        for j in range(len_xs + 1 - self.local_len, 1, -(self.local_len - 1)):
            prev_sqidx = np.argmin(self.costs[j, :])
            state_seq = int2bss(prev_sqidx, self.local_len)[:2] + state_seq
            # print(prev_sqidx, bin(prev_sqidx)[2:].format('3d'), state_seq, j)

        return state_seq

    def get_smooth_cost_mat(self):
        smooth_cost_mat = []
        for i in range(self.num_state ** self.local_len):
            smooth_cost_mat.append(self.smooth_cost(int2bss(i, self.local_len)))

        return smooth_cost_mat

    def smooth_cost(self, bss):
        p = 0
        cost = 0.
        for i in range(1, len(bss)):
            if not bss[i] == bss[i - 1]:
                cost += (i - p) ** 2
                p = i

        cost += (len(bss) - p) ** 2

        return -cost

    def get_gen_cost_mat(self, xs, means):
        """
        compute generation cost with poisson distribution
        :param xs:
        :param means:
        :return:
        """

        gen_cost_mat = np.zeros(shape=[self.num_state, len(xs)], dtype=np.float32)

        for i in range(self.num_state):
            gen_cost_mat[i] = poisson.pmf(xs, means[i])

        assert gen_cost_mat.min() > 0., 'The calculated generation probability <= 0. Please reset a new propriate scale value.'

        return -np.log(gen_cost_mat)

class MultiBursty(object):
    """
    Algorithm 2: model multiple stream
    """
    def __init__(self, scale, gamma_1, gamma_2, mean, num_state=2):
        self.mean = mean
        self.num_state = num_state
        self.scale = scale
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.logger = logging.getLogger('fei')

    def optimize(self):
        raise NotImplementedError

    def get_gen_cost_mat(self, xs, means):
        """
        compute generation cost with poisson distribution
        :param xs:
        :param means:
        :return:
        """

        gen_cost_mat = np.zeros(shape=[self.num_state, len(xs)], dtype=np.float32)

        for i in range(self.num_state):
            gen_cost_mat[i] = poisson.pmf(xs, means[i])

        assert gen_cost_mat.min() > 0., 'The calculated generation probability <= 0. Please reset a new propriate scale value.'

        return -np.log(gen_cost_mat)

if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan, linewidth=3000)

    xs = [10, 70, 60, 60, 70, 10, 10, 119, 120, 13, 10]
    local_len = 3  # number of states
    gamma_1 = .1  # trade-off parameter between generation cost and transition cost
    scale = 1.2  # scale factor between two adjacent states

    model = SingleBursty(local_len=local_len, scale=scale, gamma_1=gamma_1)

    state_seq = model.optimize(xs)

    print('Optimal state sequence, ', state_seq)
