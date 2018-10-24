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

def int2bssarray(digit, M, L):
    bss = [int(z) for z in format(digit, '0{:d}b'.format(M*L))]
    bssarray = [[bss[i] for i in range(m, len(bss), M)] for m in range(M) ]

    return bssarray

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
        size_local_int = self.num_state ** self.local_len

        self.costs = np.zeros(shape=[len_xs+1, size_local_int], dtype=np.float32) # the extra entries in first dimension are for initialization before index i=1
        self.costs.fill(np.inf) # fill in all entries of costs with infinity
        self.costs[0, :] = 0

        if self.mean == None:
            self.mean = np.mean(xs)

        self.logger.debug('Mean of input frequency sequency: {:.5f}'.format(self.mean))

        if self.scale == None:
            means = np.linspace(self.mean, np.max(xs), num=self.num_state).tolist()
        else:
            assert self.scale > 1., 'Parameter scale should be larger than 1.'
            means = [self.mean * self.scale ** i for i in range(self.num_state)]

        self.logger.debug('Mean values at different state: {}'.format(means))

        gen_cost_mat = self.get_gen_costs(xs, means)  # store costs in different states, different time slots, start from 0
        smooth_cost_list = self.get_smooth_costs() # smooth costs for different state sequences, indexed by integer converted by the binary sequence

        # bottom up, dynamic program
        for i in range(1, len_xs + 1): # from 1 to T
            for s in range(self.num_state): # 0, 1
                for prev_sqidx in range(size_local_int): # from 0 to 2^(L)-1, previous L time slots
                    cur_int_bss = (prev_sqidx << 1 | s) & (size_local_int - 1) # convert bss to integer, start from 0

                    cur_smooth_cost = smooth_cost_list[cur_int_bss] if i > self.local_len - 1 else self.smooth_cost(int2bss(cur_int_bss, self.local_len)[-i:])
                    tmp_cost = self.costs[i - 1, prev_sqidx] + gen_cost_mat[s, i - 1] + self.gamma_1 * cur_smooth_cost
                    self.logger.debug('cureent int={:0{width}}, current bss={:0>{width}}, prev int={:0{width}}, prev bss={:0>{width}}, smooth cost={:.4f}, tmp cost={:.4f}, current gen cost={:.4f}, prev cost={:4f}.'.format(cur_int_bss, str(bin(cur_int_bss)[2:]), prev_sqidx, str(bin(prev_sqidx)[2:]), cur_smooth_cost, tmp_cost, gen_cost_mat[s, i - 1], self.costs[i - 1, prev_sqidx], width=self.local_len))
                    if tmp_cost < self.costs[i, cur_int_bss]:
                        self.costs[i, cur_int_bss] = tmp_cost

        # print(gen_cost_mat)
        prev_sqidx = np.argmin(self.costs[-1, :])
        # print(prev_sqidx)
        state_seq = int2bss(prev_sqidx, self.local_len)
        # backtrack to find optimal path
        for j in range(len_xs + 1 - self.local_len, 1, -(self.local_len - 1)):
            s = state_seq[0]
            tmp_cost = np.inf
            tmp_seq = int2bss(0, self.local_len)
            for k in range(s, self.num_state ** self.local_len, self.num_state):
                if self.costs[j, k] < tmp_cost:
                    tmp_cost = self.costs[j, k]
                    tmp_seq = int2bss(k, self.local_len)
            state_seq = tmp_seq[:self.local_len - 1] + state_seq


        # for j in range(len_xs + 1 - self.local_len, 1, -(self.local_len - 1)):
        #     prev_sqidx = np.argmin(self.costs[j, :])
        #     state_seq = int2bss(prev_sqidx, self.local_len)[:self.local_len - 1] + state_seq
        #     # print(prev_sqidx, bin(prev_sqidx)[2:].format('3d'), state_seq, j)

        return state_seq

    def get_smooth_costs(self):
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

    def get_gen_costs(self, xs, means):
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
    S: num_state
    M: num_stream
    T: num_time_slot
    L: length_local_window
    Code of int of bss, suppose M=3, L=3
        L_1 L_2 L_3
    M_1 0   3   6
    M_2 1   4   7
    M_3 2   5   8
    """
    def __init__(self, local_len, num_stream, scale, gamma_1, gamma_2, mean, num_state=2):
        self.mean = mean
        self.num_state = num_state
        self.local_len = local_len
        self.num_stream = num_stream
        self.scale = scale
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.logger = logging.getLogger('fei')

    def optimize(self, xs):
        len_xs = len(xs) # TxM
        size_local_bit = self.local_len * self.num_stream # LxM
        size_local_int = self.num_state ** size_local_bit # S^(LxM)

        # the extra entries in first dimension are for initialization before index i=1
        # (T+1)xS^(L*M)
        self.costs = np.zeros(shape=[len_xs + 1, size_local_int], dtype=np.float32)
        self.costs.fill(np.inf)  # fill in all entries of costs with infinity
        self.costs[0, :] = 0

        if self.mean == None:
            self.mean = np.mean(xs, axis=0) # 1xM

        self.logger.debug('Mean of input frequency sequency: {:s}'.format(str(self.mean)))

        if self.scale == None:
            means = np.array([np.linspace(self.mean[i], np.max(xs[i]), self.num_state) for i in range(self.num_stream)]) # MxS
        else:
            assert self.scale > 1., 'Parameter scale should be larger than 1.'
            means = np.array([self.mean * self.scale ** i for i in range(self.num_state)]).transpose() # MxS

        self.logger.debug('Mean values at different state: {}'.format(means))

        gen_cost_mat = self.get_gen_costs(xs, means) # SxTxM
        smooth_cost_list = self.get_smooth_costs() # 1xS^(M*L)

        for i in range(1, len_xs + 1): # from 1 to T
            for m in range(self.num_state ** self.num_stream): # from 0 to S^M-1
                cross_stream_bss = int2bss(m, self.num_stream) # m states in current time slot
                cur_gen_cost = np.sum([gen_cost_mat[s, i - 1, mid] for mid, s in enumerate(cross_stream_bss)]) # generation cost of m states in current time slot
                cross_stream_cost = np.sum([1 if not cross_stream_bss[x1] == cross_stream_bss[x2] else 0 for x1 in range(self.num_stream) for x2 in range(x1+1, self.num_stream)])
                for prev_sqidx in range(size_local_int): # access all bits (integers) of previous time slot, from 0 to S^(L*M)-1
                    # derive corresponding integers of current local window
                    cur_int_bss = (prev_sqidx << self.num_stream | m) & (size_local_int - 1)
                    # x = int2bssarray(prev_sqidx, self.num_stream, self.local_len)
                    # x = int2bssarray(cur_int_bss, self.num_stream, self.local_len)

                    cur_smooth_cost = smooth_cost_list[cur_int_bss] if i > self.local_len - 1 else self.smooth_cost(int2bss(cur_int_bss, self.local_len)[-i * self.num_stream:])

                    tmp_cost = self.costs[i - 1, prev_sqidx] + cur_gen_cost + self.gamma_1 * cur_smooth_cost + self.gamma_2 * cross_stream_cost
                    print(int2bssarray(cur_int_bss, self.num_stream, self.local_len))
                    print(cross_stream_bss, cur_gen_cost, cur_smooth_cost, self.costs[i - 1, prev_sqidx])

                    if tmp_cost < self.costs[i, cur_int_bss]:
                        self.costs[i, cur_int_bss] = tmp_cost

        # print(gen_cost_mat)
        prev_sqidx = np.argmin(self.costs[-1, :])
        # print(prev_sqidx)
        prev_seq = int2bss(prev_sqidx, self.local_len)
        state_seq = [[prev_seq[i] for i in range(m, len(prev_seq), self.num_stream)] for m in range(self.num_stream)]
        # backtrack to find optimal path
        for j in range(len_xs + 1 - self.local_len, 1, -(self.local_len - 1)):
            cur_bits = [sub_seq[0] for sub_seq in state_seq]
            cur_int = np.sum([s * self.num_state ** (len(cur_bits) - i - 1) for i, s in enumerate(cur_bits)])
            tmp_cost = np.inf
            tmp_seq = int2bss(0, size_local_bit)
            for k in range(cur_int, size_local_int, self.num_state ** self.num_stream):
                if self.costs[j, k] < tmp_cost:
                    tmp_cost = self.costs[j, k]
                    tmp_seq = int2bss(k, size_local_bit)

            state_seq = [[tmp_seq[x] for x in range(m, len(tmp_seq), self.num_stream)] + state_seq[m] for m in range(self.num_stream)]

        return state_seq

    def get_gen_costs(self, xs, means):
        """
        compute generation cost with poisson distribution
        :param xs: TxM
        :param means:
        :return:
        """

        gen_cost_mat = np.zeros(shape=[self.num_state, len(xs), self.num_stream], dtype=np.float32) # SxTxM

        for s in range(self.num_state):
            for m in range(self.num_stream):
                gen_cost_mat[s, :, m] = poisson.pmf(xs[:, m], means[m, s])

        assert gen_cost_mat.min() > 0., 'The calculated generation probability <= 0. Please reset a new propriate scale value.'

        return -np.log(gen_cost_mat)

    def get_smooth_costs(self):
        smooth_cost_mat = [] # S^(M*L)
        bit_len = self.local_len * self.num_stream
        for i in range(self.num_state ** (self.local_len * self.num_stream)):
            smooth_cost_mat.append(self.smooth_cost(int2bss(i, bit_len)))

        return smooth_cost_mat

    def get_smooth_cost_mat(self):
        smooth_cost_mat = []
        for i in range(self.num_state ** (self.num_stream * self.local_len)):
            smooth_cost_mat.append(self.smooth_cost(int2bss(i, self.local_len * self.num_stream)))

        return smooth_cost_mat

    def smooth_cost(self, bss):

        cost = 0.
        bss_len = len(bss)
        for m in range(self.num_stream): # calculate smooth cost of each subsequence in m-th stream
            p = m # p, start idx of subsequence with same bits
            for i in range(self.num_stream + m, bss_len, self.num_stream):
                if not bss[i] == bss[i - self.num_stream]:
                    cost += ((i - p) / self.num_stream) ** 2
                    p = i

            cost += ((m + bss_len - p) / self.num_stream) ** 2

        return -cost

if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan, linewidth=3000)

    xs = np.array([10, 70, 60, 60, 70, 10, 10, 119, 120, 13, 10])
    local_len = 3  # number of states
    gamma_1 = .1  # trade-off parameter between generation cost and transition cost
    scale = 1.2  # scale factor between two adjacent states

    model = SingleBursty(local_len=local_len, scale=scale, gamma_1=gamma_1)

    state_seq = model.optimize(xs)

    print('Optimal state sequence, ', state_seq)

    xs = np.array([[10, 11, 9], [70, 40, 50], [60, 30, 40], [60, 50, 55], [70, 60, 50], [10, 20, 13],
          [10, 8, 9], [119, 115, 117], [120, 121, 110], [13, 10, 11], [10, 9, 12]])
    local_len = 3
    num_stream = 3
    gamma_1 = .1
    gamma_2 = .1
    scale = 1.2

    model = MultiBursty(local_len=local_len, num_stream=num_stream, scale=scale, gamma_1=gamma_1, gamma_2=gamma_2, mean=None)

    state_seq = model.optimize(xs)
    print(state_seq)

    # for m in range(2 ** 3):
    #     for prev_sqidx in range(2 ** (3 * 3)):  # access all bits (integers) of previous time slot
    #         # derivate corresponding integers of current local window
    #         cur_int_bss = (prev_sqidx << 3 | m) & (2 ** 9 - 1)
    #         print(prev_sqidx << 3, bin(prev_sqidx << 3)[2:], int2bss(prev_sqidx, 9)[:6])
    #         print(cur_int_bss, bin(cur_int_bss), int2bss(cur_int_bss, 9))
    #
    #
    # cost = 0.
    # bss = [0, 0, 1, 1, 0, 1, 1, 1, 0]
    # for m in range(3):
    #     p = 0+m
    #     for i in range(3+m, 9, 3):
    #         print('i', i)
    #         if not bss[i] == bss[i - 3]:
    #             cost += ((i - p) / 3) ** 2
    #             print('cost', cost)
    #             p = i
    #
    #     cost += ((m+9 - p) / 3) ** 2
    #
    # print(cost)