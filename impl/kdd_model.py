#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : kdd_model
# @Date : 09/04/2018 22:05:11
# @Poject : bursty
# @Author : FEI, hfut_jf@aliyun.com
# @Desc : Implementation of KDD model
# using poisson distribution to calculate generation cost

import numpy as np

from scipy.stats import poisson

import logging
from logging.config import fileConfig

fileConfig('../logging.conf')

class SimpleBursty(object):

    def __init__(self, num_state, gamma, scale=None, trans_prob=None, mean=None):
        self.mean = mean
        self.num_state = num_state
        self.scale = scale
        self.trans_prob = trans_prob
        self.gamma = gamma
        self.logger = logging.getLogger('fei')
        self.logger.debug('Initialize model.')

    def optimize(self, xs):
        """
        Calculate optimal state sequence with dynamic program.
        :param xs:
        :return:
        """

        len_xs = len(xs)

        self.costs = np.zeros(shape=[self.num_state, len_xs], dtype=np.float32)
        self.optimal_prev_state = np.zeros(shape=[self.num_state, len_xs], dtype=np.int) # store optical previous state for current batch

        if self.mean == None:
            self.mean = np.mean(xs)

        if self.trans_prob == None:
            self.trans_prob = len_xs

        self.logger.debug('Mean of input frequency sequency: {:.5f}'.format(self.mean))

        if self.scale == None:
            means = np.linspace(self.mean, np.max(xs), num=self.num_state).tolist()
        else:
            assert self.scale > 1., 'Parameter scale should be larger than 1.'
            means = [self.mean*self.scale**i for i in range(self.num_state)]

        self.logger.debug('Mean values at different state: {}'.format(means))

        gen_cost_mat = self.get_gen_cost_mat(xs, means) # store costs in different states, different batches
        trans_cost_pair = self.get_trans_cost_mat(len_xs) # transition costs between different states

        self.costs[:, 0] = gen_cost_mat[:, 0]

        # bottom up, dynamic program
        for i in range(1, len_xs):
            for cur_sidx in range(self.num_state):
                self.costs[cur_sidx, i] = np.inf
                for prev_sidx in range(self.num_state):
                    gen_cost = gen_cost_mat[cur_sidx, i]
                    trans_cost = trans_cost_pair[prev_sidx, cur_sidx]
                    tmp_cost = gen_cost + trans_cost + self.costs[prev_sidx, i - 1]
                    if tmp_cost < self.costs[cur_sidx, i]:
                        self.costs[cur_sidx, i] = tmp_cost
                        self.optimal_prev_state[cur_sidx, i] = prev_sidx
                # print(i, cur_sidx, self.costs[cur_sidx, i])

        state_seq = []
        prev_sidx = np.argmin(self.costs[:, -1])
        state_seq.append(prev_sidx)
        # backtrack to find optimal path
        for j in range(len_xs - 2, 0, -1):
            state_seq.insert(0, self.optimal_prev_state[prev_sidx, j])
            prev_sidx = self.optimal_prev_state[prev_sidx, j]

        return state_seq

    def get_trans_cost_mat(self, len_xs):
        """
        compute transit cost matrix
        :param len_xs:
        :return:
        """

        trans_cost_mat = np.zeros(shape=[self.num_state, self.num_state], dtype=np.float32)

        for i in range(self.num_state):
            for j in range(self.num_state):
                if i >= j:
                    trans_cost_mat[i, j] = 0.
                else:
                    trans_cost_mat[i, j] = (j - i) * self.gamma * np.log(self.trans_prob)

        return trans_cost_mat

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

    xs = [10,70,60,60,70,10,10,119,120,13,10]
    num_state = 3 # number of states
    gamma = .1 # trade-off parameter between generation cost and transition cost
    scale = 1.2 # scale factor between two adjacent states

    model = SimpleBursty(num_state=num_state, gamma=gamma, scale=scale)
    # model = SimpleBursty(num_state=num_state, gamma=gamma)

    state_seq = model.optimize(xs)

    print('Optimal state sequence, ', state_seq)

