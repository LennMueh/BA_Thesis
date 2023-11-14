import numpy as np
import math


def scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, foo):

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = foo(i, j)
            if np.isnan(score):
                score = 0
            scores[i][j] = score

    flat_scores = [float(item) for sublist in scores for item in sublist if not math.isnan(float(item))]

    # grab the indexes of the highest suspicious_num scores
    if suspicious_num >= len(flat_scores):
        flat_indexes = range(len(flat_scores))
    else:
        flat_indexes = np.argpartition(flat_scores, -suspicious_num)[-suspicious_num:]

    suspicious_neuron_idx = []
    for idx in flat_indexes:
        # unflatten idx
        i = 0
        accum = idx
        while accum >= len(scores[i]):
            accum -= len(scores[i])
            i += 1
        j = accum

        if trainable_layers is None:
            suspicious_neuron_idx.append((i, j))
        else:
            suspicious_neuron_idx.append((trainable_layers[i], j))

    return suspicious_neuron_idx


def tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    def tarantula(i, j):
        return float(float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j])) / \
            (float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j]) + float(num_cs[i][j]) / (num_cs[i][j] + num_us[i][j]))

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, tarantula)


def ochiai_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    def ochiai(i, j):
        return float(num_cf[i][j]) / ((num_cf[i][j] + num_uf[i][j]) * (num_cf[i][j] + num_cs[i][j])) **(.5)

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, ochiai)


def dstar_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, star):
    def dstar(i, j):
        return float(num_cf[i][j]**star) / (num_cs[i][j] + num_uf[i][j])

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, dstar)

