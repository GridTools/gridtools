# -*- coding: utf-8 -*-

from scipy.stats import ttest_ind

from perftest import logger


def ttest(result_a, result_b):
    p_values = []
    for t_a, t_b in zip(result_a.times, result_b.times):
        assert t_a.stencil == t_b.stencil

        _, p_value = ttest_ind(t_a.measurements, t_b.measurements)

        p_values.append(p_value)
        logger.info(f't-test p-value for stencil "{t_a.stencil}": {p_value}')

    return list(zip(result_a.stencils, p_values))


def ttest_validate(result_a, result_b, p_value):
    passed = all(p >= p_value for _, p in ttest(result_a, result_b))
    if not passed:
        logger.info('Validation failed')
    return passed
