import os
import numpy as np
from espei.analysis import truncate_arrays
from statistics import harmonic_mean
from decimal import *
import math

def harmonic_mean_estimator(trace_file, lnprob_file, nburn, log=False):
    """
    Return evidence/marginal likelihood from MCMC using the harmonic mean estimation method

    Parameters
    ----------
    trace_file : string
        file name of trace file from MCMC
    lnprob_file : string
        file name of lnprob file from MCMC
    nburn : int
        number of burn-in steps, number of steps that will be removed from estimating evidence

    Returns
    -------
    evidence value: decimal
        the default is not using log value, if 'log=True', return ln(evidence) value

    """
    trace = np.load(trace_file)
    lnprob = np.load(lnprob_file)
    trace, lnprob = truncate_arrays(trace, lnprob)
    samples = np.ascontiguousarray(lnprob[:,nburn:])
    samples_list=[]
    for i in range(0, len(samples)):
        for j in samples[i]:
            samples_list.append(Decimal(j).exp())
    if log == True:
        evidence=Decimal(harmonic_mean(samples_list)).ln()
        return evidence
    else:
        evidence=Decimal(harmonic_mean(samples_list))
        return evidence

def bayes_factor_selection(evidence_1, evidence_2, log=False, use_log10_K=False):
    """
    Return Bayes factor and strength of evidence

    Parameters
    ----------
    evidence_1 : decimal
        evidence of model 1
    evidence_2 : decimal
        evidence of model 2
    log : default is False
        set 'log=True' if use ln(evidence) as input for calculating Bayes factor

    Returns
    -------
    Bayes factor and strength of evidence

    """
    for i in evidence_1, evidence_2:
        if type(i) is not Decimal:
            print('Please use Decimal type of evidence value for calculations')
    if log == True:
        K=evidence_1.exp()/evidence_2.exp()
    else:
        K=evidence_1/evidence_2
    if use_log10_K == True:
        log10_K=math.log10(K)
        if log10_K < 0:
            print('Bayes factor is', log10_K)
            print('Model 2 is favored by data')
        if 0 <= log10_K < 0.5:
            print('Bayes factor is', log10_K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Not worth more than a bare mention')
        if 0.5 <= log10_K < 1:
            print('Bayes factor is', log10_K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Substantial')
        if 1 <= log10_K < 2:
            print('Bayes factor is', log10_K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Strong')
        if log10_K >= 2:
            print('Bayes factor is', log10_K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Decisive')
    else:
        if K < 1:
            print('Bayes factor is', K)
            print('Model 2 is favored by data')
        if 1 <= K < 3.2:
            print('Bayes factor is', K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Not worth more than a bare mention')
        if 3.2 <= K < 10:
            print('Bayes factor is', K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Substantial')
        if 10 <= K < 100:
            print('Bayes factor is', K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Strong')
        if K >= 100:
            print('Bayes factor is', K)
            print('Model 1 is favored by data')
            print('Strength of evidence: Decisive')
