import pandas as pd
import numpy as np

from scipy.stats import norm, chi2


chat_id = 871302863

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    mean = np.mean(x)
    variance = np.var(x, ddof=1)
    statistic = (n-1)*variance/(3*mean**2)
    left = chi2.ppf(p/2, n-1)/(3*statistic)
    right = chi2.ppf(1-p/2, n-1)/(3*statistic)
    return (left, right)
