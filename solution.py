import pandas as pd
import numpy as np

from scipy.stats import norm, chi2


chat_id = 871302863

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    s = np.sum(x**2)
    alpha = 1 - p
    left = np.sqrt(s / chi2.ppf(1 - alpha/2, 2*n))
    right = np.sqrt(s / chi2.ppf(alpha/2, 2*n))
    return (left / np.sqrt(3), right / np.sqrt(3))
