from typing import List
import numpy as np

def computeRTG(rewards: List[float]) -> List[float]:
    """
    rewards: list of rewards [r0, r1, ..., rT]
    returns: list of RTG values [RTG0, RTG1, ..., RTG_T]
    """
    rtg = []
    cumulative = 0

    for r in reversed(rewards):
        cumulative += float(r)
        rtg.append(cumulative)

    return list(reversed(rtg))

def heuristic_policy(state: List[float], K = 0.0) -> float:
    """
    state: [tilt angle, angular vel, cart position, cart velocity]
    return: external force
    """
    theta, *_ = state

    # proportional controller
    u = K * theta
    return u

def _softmax(M: np.array) -> np.array:
    b = np.max(M)
    e_x = np.exp(M - b)
    return e_x / np.sum(e_x)

def attention(Q: np.array, K: np.array, V: np.array):
    """
    Q: Qeury Matrix
    K: Key Matrix
    V: Value Matrix
    """
    result = np.dot(Q, K.T)
    normalised = _softmax(result)
    output = np.dot(normalised, V)

    return output