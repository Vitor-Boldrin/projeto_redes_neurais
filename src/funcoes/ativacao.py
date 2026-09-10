import numpy as np

def _sigmoide(theta,x) -> float:
    return 1/(1+np.e**(-np.dot(theta,x)) )