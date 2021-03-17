import numpy as np
import pandas as pd

class KalmanControl():
    def __init__(self):
        self.x = np.zeros(4)
        self.F = np.zeros((4,4))

        self.y = 1
