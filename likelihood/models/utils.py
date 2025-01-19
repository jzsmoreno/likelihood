import numpy as np

from likelihood.tools import cal_average


class FeaturesArima:
    def forward(self, y_sum: np.ndarray, theta: list, mode: bool, noise: float):
        if mode:
            y_vec = []

            y_t = np.dot(theta, y_sum)

            n = y_sum.shape[0]

            for i in range(n):
                try:
                    n_int = np.where(y_sum != y_sum[i])[0]
                    y_i = (y_t - np.dot(theta[n_int], y_sum[n_int])) / theta[i]
                    y_i += np.random.rand() * noise
                except:
                    y_i = (y_t - np.dot(theta[0:i], y_sum[0:i])) / theta[i]
                y_vec.append(y_i)
        else:
            y_t = np.dot(theta, y_sum) + y_sum[0]
            n_int = np.where(y_sum != y_sum[0])[0]
            y_i = (y_t - np.dot(theta[n_int], y_sum[n_int])) / theta[0]
            y_i += np.random.rand() * noise
            return y_i

        return np.array(y_vec)

    def integrated(self, datapoints: np.ndarray):
        datapoints = self.datapoints
        y_sum = list(np.diff(datapoints, self.d))
        y_sum.insert(0, datapoints[0])

        return np.array(y_sum)

    def average(self, datapoints: np.ndarray):
        y_sum_average = cal_average(datapoints)
        y_sum_eps = datapoints - y_sum_average

        return y_sum_eps
