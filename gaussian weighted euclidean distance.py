import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl


def gaussian(dist, mu=0, sigma=1):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


test_x = np.linspace(-400, 400, 200)
ss = skl.StandardScaler()
test_x = ss.fit_transform(test_x.reshape(-1, 1))
test_y = gaussian(test_x)
print(test_y)
plt.scatter(test_x, test_y)
plt.show()
# print(gaussian(0))
