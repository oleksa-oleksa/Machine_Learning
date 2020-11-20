import numpy as np
from matplotlib import pyplot as plt

mu_G, sigma_G = -1, 1  # mean and standard deviation
G = np.random.normal(mu_G, sigma_G, 1000)
mu_G = np.mean(G) # compute mean with numpy (make no sense, but it is a task)
mu_K, sigma_K = 1, 1
np.random.seed(42)
K = np.random.normal(mu_K, sigma_K, 1000)

# Create dict (wtf?!)
D = dict()
D['green'] = G
D['black'] = K

# Prepare the plot
f = plt.figure(figsize=(8, 6))

# Green distribution
nums, _, _ = plt.hist(G, bins=50, label='green', color='g')
max_G = np.max(nums)

# Gray transparent distribution
nums, _, _ = plt.hist(K, bins=50, label='black', color='k', alpha=0.3)
max_K = np.max(nums)


# Mean G
args = ([mu_G, mu_G], [0, max_G])
_ = plt.plot(*args, ls='--', linewidth=4, c='r')
# Mean K
args = ([mu_K, mu_K], [0, max_K])
_ = plt.plot(*args, ls='--', linewidth=4, c='r')

# Plot parameters
plt.title('Data histogram', fontsize=20)
plt.xlabel('X-axis')
plt.ylabel('Y-axis', rotation=0)
plt.grid(axis='x')
_ = plt.legend(loc='upper left')
plt.show()

