"""
https://github.com/rasmusab/bayesianprobabilitiesworkshop/blob/master/Exercise%201.ipynb

Swedish Fish Incorporated is the largest Swedish company delivering fish by mail order.
They are now trying to get into the lucrative Danish market by selling one year Salmon subscriptions.
The marketing department have done a pilot study and tried the following marketing method:

A: Sending a mail with a colorful brochure that invites people to sign up for a one year salmon subscription.

The marketing department sent out 16 mails of type A.
Six Danes that received a mail signed up for one year of salmon
and marketing now wants to know, how good is method A?"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Number of random draws from the prior
n_draws = 10000

# Here you sample n_draws draws from the prior into a pandas Series (to have convenient
# methods available for histograms and descriptive statistics, e.g. median)
prior_rate = pd.Series(np.random.uniform(0, 1, size=n_draws))
# plt.hist(prior_rate)
# plt.show()


# Defining the generative model
def gen_model(prob):
    mod = np.random.binomial(16, prob)
    # print(mod)
    return mod


#  the generative model
subscribers = list()

# Simulating the data
for p in prior_rate:
    # print(p)
    subscribers.append(gen_model(p))

# Observed data
observed_data = 6

# Here you filter off all draws that do not match the data.
post_rate = prior_rate[list(map(lambda x: x == observed_data, subscribers))]

plt.hist(post_rate)
plt.show()

# See that we got enought draws left after the filtering.
# There are no rules here, but you probably want to aim for >1000 draws.

# Now you can summarize the posterior, where a common summary is to take the mean or the median posterior,
# and perhaps a 95% quantile interval.


print(
    'Number of draws left: %d, Posterior mean: %.3f, Posterior median: %.3f, Posterior 95%% quantile interval: %.3f-%.3f' %
    (len(post_rate), post_rate.mean(), post_rate.median(), post_rate.quantile(.025), post_rate.quantile(.975)))

"""What’s the probability that method A is better than telemarketing?"""
print(np.mean(post_rate > 0.2))

"""If method A was used on 100 people what would be number of sign-ups?"""
signups = list()

for p in post_rate:
    signups.append(np.random.binomial(100, p))


plt.hist(signups)
plt.show
print('Sign-up Posterior 95%% quantile interval: %.3f-%.3f' % (post_rate.quantile(.25), post_rate.quantile(.975)))

