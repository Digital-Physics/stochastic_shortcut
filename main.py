from numpy import random
# import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import math


def undiscounted_payout(price, strike):
    return max(price-strike, 0)


def stochastic_factor(mu, sigma):
    return math.exp(mu + sigma*random.normal())


def stochastic_runs(num_sims, num_periods, strike_price, mu, sigma):
    stochastic_paths = []
    payouts = []
    for stochastic_run_idx in range(num_sims):
        stock_value = 1  # assume all paths start with a value of 1
        temp_run = [stock_value]
        for time_period in range(num_periods-1):
            stock_value *= stochastic_factor(mu, sigma)
            # put some path-dependent logic in here too? something so it seems like simulating many paths is really helpful
            # something so we know there is not an easy, known, closed-form solution like Black-Scholes (which is discounted)
            temp_run.append(stock_value)
        # compute average without doing a sum over a list of values at the end
        #payouts.append(undiscounted_payout(stock_value, strike_price))
        if stochastic_run_idx > 0:
            avg_payout = avg_payout + (undiscounted_payout(stock_value, strike_price) - avg_payout)/(stochastic_run_idx+1)
        else:
            avg_payout = undiscounted_payout(stock_value, strike_price)
        stochastic_paths.append(temp_run)

    data_matrix = np.zeros((num_periods, num_sims))
    for column in range(num_sims):
        data_matrix[:, column] = stochastic_paths[column]
        df = pd.DataFrame(data_matrix, columns=["run_"+str(i) for i in range(num_sims)])

    return df, avg_payout  # sum(payouts)/num_sims


start = time.time()
df, avg = stochastic_runs(1000, 30*12, 100, 0.01, 0.1)
end = time.time()
print("elapsed time:", end-start)
print("computed average", avg)

# plt.plot([i for i in range(30*12)], df)
# plt.show()