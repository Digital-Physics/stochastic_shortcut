from numpy import random
# import matplotlib.pyplot as plt
import time
# import pandas as pd
import numpy as np
import math
import csv


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
            # we put some path-dependent logic in here too... something so it seems like simulating many paths is really helpful
            # something so we know there is not an easy, known, closed-form solution like Black-Scholes (note: B-S is discounted)
            # here we made a option that restores the value of the stock at the 3rd-6th anniversaries
            if time_period in [36, 48, 60, 72] and stock_value < temp_run[12]:
                stock_value = max(temp_run[0], temp_run[12], temp_run[24], temp_run[36], 1)
            temp_run.append(stock_value)
        payouts.append(undiscounted_payout(stock_value, strike_price))
        # compute average without doing a sum over a list of values at the end
        # if stochastic_run_idx > 0:
        #     avg_payout = avg_payout + (undiscounted_payout(stock_value, strike_price) - avg_payout)/(stochastic_run_idx+1)
        # else:
        #     avg_payout = undiscounted_payout(stock_value, strike_price)
        stochastic_paths.append(temp_run)

    # data_matrix = np.zeros((num_periods, num_sims))
    # for column in range(num_sims):
    #     data_matrix[:, column] = stochastic_paths[column]
    #     df = pd.DataFrame(data_matrix, columns=["run_"+str(i) for i in range(num_sims)])

    return sum(payouts)/num_sims  # df, avg_payout


def create_training_data():
    mus = np.arange(0, 0.01, 0.0005)  # monthly
    sigmas = np.arange(0, 0.07, 0.01)
    strike_prices = np.arange(0, 100, 10)

    training_data = []

    for mu in reversed(mus):
        for sigma in sigmas:
            for strike in strike_prices:
                start = time.time()
                # 10000 runs is not enough time for this to converge to the right answer
                # but if it isn't biased it may still offer a good enough "ground truth"
                avg = stochastic_runs(10000, 30*12, strike, mu, sigma)
                end = time.time()
                print()
                print("elapsed time:", end-start)
                print("mu, sigma, strike", mu, sigma, strike)
                print("estimated undiscounted option value", avg)
                training_data.append([strike, mu, sigma, avg])

    return training_data


def list_to_csv(lists, headers):
    with open("stochastic_training_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(lists)
        print("stochastic_training_data.csv file written")


training_data_results = create_training_data()
list_to_csv(training_data_results, ["strike", "mu", "sigma", "avg"])

# plt.plot([i for i in range(30*12)], df)
# plt.show()

# below we will import the csv of training data and try to fit a model that estimates the option value
# we will then analyze how accurate the model is (compared to our noisy, not-fully-converged "ground truth")
# we will also compare how long the full stochastic model takes to run vs a forward pass on the fitted model
