from numpy import random
# import matplotlib.pyplot as plt
import time
# import pandas as pd
import numpy as np
import math
import csv


def un_discounted_payout(account_value, strike):
    return max(account_value-strike, 0)


def stochastic_factor(mu, sigma):
    return math.exp(mu + sigma*random.normal())


def stochastic_runs(num_sims, num_periods, strike_price, mu, sigma):
    stochastic_paths_stock = []
    stochastic_paths_account = []
    payouts = []
    option_activated = 0  # an option on the option

    for stochastic_run_idx in range(num_sims):
        stock_value = 1  # assume all paths start with a value of 1
        account_value = 1
        temp_run_stock = [stock_value]
        temp_run_account = [account_value]
        for time_period in range(num_periods-1):
            stochastic_mult = stochastic_factor(mu, sigma)
            stock_value *= stochastic_mult
            temp_run_stock.append(stock_value)

            # we put some path-dependent, exotic options in here so simulating many paths is actually helpful
            # ...so no easy, known, closed-form solution like Black-Scholes (note: B-S is discounted)
            # here we made a guarantee on an underlying "account value"
            # this is an embedded option on the contract which is a call option itself
            if time_period in [35, 47, 59, 71] and stock_value < 1:
                option_activated += 1
                account_value = min(max(1, temp_run_account[11], temp_run_account[23], temp_run_account[35]), stock_value)
                temp_run_account.append(account_value)
            else:
                account_value *= stochastic_mult
                temp_run_account.append(account_value)

        payouts.append(un_discounted_payout(account_value, strike_price))
        stochastic_paths_stock.append(temp_run_stock)
        stochastic_paths_account.append(temp_run_account)

    # data_matrix = np.zeros((num_periods, num_sims))
    # for column in range(num_sims):
    #     data_matrix[:, column] = stochastic_paths[column]
    #     df = pd.DataFrame(data_matrix, columns=["run_"+str(i) for i in range(num_sims)])

    print(f"account value bump-up guarantee activated on average {option_activated/num_sims} times/run.")
    return sum(payouts)/num_sims  # df, avg_payout


def create_training_data():
    # hyper-parameters:
    # 20 values of monthly drift; some positive bull market; some negative bear market
    mus = np.arange(-0.01, 0.01, 0.001)
    # 10 values for volatility magnitude; 0 is deterministic; rest add noise
    sigmas = np.arange(0, 0.1, 0.01)
    # 10 strike price levels; 0 is "in the money"; rest are "out of the money"
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
                print("mu, sigma, strike:", mu, sigma, strike)
                print("estimated, un-discounted, account/option payoff at expiration:", avg)
                print("10,000-path, 30-year, monthly, stochastic projection time:", end-start)
                print()
                training_data.append([mu, sigma, strike, avg])

    return training_data


def list_to_csv(lists, headers):
    with open("stochastic_training_data.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(lists)
        print("stochastic_training_data.csv file written!")


training_data_results = create_training_data()
list_to_csv(training_data_results, ["mu", "sigma", "strike", "avg"])

# plt.plot([i for i in range(30*12)], df)
# plt.show()

# below we will import the csv of training data and try to fit a model that estimates the option value
# we will then analyze how accurate the model is (compared to our noisy, not-fully-converged "ground truth")
# we will also compare how long the full stochastic model takes to run vs a forward pass on the fitted model
