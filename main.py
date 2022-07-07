import random
import time
import numpy as np
import math
import csv
# import pandas as pd
# import matplotlib.pyplot as plt


# we will do Guaranteed Minimum Account Balance style contract
# this GMAB can be looked at as a put option
# we will value the embedded put option on a variety of policies with in a variety of economic scenarios
def un_discounted_payout(account_value, stock_value):
    return max(account_value-stock_value, 0)


def stochastic_factor(mu, sigma):
    return math.exp(mu + sigma*np.random.normal())


# using policyholders makes the shortcut model nontrivial; we can't do a high-dimensional lattice of all possible inputs and interpolate
# curse of dimensionality: our proxy data science model may need to interpolate between sparse, non-symmetric training model points
# if we just had mu-sigma-strike, create lattice of training data & interpolate between 8 neighbors' Y in 3-d lattice
def generate_contracts(num_of_accounts=100):
    block_of_contracts = []
    for contract_num in range(num_of_accounts):
        # Note: our model will not leverage the previous valuations period's known valuation for a given policy, which could be beneficial
        # Note: this is not a risk-neutral approach to option valuation

        # this can range from 0 to infinity
        account_to_stock_ratio = np.round(math.exp(0.05 + 0.2*np.random.normal()), 4)
        # years to maturity; account balance guarantee date; option date
        time_to_maturity = random.randint(1, 20)
        # this is the monthly roll-up rate on the initial account value
        crediting_rate = random.choice([0.003, 0.005, 0.007, 0.009])
        # monthly discount_rate
        discount_rate = random.choice([0.002, 0.004, 0.006, 0.008, 0.01])

        block_of_contracts.append([contract_num, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate])

    return block_of_contracts, num_of_accounts


def stochastic_runs(num_sims, mu, sigma, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate):
    stochastic_paths_stock = []
    stochastic_paths_account = []
    payouts = []
    num_periods = 12*time_to_maturity

    for stochastic_run_idx in range(num_sims):
        stock_value = 100  # assume all paths start with a value of 100
        temp_run_stock = [stock_value]
        account_value = account_to_stock_ratio*stock_value  # this ratio will change during the projection so we change the variable name
        temp_run_account = [account_value]

        for time_period in range(num_periods-1):
            stochastic_mult = stochastic_factor(mu, sigma)
            stock_value *= stochastic_mult
            temp_run_stock.append(stock_value)

            # we could add some policy features such as account value bump-ups to make it more exotic option to value
            # right now this could use a closed-form solution like the Black-Scholes formula (in a risk-neutral framework)
            account_value *= math.exp(crediting_rate)
            temp_run_account.append(account_value)

        payouts.append(un_discounted_payout(account_value, stock_value))
        stochastic_paths_stock.append(temp_run_stock)
        stochastic_paths_account.append(temp_run_account)

    undiscounted_option_value = sum(payouts)/num_sims
    discount_factor = 1/math.exp(discount_rate*num_periods)
    option_val = undiscounted_option_value*discount_factor
    print("estimated un-discounted payout value:", undiscounted_option_value)
    print("discount factor", discount_factor)
    print("PV of option val", option_val)

    return np.round(option_val, 4)  # df


# could look at Sobol sequences for generating data in a way that nicely fills in the gaps in the high dimensional space
def create_training_data(block_of_contracts):
    # hyper-parameters of stochastic model: drift, volatility, strike
    # 10 values of monthly drift; some positive bull market; some negative bear market
    mus = np.round(np.arange(-0.01, 0.01, 0.002), 3)
    # 5 values for volatility magnitude; 0 is deterministic; rest add noise to stock return
    sigmas = np.round(np.arange(0, 0.1, 0.02), 2)

    training_data = []

    for policy_details in block_of_contracts:
        # stock market is defined by an average period drift mu of one stock, and a volatility parameterized by sigma
        for mu in mus:
            for sigma in reversed(sigmas):
                contract_num, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate = policy_details
                start = time.time()
                # 10000 runs is not enough time for this to converge to the right answer
                # but if it isn't biased it may still offer a good enough "ground truth"
                option_value = stochastic_runs(10000, mu, sigma, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate)
                end = time.time()
                print("contract_num, account_to_stock_ratio, time_to_mat, crediting_rate, discount_rate, mu, sigma:")
                print(contract_num, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate, mu, sigma)
                print("estimated option value:", option_value)
                print(f"10,000-path, {time_to_maturity}-year, monthly, stochastic projection time (seconds): {end-start}")
                print()
                training_data.append([contract_num,
                                      account_to_stock_ratio,
                                      time_to_maturity,
                                      crediting_rate,
                                      discount_rate,
                                      mu,
                                      sigma,
                                      option_value])

    return training_data


def list_to_csv(lists, headers):
    with open("stochastic_training_data.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(lists)
        print("stochastic_training_data.csv file written!")


policy_details, num_of_contracts = generate_contracts()
global_start = time.time()
training_data_results = create_training_data(policy_details)
list_to_csv(training_data_results, ["contract_num",
                                    "starting_account_to_stock_ratio",
                                    "years_to_maturity",
                                    "monthly_account_crediting_rate",
                                    "monthly_valuation_discount_rate",
                                    "mu_stock_drift",
                                    "sigma_stock_vol",
                                    "option_value_time_0"])
global_end = time.time()

print(f"run time for {num_of_contracts} contracts in 50 different stock market environments(minutes): {(global_end-global_start)/60}")

# plt.plot([i for i in range(30*12)], df)
# plt.show()

# below we will import the csv of training data,
# split it in to training-val-test sets
# normalize and transform the fields using x/(x_max - x_min) or something like that
# and try to fit a model that estimates the option value
#
# we will then analyze how accurate the model is compared to the theoretically computed stochastic valuation
# we will also compare how long the full stochastic model takes to run vs a forward pass on the fitted model

