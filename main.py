import random
import time
import numpy as np
import math
import csv
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# if you already have a .csv of training data or
# if you want to skip the computationally expensive step of generating training data and jump to the ML fitting step, set flag to False
# generate_data_flag = False
generate_data_flag = True


# we will model a Guaranteed Minimum Accumulation Benefit type of contract
# this GMAB can be looked at as a put option (with some additional contract features that make it exotic)
def un_discounted_payout(account_value, stock_value):
    return max(account_value-stock_value, 0)


def stochastic_factor(mu, sigma):
    return math.exp(mu + sigma*np.random.normal())

# side thoughts:
# using policyholders makes the shortcut model nontrivial;
# we can't do a high-dimensional lattice of all possible inputs and interpolate... or can we?
# does the curse of dimensionality come into play?
# it may not hit this model yet, especially since we are generating the training data we want
# will our XGBoost model need to work with sparse, non-symmetric training model points?
# if we just had mu-sigma-strike, creating a lattice of training data & interpolate between 8 neighbors' Ys in 3-d lattice is possible
# how would you locate near points in general, when not in a lattice? Dot product or cosine similarity ranking?


# generate some fake policy parameters for our model inputs
def generate_contracts(num_of_accounts=100):
    # this is really just training data, not a block of business
    # the economic stock scenarios will be different for each policy in the training data
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
        # monthly valuation discount_rate
        discount_rate = random.choice([0.002, 0.004, 0.006, 0.008, 0.01])

        # not really a block of business, but rather a set of training data
        block_of_contracts.append([contract_num, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate])

    return block_of_contracts


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

            # we put some path-dependent logic in here too so it seems like simulating many paths is really helpful for valuation
            # ...something so we know there is not an easy, known, closed-form solution like Black-Scholes, etc.
            # here we made an option (on the GMAB option) that ratchets up the account value at a few dates close to maturity
            # the starting random av-to-stock x time_to_maturity ratios we generate below may not always be...
            # statistically reasonable considering this policy feature, but i think this is ok, especially for demo purposes
            # the question remains: can we shortcut the number that comes out of this stochastic, many-path, large-compute model?
            if (num_periods - time_period) in [17, 29, 41, 53] and stock_value > account_value:
                account_value = max(temp_run_stock[-6], stock_value)
            else:
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

    return np.round(option_val, 4)


# side thought:
# should we look at Sobol sequences for generating data in a way that nicely fills in the gaps in the high dimensional space?
# should we sample certain inputs that are more variable and/or influential in driving the results?
def create_training_data(block_of_contracts):
    # stochastic stock parameters:
    #   a deterministic drift mu component and
    #   a stochastic volatility parameter

    # 10 values of monthly drift; some represent positive bull markets; some negative represent negative bear markets
    mus = np.round(np.arange(-0.01, 0.01, 0.002), 3)
    # 5 values for volatility magnitude; 0 is deterministic; rest add noise to stock return
    sigmas = np.round(np.arange(0, 0.1, 0.02), 2)

    training_data = []

    for policy_params in block_of_contracts:
        # stock market is defined by an average period drift mu of one stock, and a volatility parameterized by sigma
        for mu in mus:
            for sigma in reversed(sigmas):
                contract_num, account_to_stock_ratio, time_to_maturity, crediting_rate, discount_rate = policy_params
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


if generate_data_flag:
    policy_details = generate_contracts()
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

    print(f"run time: {len(policy_details)} option contracts, 50 different stock environments(minutes):{(global_end-global_start)/60}")

# Machine Learning model below
print()
print("Machine Learning Model: XGBoost")
print("import training data from stochastic model simulations...")
df = pd.read_csv("stochastic_training_data.csv")
print()
print("review data:")
pd.set_option("display.max_columns", None)
print(df.head())
print("shape:", df.shape)
print("count of missing values by column:")
print(df.isnull().sum())
print("statistics by column:")
print(df.describe())
print("data types by column, memory used, and other info:")
print(df.info())

print()
print("drop contract number column from training data...")
df.drop('contract_num', axis=1, inplace=True)

print()
print("split data into X, our model inputs, and Y, our target variable...")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print()
print("split data into training and test data... training data will go through 10-fold cross validation")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbr = XGBRegressor()
print()
print("XGBoost Regression model details:")
print(xgbr)

# we don't do any data clean-up or normalization, but it didn't seem to affect the model accuracy
print()
print("fit model...")
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)
print("Training score:", score)

# cross validation should fit as well
cv_scores = cross_val_score(xgbr, X_train, y_train, cv=10)
print("C-V training scores:", cv_scores)
print("Cross-validation average:", cv_scores.mean())

# predict results
print()
print("Predict results on test data and note the time needed...")
ml_start = time.time()
y_pred = xgbr.predict(X_test)
ml_end = time.time()

print("Here's a time stat for comparison to full-blown stochastic model:")
print(f"XGBoost fitted model run time for {len(X_test)} option contracts (seconds): {(ml_end-ml_start)}")
print("the average time to do a 10,000 path stochastic run valuation on one contract was several seconds")

if generate_data_flag:
    stochastic_time = (global_end-global_start)/len(policy_details)/50  # each policy has 50 (mu, sigma) stochastic run pairs
    xgboost_time = (ml_end-ml_start)/len(X_test)
    speed_up_factor = stochastic_time/xgboost_time

    with open("time_speed_up.txt", "a") as f:
        print(f"A trained XGBoost model predicts {speed_up_factor} times faster than the 10,000-projection stochastic model!", file=f)

# get some metrics
print()
print("accuracy metrics on XGBoost Regression model:")
mse = mean_squared_error(y_test, y_pred)
print("MSE: %0.2f" % mse)
print("RMSE: %0.2f" % (mse**(1/2)))

# compare the models
print("test set: prediction vs estimated ground truth:")
df2 = pd.DataFrame({"prediction": y_pred, "estimated ground truth": y_test})
df2['error percentage (on Estimated stochastic ground truth)'] = df2["prediction"]/df2["estimated ground truth"]-1
print(df2)
df2.to_csv("test_set_XGBoost_predictions_vs_stochastic_truths.csv", index=False)

# visualize predictions
print()
print(".png written to visualize XGBoost prediction vs. stochastic (non-converged, estimated) ground truth")
x_axis = range(len(y_test))
plt.plot(x_axis, y_pred, label="ML model prediction")
plt.plot(x_axis, y_test, label="Estimated ground truth")
plt.title("Stochastic Shortcut Accuracy Comparison")
plt.legend()
plt.savefig('stochastic_shortcut_accuracy.png')
plt.show()

