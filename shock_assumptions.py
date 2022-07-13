# adjust parameters for each policy in block for rerun
#
# 6 policy level inputs (assuming contract number was already removed):
#
# starting_account_to_stock_ratio,
# years_to_maturity,
# monthly_account_crediting_rate,
# monthly_valuation_discount_rate,
# mu_stock_drift,
# sigma_stock_vol
#
# example:
# shock amount vector to see effect in valuation interest rate increase: [None, None, None, 0.002, None, None]
def shock_block(df_block_of_contracts, shock_amounts):
    for i in range(len(df_block_of_contracts)):
        for col_idx, shock in enumerate(shock_amounts):
            if shock is not None:
                df_block_of_contracts.iloc[i, col_idx] += shock

    return df_block_of_contracts

