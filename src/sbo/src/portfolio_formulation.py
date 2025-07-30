import numpy as np
import pandas as pd


def load_data(file_path="assets_sample.csv"):
    #  CSV columns: asset_id, expected_return, variance
    df = pd.read_csv(file_path)
    return df

# Build covariance matrix (for demonstration, identity or simple diagonal)
def build_covariance(df):
    # Replace this with your actual covariance matrix of asset returns
    n = len(df)
    cov = np.diag(df['variance'].values)
    return cov

# Portfolio problem formulation
def formulate_portfolio_problem(df, risk_aversion=0.5, penalty_factor=100.0):
    n = len(df)
    
    # Decision variables: x_i ∈ {0,1} (buy or don't buy asset i)
    # Objective: minimize risk - risk_aversion * return
    # Quadratic term = risk part = x^T Cov x
    # Linear term = - risk_aversion * expected return^T x
    cov = build_covariance(df)
    returns = df['expected_return'].values
    
    Q = cov  # quadratic term matrix (risk)
    c = -risk_aversion * returns  # linear term (negative to maximize return)
    
    # Constraint example: invest in exactly k assets
    k = 2
    # Constraint: sum x_i = k --> penalty term: penalty_factor * (sum x_i - k)^2
    # Expand penalty term:
    # = penalty_factor * (x^T 11^T x - 2k 1^T x + k^2)
    # where 1 is vector of ones
    
    one_vec = np.ones(n)
    penalty_quadratic = penalty_factor * np.outer(one_vec, one_vec)
    penalty_linear = -2 * penalty_factor * k * one_vec
    penalty_constant = penalty_factor * k**2
    
    # Combine objective + penalty
    Q_total = Q + penalty_quadratic
    c_total = c + penalty_linear
    constant = penalty_constant
    
    # Return the quadratic problem data
    return Q_total, c_total, constant

# print objective in readable format
def print_objective(Q, c, const):
    print("Quadratic term Q:\n", Q)
    print("Linear term c:", c)
    print("Constant:", const)

if __name__ == "__main__":
    # Load example data
    df = pd.DataFrame({
        'asset_id': ['A1', 'A2', 'A3'],
        'expected_return': [0.1, 0.2, 0.15],
        'variance': [0.05, 0.1, 0.07]
    })
    
    Q, c, const = formulate_portfolio_problem(df)
    print_objective(Q, c, const)

