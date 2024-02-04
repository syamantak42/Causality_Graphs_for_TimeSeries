import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import minimize
from scipy.integrate import quad
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score 
# from sklearn.preprocessing import StandardScaler



def pairwise_gc( df, p_sign, nLag):
    test   = 'ssr_chi2test'
    for col1 in df:
        for col2 in [col for col in df if col != col1]:
            data = df[[col1, col2]]
            gc_res = grangercausalitytests(data, nLag, verbose = False)
           #print(dir(gc_res))
            p_values = [round(gc_res[i+1][0][test][1],4) for i in range(nLag)]
            #print(col1, col2, p_values)
            print(p_values)
            if np.min(p_values)>=p_sign:
                print(col2 + " ----> GC ----> " + col1)
    return 


def pairwise_wf( df1, dflags, epsilon = 0.001):
    nSeries = df1.shape[1]
    AdjMat = np.empty([nSeries, nSeries])
    Names = df1.columns.tolist()
    #print(df1.columns, dflags.columns)
    for col1 in Names:
        for col2 in [col for col in df1 if col != col1]:
            features = [c for c in dflags if col2 in c]
            #print(col1, col2, features)
            X = dflags[features].values
            Y = df1[col1].values
            model = LinearRegression()
            model.fit(X,Y)
            Y_pred = model.predict(X)
            mse = np.mean((Y-Y_pred)**2)
            error_gain = 1 -mse/np.var(Y)
            #ratio = r2_score(Y, Y_pred)
            AdjMat[Names.index(col1), Names.index(col2)] = error_gain
    
    AdjMat[AdjMat < epsilon ] = 0
    mask = AdjMat > AdjMat.T  # Create a boolean mask for values where AdjMat(i, j) > AdjMat(j, i)

    AdjMat[mask] = 1   # Set values where A(i, j) > A(j, i) to 0
    AdjMat[~mask] = 0   # Set values where A(i, j) <= A(j, i) to 1
    np.fill_diagonal(AdjMat, 0)
            #print(col2+" --> "+ col1 + " :  "+ str(np.round(mse, decimals = 2)))
    results = pd.DataFrame(data = AdjMat, columns = Names, index = Names)
    return results

def lasso_objective(coeffs, X, Y, alpha, group_size):
    predicted_values = np.dot(X, coeffs)
    mse = np.mean((predicted_values - Y) ** 2)

    # Add L1 regularization term (lasso penalty)
    l1_penalty = alpha * np.sum(np.abs(coeffs))

    return mse + l1_penalty

def group_lasso_objective(params, X, Y, alpha, group_size):
    num_groups = len(params) // group_size
    coefficients = params.reshape((num_groups, group_size))

    predicted_values = np.dot(X, coefficients.flatten())
    mse = np.mean((predicted_values - Y) ** 2)

    # Add group L1 regularization term (group lasso penalty)
    group_lasso_penalty = alpha * np.sum(np.linalg.norm(coefficients, ord=2, axis=1))

    return mse + group_lasso_penalty

def phi(x, group_size):
    phi_x = []
    for t in range(group_size):
        phi_x.append(np.exp(2*(np.pi)*x*t*1j))
    phi_x = np.array(phi_x)
    return phi_x 

def freq_lasso_objective(params, X, Y, alpha, group_size):
    num_groups = len(params) // group_size
    coefficients = params.reshape((num_groups, group_size))

    predicted_values = np.dot(X, coefficients.flatten())
    mse = np.mean((predicted_values - Y) ** 2)

    # Add group integral regularization term
    group_integral_penalty = 0.0
    for group_coefficients in coefficients:
        group_function = lambda x: abs(np.dot(group_coefficients, phi(x, group_size)))
        integral_result, _ = quad(group_function, -0.5, 0.5)
        group_integral_penalty += integral_result

    group_integral_penalty *= alpha

    return mse + group_integral_penalty    

def sparse_regression_scipy(X, Y, objective_func, alphaL, alphaG, alphaF, group_size):
    num_features = X.shape[1]
    num_groups = num_features // group_size
    initial_params = np.zeros(num_groups * group_size)
    if objective_func.__name__ == "lasso_objective":
        alpha = alphaL
    if objective_func.__name__ == "group_lasso_objective":
        alpha = alphaG    
    if objective_func.__name__ == "freq_lasso_objective":
        alpha = alphaF  
    objective_function = lambda params: objective_func(params, X, Y, alpha, group_size)

    result = minimize(objective_function, initial_params, method='L-BFGS-B')

    optimal_params = result.x
    #coefficients = optimal_params.reshape((num_groups, group_size))

    return optimal_params


def simplify_parameters(parameter_df, tickList, feature_names) :
    abs_para = parameter_df.abs()
    adj_df = pd.DataFrame()
    
    for tick in tickList:
        lag_features = [c for c in feature_names if tick in c]
        adj_df[tick] = abs_para[lag_features].sum(axis=1)
    adj_df.index = tickList 
    return adj_df
