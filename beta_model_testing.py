from election_utils import get_data, \
    make_state_predictions, estimate_bayes_heirarchal, \
    run_simulation, calc_simulation_interval, \
    estimate_bayes_beta, estimate_bayes_beta_alt, \
    estimate_bayes_beta_cstm_priors
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# Fetch Data
y_vec, x_matrix, state_dict = get_data()

# Estimate Model
model, trace = estimate_bayes_beta_alt(y_vec, x_matrix, state_dict)

preds = make_state_predictions(model, state_dict, x_matrix, trace)

win_perc, sim_data = run_simulation(preds, 50000)



# Estimate Model
model, trace = estimate_bayes_beta(y_vec, x_matrix, state_dict)

preds = make_state_predictions(model, state_dict, x_matrix, trace)

win_perc, sim_data = run_simulation(preds, 50000)




# Estimate Model
priors = pd.read_csv('./data/priors.csv')
priors.sd = priors.sd * 20
model, trace = estimate_bayes_beta_cstm_priors(y_vec, x_matrix, state_dict, priors)

preds = make_state_predictions(model, state_dict, x_matrix, trace)

win_perc, sim_data = run_simulation(preds, 50000)






az.plot_trace(trace)

priors = az.summary(trace, kind="stats", var_names=['~Intercept']) \
    .reset_index() \
    .rename(columns = {'index':'var'}) \
    [['var', 'mean', 'sd']]

states_dict = state_dict

#simple one
with model:
    pm.set_data({
        "X1": [0 for x in range(len(states_dict))],
        "X2": [0 for x in range(len(states_dict))],
        "X3": [0 for x in range(len(states_dict))],
        "X4": [x_matrix['month'].max() for x in range(len(states_dict))],
        "X5": [1 for x in range(len(states_dict))],
        "X6": [2000 for x in range(len(states_dict))],
        "X7": [1 for x in range(len(states_dict))],
        "X8": [1 for x in range(len(states_dict))],
        "X9": [0 for x in range(len(states_dict))],
        "X10": [1 for x in range(len(states_dict))],
        'Y_obs': [-1000 for x in range(len(states_dict))],
        'states': list(states_dict.values())
    })
    pp = pm.sample_posterior_predictive(
        trace, predictions=True, random_seed=1
    )
pred_matrix = pp['predictions']['y'].mean(('chain'))

results = {}

for i in range(pred_matrix.shape[1]):
    val = np.divide(np.sum(np.greater(pred_matrix[:,i],0.5)),len(pred_matrix[:,1]))
    val = round(float(val)*100,2)
    # if val > 99:
    #     val = 99
    # if val < 1:
    #     val = 1
    results[list(states_dict.keys())[i]] = val

print(results)

preds = make_state_predictions(model, state_dict, x_matrix, trace)
win_perc, sim_data = run_simulation(preds, 50000)
