from election_utils import get_data, \
    make_state_predictions, estimate_bayes_heirarchal, \
    run_simulation, calc_simulation_interval, estimate_bayes_heirarchal_cstm_priors
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import arviz as az

# Fetch Data
y_vec, x_matrix, state_dict = get_data()

# Estimate Model
model, trace = estimate_bayes_heirarchal(y_vec, x_matrix, state_dict)

priors = az.summary(trace, kind="stats", var_names=['~Intercept']) \
    .reset_index() \
    .rename(columns = {'index':'var'}) \
    [['var', 'mean', 'sd']]

model2, trace2 = estimate_bayes_heirarchal_cstm_priors(y_vec, x_matrix, state_dict, priors)


# Predict State Level Probabilities
preds = make_state_predictions(model, state_dict, x_matrix, trace)
preds2 = make_state_predictions(model2, state_dict, x_matrix, trace2)




def get_prior(var, metric, priors):
    return priors.query("var == @var")[metric].iloc[0]