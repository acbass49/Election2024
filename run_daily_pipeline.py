# Title: 2024 Presidential Election Modeling and Processing
# Author: Alex Bass
# Date: 30 December 2023

from election_utils import get_data, \
    make_state_predictions, estimate_bayes_heirarchal, \
    run_simulation, calc_simulation_interval
import pandas as pd
import numpy as np
from datetime import datetime

# Fetch Data
y_vec, x_matrix, state_dict = get_data()

# Estimate Model
model, trace = estimate_bayes_heirarchal(y_vec, x_matrix, state_dict)

# Predict State Level Probabilities
preds = make_state_predictions(model, state_dict, x_matrix, trace)

# Run Presidential Simulations
win_perc, sim_data = run_simulation(preds, 50000)

# A Few Post-Processing Steps
sim_data = sim_data.assign(winner = lambda x:np.where(x.winner == 0, "Biden", "Trump"))
to_join = pd.read_csv('https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv')
prob_data = pd.DataFrame({
    'State':list(preds.keys()),
    'Trump Win Prob.':list(preds.values())
}) \
    .merge(to_join, on='State') \
    .assign(State = lambda x:x.Abbreviation) \
    .drop(columns = ['Abbreviation'])

# Calculate Simulation Confidence Interval
LB, UB = calc_simulation_interval(sim_data)

# Add new row to tracker
tracking_data = pd.read_csv("./data/tracking_data.csv")
current_date = datetime.now().date()

new_row = pd.DataFrame({
    'Candidate':['Trump', 'Biden'],
    'Win Percentage':[win_perc, 1-win_perc], #perhaps add a confidence interval to this?
    'Date' : current_date,
    'LB' : [LB,(1-win_perc)-(win_perc-LB)],
    'UB' : [UB,(1-win_perc)+(UB-win_perc)]
})

tracking_data = pd.concat([tracking_data, new_row])

# Saving Data
prob_data.to_csv("./data/state_probabilities.csv", index = False)
sim_data.to_csv("./data/simulation_data.csv", index = False)
tracking_data.to_csv("./data/tracking_data.csv", index = False)
