# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# | output: false
from election_utils import get_data, \
    make_state_predictions, estimate_bayes_heirarchal, \
    run_simulation

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

y_vec, x_matrix, state_dict = get_data()
model, trace = estimate_bayes_heirarchal(y_vec, x_matrix, state_dict)
preds = make_state_predictions(model, state_dict, x_matrix, trace)
win_perc, sim_data = run_simulation(preds, 50000)
sim_data = sim_data.assign(winner = lambda x:np.where(x.winner == 0, "Biden", "Trump"))

#
#
#
plt.hist(sim_data.query("winner == 'Trump'")[['points']], bins=45, color='red',edgecolor='black')
plt.hist(sim_data.query("winner == 'Biden'")[['points']], bins=50, color='skyblue',edgecolor='black')
plt.xlabel('EC Votes Trump Wins')
plt.ylabel('Simulation Wins')
plt.title(f'Today Trump Won {round(win_perc*100,2)}% of the Election Simulations')
plt.legend(["Trump", "Biden"], loc ="upper right", title='Winner')
plt.show()
#
#
#
#
#
to_join = pd.read_csv('https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv')

prob_data = pd.DataFrame({
    'State':list(preds.keys()),
    'Trump Win Prob.':list(preds.values())
}).merge(to_join, on='State') \
    .assign(State = lambda x:x.Abbreviation)

fig = go.Figure(data=go.Choropleth(
    locations=prob_data['State'], # Spatial coordinates
    z = prob_data['Trump Win Prob.'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'rdylbu_r',
    colorbar_title = "Trump Win Probability",
    marker_line_color='white'
))

fig.update_layout(
    title_text = 'Presidential Election Projections By State',
    geo_scope='usa', # limite map scope to USA
)
config = {'displayModeBar': False}
fig.show(config=config)

#
#
#
#
#
#
#
