# Title: Aquire and Clean data
# Author: Alex Bass
# Date: 16 December 2023

import pandas as pd
import numpy as np
import pymc as pm

def get_data():

    link_538 = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'

    data = pd.read_csv(link_538)

    # Collapsing `methodology` variable
    panels_to_keep = [
        'Online Panel', 
        'Live Phone',
        'Probability Panel',
        'App Panel'
    ]

    data.loc[
        (~data.methodology.isin(panels_to_keep)), 'methodology'
    ] = 'Other'

    # Collapsing `population` variable
    data.population = data.population.replace({'v':'a'})

    # Adding a partisan variable
    data['rep_poll'] = np.where(data['partisan'] == 'REP', 1, 0)

    #filter out where state is NA
    data = data.loc[~data.state.isna(),:].reset_index(drop=True)

    def identify_multi_candidate(data):
        #identify multi response questions
        if data.shape[0] > 2:
            data['MultiCandidate'] = 1
        elif data.shape[0] == 2:
            data['MultiCandidate'] = 0
        else:
            print(data)
            raise Exception("Something weird going on")
        #identify non biden v trump questions
        trump_biden_ids = [19368,16651]
        keep = set(data['candidate_id'].to_list()) \
            .intersection(set(trump_biden_ids))
        keep = int(len(keep)>1)
        data['trump_v_biden'] = keep
        return data

    def rescale_to_100(data):
        data.pct = np.divide(data.pct,data.pct.sum())
        return data

    cols_to_keep = [
        'methodology',
        'rep_poll',
        'population',
        'state',
        'sample_size',
        'MultiCandidate',
        'end_date',
        'fte_grade',
        'pct'
    ]

    data = (
        data
            .groupby("question_id")
            .apply(identify_multi_candidate)
            .reset_index(drop=True)
            .query('trump_v_biden == 1')
            .drop(columns = ['trump_v_biden'])
            .reset_index(drop=True)
            .query('candidate_id == 19368 or candidate_id == 16651')
            .groupby("question_id")
            .apply(rescale_to_100)
            .reset_index(drop=True)
            .query("candidate_id == 16651")
            [cols_to_keep]
    )

    categories = [
        'A+',
        'A',
        'A-',
        'A/B',
        'B+',
        'B',
        'B-',
        'B/C',
        'C+',
        'C',
        'C-',
        'C/D'
    ]

    # There are missing grades, so need to impute
    data.fte_grade = data.fte_grade.astype("category") \
        .cat.set_categories(categories, ordered=True)

    def find_median_category(data):
        value_counts = data.value_counts()
        sorted_counts = value_counts.sort_index()
        sorted_props = sorted_counts/sum(sorted_counts)*100
        index_of_median = (len(sorted_props) - \
            sum(sorted_props.cumsum()>50)) -1
        return sorted_props.index[index_of_median]

    median_cat = find_median_category(data.fte_grade)

    data.fte_grade = data.fte_grade.fillna(median_cat)

    data = (
        data
            .assign(
                date_maker = pd.to_datetime(data.end_date),
                month2 = lambda x:x.date_maker.dt.month,
                year = lambda x:x.date_maker.dt.year - 2021,
                date_maker2 = lambda x:x.month2 + x.year*12,
                month = lambda x:x.date_maker2 - x.date_maker2.min()
            )
            .drop(
                columns = [
                    'date_maker',
                    'date_maker2',
                    'month2',
                    'year',
                    'end_date'
                ]
            )
    )
    
    states = data.state.value_counts().index.to_list()
    states_dict = {x:i for i,x in enumerate(states)}
    data.state = data.state.replace(states_dict)
    
    method = pd.get_dummies(data.methodology) \
        .drop(columns = ['Probability Panel']) \
        .astype('int')
    
    population = pd.get_dummies(data.population) \
        .drop(columns = ['a']) \
        .astype('int')
    
    data['grade'] = (data.fte_grade > 'B/C').astype('int')
    
    data.drop(columns = ['methodology', 'population', 'fte_grade'], inplace = True)
    
    data = pd.concat([data,method,population], axis=1)
    
    return data['pct'].values, \
        data, \
        states_dict

def make_state_predictions(model, states_dict, x_matrix, trace):
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
            if val > 99:
                val = 99
            if val < 1:
                val = 1
            results[list(states_dict.keys())[i]] = val  
        
        # Add in states not in polling dataset
        file_url = 'https://projects.fivethirtyeight.com/2020-general-data/presidential_poll_averages_2020.csv'

        old_data = pd.read_csv(file_url).query("candidate_name == 'Donald Trump' and modeldate == '11/3/2020'")[['state', 'pct_estimate']] \
            .assign(pct_estimate = lambda x:np.where(x.pct_estimate>50,99,1))
        
        old_data = pd.concat([old_data,pd.DataFrame({'state':"NE-3", 'pct_estimate':99}, index=[0])])

        states_to_drop = list(results.keys())

        old_data = old_data.loc[~old_data.state.isin(states_to_drop),:]
        
        for row in old_data.iterrows():
            results[row[1][0]] = row[1][1]
        
        return {k:v for k,v in results.items() if k != 'National'}

def estimate_bayes_heirarchal(y_vec, x_matrix, state_dict):
    n_state = len(state_dict)
    state_r = x_matrix.state.values

    with pm.Model() as model:

        # b0 - intercept 
        mu_b0 = pm.Normal('mu_b0', 0, sigma=1)
        sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
        
        # Random intercepts as offsets
        a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=n_state)
        b0 = pm.Deterministic("Intercept", mu_b0 + a_offset * sigma_b0)

        # Setting data
        X1 = pm.MutableData("X1", x_matrix['Live Phone'].values)
        X2 = pm.MutableData("X2", x_matrix['Online Panel'].values)
        X3 = pm.MutableData("X3", x_matrix['Other'].values)
        X4 = pm.MutableData("X4", x_matrix['month'].values)
        X5 = pm.MutableData("X5", x_matrix['rep_poll'].values)
        X6 = pm.MutableData("X6", x_matrix['sample_size'].values)
        X7 = pm.MutableData("X7", x_matrix['MultiCandidate'].values)
        X8 = pm.MutableData("X8", x_matrix['lv'].values)
        X9 = pm.MutableData("X9", x_matrix['rv'].values)
        X10 = pm.MutableData("X10", x_matrix['grade'].values)
        Y_obs = pm.MutableData("Y_obs", y_vec)
        states = pm.MutableData("states", state_r)

        b1 = pm.Normal("Live Phone", mu=0, sigma=0.1)
        b2 = pm.Normal("Online Panel", mu=0, sigma=0.1)
        b3 = pm.Normal("Other", mu=0, sigma=0.1)
        b4 = pm.Normal("month", mu=0, sigma=0.1)
        b5 = pm.Normal("rep_poll", mu=0, sigma=1)
        b6 = pm.Normal("sample_size", mu=0, sigma=1)
        b7 = pm.Normal("MultiCandidate", mu=0, sigma=1)
        b8 = pm.Normal("lv", mu=0, sigma=1)
        b9 = pm.Normal("rv", mu=0, sigma=1)
        b10 = pm.Normal("grade", mu=0, sigma=1)

        formula =  (
            b0[states] + 
            b1*X1 + 
            b2*X2 + 
            b3*X3 + 
            b4*X4 + 
            b5*X5 +
            b6*X6 +
            b7*X7 +
            b8*X8 +
            b9*X9 +
            b10*X10
        )
        
        s = pm.HalfNormal('error',sigma =1)

        obs = pm.Normal('y', mu = formula, sigma=s, observed=Y_obs)

        trace = pm.sample(1000, tune=1000, cores=1)

        return model, trace

def run_simulation(preds, simulation_num):
    '''
    given a dict with each state's probability of one candidate winning
    will return number of simulations won by that candidate
    '''
    import numpy as np
    import pandas as pd
    
    ec_data = {'Arizona': 11,
    'Georgia': 16,
    'Pennsylvania': 19,
    'Michigan': 15,
    'Nevada': 6,
    'Wisconsin': 10,
    'North Carolina': 3,
    'Ohio': 17,
    'Florida': 30,
    'New Hampshire': 4,
    'New York': 28,
    'California': 54,
    'Iowa': 6,
    'Tennessee': 11,
    'Virginia': 13,
    'Missouri': 10,
    'Texas': 40,
    'Colorado': 10,
    'Montana': 4,
    'Washington': 12,
    'Illinois': 19,
    'Connecticut': 7,
    'Oklahoma': 7,
    'New Mexico': 5,
    'Kansas': 6,
    'Massachusetts': 11,
    'Minnesota': 10,
    'Kentucky': 8,
    'Alaska': 3,
    'Oregon': 8,
    'Nebraska': 2,
    'South Carolina': 9,
    'Maryland': 10,
    'Rhode Island': 4,
    'Arkansas': 6,
    'South Dakota': 3,
    'Louisiana': 8,
    'Mississippi': 6,
    'Maine': 2,
    'Utah': 6,
    'Idaho': 4,
    'Alabama': 9,
    'West Virginia': 4,
    'Indiana': 11,
    'North Dakota': 3,
    'Wyoming': 3,
    'Vermont': 3,
    'New Jersey': 14,
    'National': 1,
    'NE-2': 1,
    'NE-1': 1,
    'NE-1': 1,
    'NE-3':1,
    'ME-2': 1,
    'ME-1': 1,
    'Hawaii': 4,
    'District of Columbia': 3,
    'Delaware': 3}
    
    def simulate_state(prob, points):
        prob = prob/100
        trump_win = np.random.choice([0,1], p=[1-prob, prob])
        return trump_win*points
    
    winner = []
    points = []
    sim_num = []
    
    for _ in range(simulation_num):
        votes = [simulate_state(prob,ec_data[state]) for state,prob in preds.items()]
        tot_votes = sum(votes)
        winner.append(np.where(tot_votes>=270, 1,0))
        points.append(tot_votes)
    
    data = pd.DataFrame({
        'winner':winner,
        'points':points
    })
    
    trump_won = sum(data.winner)/data.shape[0]
    
    return trump_won, data

# #reset tracking csv
win_perc = 0.18

from datetime import datetime
current_date = datetime.now().date()

pd.DataFrame({
    'Candidate':['Trump', 'Biden'],
    'Win Percentage':[win_perc, 1-win_perc], #perhaps add a confidence interval to this?
    'Date' : current_date
}).to_csv("./data/tracking_data.csv", index = False)
