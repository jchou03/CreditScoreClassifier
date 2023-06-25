import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

# machine learning basic projects
# https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265
# https://thecleverprogrammer.com/2022/12/05/credit-score-classification-with-machine-learning/

data = pd.read_csv("Credit Score Data/train.csv")
color_discrete_map = {'Poor':'red',
                      'Standard':'yellow',
                      'Good':'green'}
# print(data.head())

# print(data.info())

print(data["Credit_Score"].value_counts())

# exploring different aspects of the occupation data
# fig = px.box(data, 
#              x="Occupation", 
#              color="Credit_Score", 
#              title="Credit Score Based on Occupation",
#              color_discrete_map={
#                  'Poor': 'red',
#                  'Standard': 'yellow',
#                  'Good': 'green'
#              })

# fig = px.box(data,
#              x="Credit_Score",
#              y="Annual_Income",
#              color="Credit_Score",
#              title="Credit Scores Based on Annual Income",
#              color_discrete_map={'Poor':'red',
#                                  'Standard':'yellow',
#                                  'Good':'green'})

# fig = px.box(data,
#              x="Credit_Score",
#              y="Monthly_Inhand_Salary",
#              color="Credit_Score",
#              title="Credit Scores Based on Monthly Inhand Salary",
#              color_discrete_map=color_discrete_map)

fig = px.box(data,
             x="Credit_Score",
             y="Num_Bank_Accounts",
             color="Credit_Score",
             title="Credit Score based on number of bank accounts",
             color_discrete_map=color_discrete_map)

fig.update_traces(quartilemethod="exclusive")
fig.show()