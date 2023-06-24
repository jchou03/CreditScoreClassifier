import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

# machine learning basic projects
# https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265
# https://thecleverprogrammer.com/2022/12/05/credit-score-classification-with-machine-learning/

data = pd.read_csv("Credit Score Data/train.csv")
# print(data.head())

# print(data.info())

print(data["Credit_Score"].value_counts())

# exploring different aspects of the occupation data
fig = px.box(data, 
             x="Occupation", 
             color="Credit_Score", 
             title="Credit Score Based on Occupation",
             color_discrete_map={
                 'Poor': 'red',
                 'Standard': 'yellow',
                 'Good': 'green'
             })

fig.show()