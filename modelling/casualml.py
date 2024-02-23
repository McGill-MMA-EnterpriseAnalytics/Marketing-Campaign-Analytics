# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:58:01 2024

@author: Tony
"""

import pandas as pd
import numpy as np
from causalml.inference.meta import LRSRegressor, XGBTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.metrics import plot_gain
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Tony/Desktop/Code repos/train_df.csv')
# Assuming 'df' is your DataFrame and it's already loaded
# Let's split df into features (X) and target (y), assuming 'treatment' is the treatment column
campaigns_cols = [col for col in df.columns if 'Cmp' in col]
X = df.drop(campaigns_cols, axis=1)
X = X.drop(['response'], axis=1)

y = df['AcceptedCmp1']

# Treatment column
treatment = 'MntTotal'
# Assuming you want to classify as treated those above a certain threshold
threshold = df[treatment].median()  # For example, using median as a simple threshold
X[treatment] = (X[treatment] > threshold).astype(int)

# Convert treatment to binary
X[treatment] = X[treatment].apply(lambda x: 1 if x == 1 else 0)

# Control features (assuming all other columns are control features)
X_control = X.drop([treatment], axis=1)

# For simplicity, let's use X_control as our features for this example
# In a more complex scenario, you might select specific features based on domain knowledge or feature selection techniques

# Initialize models
lr_model = LRSRegressor()
xgb_model = XGBTRegressor(random_state=42)

# Fit the models
lr_estimates = lr_model.estimate_ate(X=X_control, treatment=X[treatment], y=y)
xgb_estimates = xgb_model.estimate_ate(X=X_control, treatment=X[treatment], y=y)

# Print ATE (Average Treatment Effect) estimates
print("Linear Regression ATE:", lr_estimates[0])
print("XGBoost ATE:", xgb_estimates[0])


#---------------------------------------------------------------

import pandas as pd
import numpy as np
from causalml.inference.meta import LRSRegressor, XGBTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.metrics import plot_gain
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Tony/Desktop/Code repos/train_df.csv')
# Assuming 'df' is your DataFrame and it's already loaded
# Let's split df into features (X) and target (y), assuming 'treatment' is the treatment column
campaigns_cols = [col for col in df.columns if 'Cmp' in col]
X = df.drop(campaigns_cols, axis=1)
X = X.drop(['response'], axis=1)

y = df['AcceptedCmp2']

# Treatment column
treatment = 'MntWines'
# Assuming you want to classify as treated those above a certain threshold
threshold = df[treatment].median()  # For example, using median as a simple threshold
X[treatment] = (X[treatment] > threshold).astype(int)

# Convert treatment to binary
X[treatment] = X[treatment].apply(lambda x: 1 if x == 1 else 0)

# Control features (assuming all other columns are control features)
X_control = X.drop([treatment], axis=1)

# For simplicity, let's use X_control as our features for this example
# In a more complex scenario, you might select specific features based on domain knowledge or feature selection techniques

# Initialize models
lr_model = LRSRegressor()
xgb_model = XGBTRegressor(random_state=42)

# Fit the models
lr_estimates = lr_model.estimate_ate(X=X_control, treatment=X[treatment], y=y)
xgb_estimates = xgb_model.estimate_ate(X=X_control, treatment=X[treatment], y=y)

# Print ATE (Average Treatment Effect) estimates
print("Linear Regression ATE:", lr_estimates[0])
print("XGBoost ATE:", xgb_estimates[0])
