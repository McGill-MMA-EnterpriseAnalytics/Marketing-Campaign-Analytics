# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:19:33 2024

@author: Tony
"""
#import raw
import pandas as pd

ifood_df = pd.read_csv('C:/Users/Tony/Desktop/Code repos/ifood_df.csv')

ifood_df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

#split dataset
from sklearn.model_selection import train_test_split

# Splitting the dataframe into training and testing sets
train_df, test_df = train_test_split(ifood_df, test_size=0.2, random_state=42)

#first feature engineering
def feature_engineering_1(df):
    # Dependents
    df['Dependents'] = df['Kidhome'] + df['Teenhome']
    
    # Total Amount Spent
    mnt_cols = [col for col in df.columns if 'Mnt' in col]
    df['TotalMnt'] = df[mnt_cols].sum(axis=1)
    
    # Total Purchases
    purchases_cols = [col for col in df.columns if 'Purchases' in col]
    df['TotalPurchases'] = df[purchases_cols].sum(axis=1)
    
    # Total Campaigns Accepted
    campaigns_cols = [col for col in df.columns if 'Cmp' in col]
    df['response'] = df[campaigns_cols].any(axis=1).astype(int)
    return df

#define rfm group
def assign_rfm_groups(df):
    # Calculate the percentiles for 'Recency', 'Frequency', and 'Monetary'
    recency_50th = df['Recency'].quantile(0.5)
    frequency_50th = df['TotalPurchases'].quantile(0.5)
    monetary_50th = df['TotalMnt'].quantile(0.5)
    
    # Assign binary scores for each R,F,M category
    df['R'] = (df['Recency'] <= recency_50th).astype(int)
    df['F'] = (df['TotalPurchases'] > frequency_50th).astype(int)
    df['M'] = (df['TotalMnt'] > monetary_50th).astype(int)
    
    # Combine the R,F,M scores to a single group identifier
    df['RFM_Score'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)
    
    # Define group names based on RFM_Score
    group_names = {
        '111': 'Loyalist',
        '011': 'Potential Loyalist',
        '101': 'New Customers',
        '001': 'Promising',
        '110': 'At Risk',
        '010': 'Need Attention',
        '100': 'About To Sleep',
        '000': 'Hibernating'
    }
    
    # Map the group names to the dataframe
    df['RFM_Group'] = df['RFM_Score'].map(group_names)
    
    # One-hot encoding the 'RFM_Group'
    df = pd.get_dummies(df, columns=['RFM_Group'])
    
    return df

#add demographic
def assign_age_demographic(df):
    # Initialize a new column with empty strings
    df['Age_Demographic'] = ""
    
    # Iterate over each row to assign the age demographic based on age
    for index, row in df.iterrows():
        if row["Age"] > 54:
            df.at[index, "Age_Demographic"] = "Baby Boomer"
        elif row["Age"] > 38:
            df.at[index, "Age_Demographic"] = "Gen X"
        elif row["Age"] > 18:
            df.at[index, "Age_Demographic"] = "Gen Y"
        else:
            df.at[index, "Age_Demographic"] = "Gen Z"
    
    df = pd.get_dummies(df, columns=['Age_Demographic'])
    
    return df



#feature selection
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
target_column = 'Response'
def feature_selection_lasso(df, target_column):
            # Separate the features and the target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Normalize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform feature selection using LassoCV
            lasso = LassoCV(cv=5).fit(X_scaled, y)
            
            # Get the coefficients
            coef = lasso.coef_
            
            # Select features that have non-zero coefficients
            selected_features = X.columns[coef != 0]
            
            # Create a new dataframe with selected features and normalize them
            df_selected = df[selected_features]
            df_selected = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
            
            # Add the target column back
            df_selected[target_column] = y
            
            return df_selected

# PCA if necessary
from sklearn.decomposition import PCA
def dimension_reduction(df, n_components):
                # Separate the features from the target if it exists
                if target_column in df.columns:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                else:
                    X = df
                    y = None
                
                # Perform PCA for dimension reduction
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                
                # Create a new dataframe with the reduced dimensions
                columns = [f'PC{i+1}' for i in range(n_components)]
                df_reduced = pd.DataFrame(X_pca, columns=columns)
                
                # Add the target column back if it exists
                if y is not None:
                    df_reduced[target_column] = y
                
                return df_reduced

#Final dataset:
train_df = feature_engineering_1(train_df)
train_df = assign_rfm_groups(train_df)
train_df = assign_age_demographic(train_df)

test_df = feature_engineering_1(test_df)
test_df = assign_rfm_groups(test_df)
test_df = assign_age_demographic(test_df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = test_df
campaigns_cols = [col for col in df.columns if 'Cmp' in col]
exclude_cols = campaigns_cols + ['response']
cols_to_transform = [col for col in test_df.columns if col not in exclude_cols]
transformed_data = scaler.fit_transform(test_df[cols_to_transform])
transformed_df = pd.DataFrame(transformed_data, columns=cols_to_transform, index=test_df.index)
test_df[cols_to_transform] = transformed_df


df = train_df
campaigns_cols = [col for col in df.columns if 'Cmp' in col]
exclude_cols = campaigns_cols + ['response']
cols_to_transform = [col for col in train_df.columns if col not in exclude_cols]
transformed_data = scaler.fit_transform(train_df[cols_to_transform])
transformed_df = pd.DataFrame(transformed_data, columns=cols_to_transform, index=train_df.index)
train_df[cols_to_transform] = transformed_df


#common_columns = set(train_df.columns).intersection(set(test_df.columns))
#train_df = train_df[list(common_columns)]
#test_df = test_df[list(common_columns)]

# Store data files
folder_path = 'C:/Users/Tony/Desktop/Code repos/'

train_df.to_csv(folder_path + 'train_df.csv', index=False)
test_df.to_csv(folder_path + 'test_df.csv', index=False)



