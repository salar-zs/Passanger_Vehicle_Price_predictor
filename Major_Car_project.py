# %%
### Project for Data Analysis, ML on data about Car sales history
### Version 1 ; Multiple linear regression and Random forest method is used to predict car prise values 

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# %%
#Comprehensive pricelist of 4000 different passenger vehicles 
# ── 1. Load raw data ──────────────────────
df = pd.read_csv('used_cars.csv')

# %%
df.head()

# %%
# Horsepower is extracted from the engine description 
df['horsepower'] = df['engine'].str.extract(r'([\d.]+)\s*HP', expand=False).astype(float)


# %%
# Cars with undefined Horspower information are assigned with a median horsepower  
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# %%
# Engine displacement (litres) is extracted from the engine description 
df['displacement_L'] = df['engine'].str.extract(r'([\d.]+)\s*L', expand=False).astype(float)


# %%
# Cars with undefined engine displacement info. are assigned with a median engine displacement 
df['displacement_L'].fillna(df['displacement_L'].median(), inplace=True)

# %%
# The non-numerical feature (fuel type) is converted to multiple binary  
#To avoid dummy variable trap (perfect multicollinearity), we drop one of the categories of the featture 
#Its important to use binary encoding to avoid mathematical false order between categories (i.e. Diesel > Gasoline)
df = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)


# %%
# The non-numerical feature (brand) is converted to multiple binary  
df = pd.get_dummies(df, columns=['brand'], drop_first=True)


# %%
# Clean mileage column
df['milage'] = (df['milage']
                 .str.replace(',', '', regex=False)
                 .str.replace('mi.', '', regex=False)
                 .str.strip()
                 .astype(float))

# %%
## we chose 0 for no accident and 1 for accident 
df['accident'] = df['accident'].map({
    'None reported': 0,
    'At least 1 accident or damage reported': 1
})

# %%
# Clean price coloumn 

# Price: "$10,300" → 10300.0
df['price'] = (df['price']
               .str.replace(r'[\$,]', '', regex=True)
               .astype(float))

# %%
#here all the binary features related to fuel type are considered
fuel_cols = [col for col in df.columns if col.startswith('fuel_')]

# %%
#here all the binary features related to brand type are considered

brand_cols = [col for col in df.columns if col.startswith('brand_')]


# %%
#Total feature array is constructed 
features = ['model_year', 'milage','accident','horsepower','displacement_L'] + brand_cols + fuel_cols

#NaN are removed from the dataset as they cause error 
clean_df = df[features + ['price']].dropna()

X = clean_df[features].values
y = clean_df['price'].values

# %%
clean_df.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
#First we are going to try multiple linear regression approach for building the predictive model 
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)

# %%
from sklearn.metrics import r2_score, mean_squared_error
r2_mlr   = r2_score(y_test, y_pred)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"mlr  R²:   {r2_mlr:.4f}")
print(f"mlr RMSE: {rmse_mlr:,.0f}")


# %%
# Second, we are trying to build the model using Random Forest approach 
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=500, random_state=0)
rf_regressor.fit(X_train, y_train)


# %%
y_pred_rf = rf_regressor.predict(X_test)

# %% [markdown]
# 

# %%

r2_rf   = r2_score(y_test, y_pred_rf)

print(f"rf  R²:   {r2_rf:.4f}")


# %%
#Multiple linear regression leads to Rsquare of 0.6
# Random Forest has Rsquared of 0.85 (Winner)

# %%
# Predict price for a specific car not in the dataset
## Now we want to predict the price of a Random car such as Audi etron only via its (brand, milage, model year, Hp, accident)
# Brand: Audi, Mileage: 40000, No accident, 310 HP, Electric, Model Year: 2020, displacement_L: NaN

input_dict = {col: 0 for col in features}


input_dict['model_year']     = 2020
input_dict['milage']         = 40000
input_dict['accident']       = 0
input_dict['horsepower']     = 310
input_dict['displacement_L'] = df['displacement_L'].median()

if 'brand_Audi' in input_dict:
    input_dict['brand_Audi'] = 1

if 'fuel_type_Electric' in input_dict:
    input_dict['fuel_type_Electric'] = 1

input_df = pd.DataFrame([input_dict])

predicted_price = rf_regressor.predict(input_df.values)
print(f"Predicted Price for Audi e-tron: ${predicted_price[0]:,.0f}")
