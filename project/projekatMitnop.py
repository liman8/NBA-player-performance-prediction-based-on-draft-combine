import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Учитавање резултата са драфт-комбајна из датотеке „draft_combine.csv” 
draft_combine_path = 'draft_combine.csv'
draft_combine_df = pd.read_csv(draft_combine_path)

# Учитавам статистику са драфт-комбајна из датотеке „rookie_stats.xlsx” 
rookie_stats_path = 'rookie_stats.xlsx'
rookie_stats_df = pd.read_excel(rookie_stats_path)

# Стардардизовање имена играча одстрањивањем вишка карактера
draft_combine_df['player_name'] = draft_combine_df['player_name'].str.strip()
rookie_stats_df['Name'] = rookie_stats_df['Name'].str.strip()

# Спајање података у један скуп, урађено је на основу имена играча и године када је играч драфтован
merged_df = pd.merge(
    draft_combine_df,
    rookie_stats_df,
    left_on=['player_name', 'yearDraft'],
    right_on=['Name', 'Year Drafted'],
    how='inner'
)

# Избацивање колона које описују статистику из датотеке „rookie_stats.xlsx”  које не предвиђамо
columns_to_drop = [
    'FGM', '3PA', 'FTA', 'FGA', 'DREB', 
    'OREB', '3P Made','GP',
    'FTM','REB', 'FTA', 'PTS','FG%', '3P%',
    'FT%', 'AST', 'STL', 'BLK', 'TOV', 
]
merged_df = merged_df.drop(columns=columns_to_drop)

# Избацивање колона које нису везане за резултате на драфт-комбајну
columns_to_drop = [
    'yearDraft', 'numberPickOverall', 'position',
    'drafted', 'yearCombine', 'player_id',
    'Name', 'Year Drafted', 'player_name'
]
merged_df = merged_df.drop(columns=columns_to_drop)

# Проверавање постојања недостајајућих вредности
missing_values = merged_df.isnull().sum()
#print(missing_values)

# Избацивање колона са превише недостајајућих вредности
columns_to_drop = [
    'timeModifiedLaneAgility',
   'lengthHandInches', 'widthHandInches'
]
merged_df = merged_df.drop(columns=columns_to_drop)

# Попуњавање недостајајућих вредности са просечном вредношћу сваке колоне за колоне са нумеричким вредностима
numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())

# Провера постојања недостајајућих вредности након измена
missing_values_final = merged_df.isnull().sum()
#print(missing_values_final)

# Функција која рачуна прилагођену ефикасност по минуту, узимајући у обзир фактор скалирања
def adjusted_epm(eff, min_played):
    if min_played == 0:
        return 0  # Избегавање дељења са нулом
    scaling_factor = 1 - np.exp(-min_played / 10)
    return (eff / min_played) * scaling_factor

# Креирање нове коплоне, применом функције „adjusted_epm” 
merged_df['Adjusted EPM'] = merged_df.apply(
    lambda row: adjusted_epm(row['EFF'], row['MIN']),
    axis=1
)

# Избацивање колоне за минуте
merged_df = merged_df.drop(columns=['MIN']) 

# 2. Провера корелације између нумеричких вредности
# Избацивање ненумеричких колона
numeric_df = merged_df.select_dtypes(include=[np.number])

# Рачунање матрице корелације и њено плотовање
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Проналазак високо корелираних колона и њихово избацивање,
# узета граница корелираности од 0.9
high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
high_corr = high_corr[high_corr >= 0.9]
high_corr = high_corr[high_corr < 1]
print("Highly Correlated Pairs:\n", high_corr)

merged_df = merged_df.drop(columns = ['reach_standing'])


# Избацивање колона које предвиђамо и прављење улазног и излазног скупа
X = merged_df.drop(columns=['EFF', 'Adjusted EPM'])  # Карактеристике
y_eff = merged_df['EFF']  # Циљна променљива за ефикасност
y_adj_epm = merged_df['Adjusted EPM']  # Циљна променљива за прилагођену ефикасност по минуту

# Подела скупова на скуп за тренирање и за тестирање
X_train, X_test, y_eff_train, y_eff_test = train_test_split(X, y_eff, test_size=0.2, random_state=42)
_, _, y_adj_epm_train, y_adj_epm_test = train_test_split(X, y_adj_epm, test_size=0.2, random_state=42)

# Препроцесирање: Стандаризација нумеричких карактеристика
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Фитовање препоцесора
preprocessor.fit(X_train)

# Дефинисање пајплајна за ефикасност са линеарном регресијом
model_eff = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Дефинисање пајплајна за прилагођену ефикасности по минуту са линеарном регресијом
model_adj_epm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Тренирање модела
model_eff.fit(X_train, y_eff_train)
model_adj_epm.fit(X_train, y_adj_epm_train)

# Предвиђање и рачунање просечне квадратне грешке и р2, ради евалуације модела
y_eff_train_pred = model_eff.predict(X_train)
y_adj_epm_train_pred = model_adj_epm.predict(X_train)

mse_eff_train = mean_squared_error(y_eff_train, y_eff_train_pred)
r2_eff_train = r2_score(y_eff_train, y_eff_train_pred)

mse_adj_epm_train = mean_squared_error(y_adj_epm_train, y_adj_epm_train_pred)
r2_adj_epm_train = r2_score(y_adj_epm_train, y_adj_epm_train_pred)

y_eff_test_pred = model_eff.predict(X_test)
y_adj_epm_test_pred = model_adj_epm.predict(X_test)

mse_eff_test = mean_squared_error(y_eff_test, y_eff_test_pred)
r2_eff_test = r2_score(y_eff_test, y_eff_test_pred)

mse_adj_epm_test = mean_squared_error(y_adj_epm_test, y_adj_epm_test_pred)
r2_adj_epm_test = r2_score(y_adj_epm_test, y_adj_epm_test_pred)

# Испис резултата линеарне регресије
print("")
print("")
print('EFF Model: (LinearRegression)')
print(f'Training - MSE: {mse_eff_train}, R2: {r2_eff_train}')
print(f'Test - MSE: {mse_eff_test}, R2: {r2_eff_test}')

print('\nAdjusted EPM Model: (LinearRegression)')
print(f'Training - MSE: {mse_adj_epm_train}, R2: {r2_adj_epm_train}')
print(f'Test - MSE: {mse_adj_epm_test}, R2: {r2_adj_epm_test}')


# Дефинисање пајплајна за ефикасност са линеарном методом случајних шума
model_eff_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Дефинисање пајплајна за прилагођену ефикасности по минуту са методом случајних шума
model_adj_epm_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Тренирање модела
model_eff_rf.fit(X_train, y_eff_train)
model_adj_epm_rf.fit(X_train, y_adj_epm_train)

# Предвиђање и рачунање просечне квадратне грешке и р2, ради евалуације модела
y_eff_rf_pred = model_eff_rf.predict(X_test)
y_adj_epm_rf_pred = model_adj_epm_rf.predict(X_test)

mse_eff_rf = mean_squared_error(y_eff_test, y_eff_rf_pred)
r2_eff_rf = r2_score(y_eff_test, y_eff_rf_pred)

mse_adj_epm_rf = mean_squared_error(y_adj_epm_test, y_adj_epm_rf_pred)
r2_adj_epm_rf = r2_score(y_adj_epm_test, y_adj_epm_rf_pred)

# Испис резултата случајних шума
print("")
print("")
print('EFF Model: (RandomForestRegressor)')
print(f'Training - MSE: {mse_eff_rf}, R2: {r2_eff_rf}')
print(f'Test - MSE: {mse_eff_test}, R2: {r2_eff_test}')

print('\nAdjusted EPM Model: (RandomForestRegressor)')
print(f'Training - MSE: {mse_adj_epm_rf}, R2: {r2_adj_epm_rf}')
print(f'Test - MSE: {mse_adj_epm_test}, R2: {r2_adj_epm_test}')


# Дефинисање пајплајна за ефикасност са XGBoost
model_eff_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Дефинисање пајплајна за прилагођену ефикасности по минуту са XGBoost
model_adj_epm_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Тренирање модела
model_eff_xgb.fit(X_train, y_eff_train)
model_adj_epm_xgb.fit(X_train, y_adj_epm_train)

# Предвиђање и рачунање просечне квадратне грешке и р2, ради евалуације модела
y_eff_xgb_pred = model_eff_xgb.predict(X_test)
y_adj_epm_xgb_pred = model_adj_epm_xgb.predict(X_test)

mse_eff_xgb = mean_squared_error(y_eff_test, y_eff_xgb_pred)
r2_eff_xgb = r2_score(y_eff_test, y_eff_xgb_pred)

mse_adj_epm_xgb = mean_squared_error(y_adj_epm_test, y_adj_epm_xgb_pred)
r2_adj_epm_xgb = r2_score(y_adj_epm_test, y_adj_epm_xgb_pred)

# Испис резултата XGBoost модела
print("")
print("")
print('EFF Model: (XGBoost)')
print(f'Training - MSE: {mse_eff_xgb}, R2: {r2_eff_xgb}')
print(f'Test - MSE: {mse_eff_test}, R2: {r2_eff_test}')

print('\nAdjusted EPM Model: (XGBoost)')
print(f'Training - MSE: {mse_adj_epm_xgb}, R2: {r2_adj_epm_xgb}')
print(f'Test - MSE: {mse_adj_epm_test}, R2: {r2_adj_epm_test}')


# Дефинисање дијаграма распршења стварних и предвиђаних вредности
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Исцртавање дијаграма за ефикасност
plot_actual_vs_predicted(y_eff_test, y_eff_test_pred, 'Actual vs Predicted - EFF (Linear Regression)')
plot_actual_vs_predicted(y_eff_test, y_eff_rf_pred, 'Actual vs Predicted - EFF (Random Forest)')
plot_actual_vs_predicted(y_eff_test, y_eff_xgb_pred, 'Actual vs Predicted - EFF (XGBoost)')

# Исцртавање дијаграма за прилагођену ефикасност по минути
plot_actual_vs_predicted(y_adj_epm_test, y_adj_epm_test_pred, 'Actual vs Predicted - Adjusted EPM (Linear Regression)')
plot_actual_vs_predicted(y_adj_epm_test, y_adj_epm_rf_pred, 'Actual vs Predicted - Adjusted EPM (Random Forest)')
plot_actual_vs_predicted(y_adj_epm_test, y_adj_epm_xgb_pred, 'Actual vs Predicted - Adjusted EPM (XGBoost)')

# Рачунање најважнијих карактеристика, припрема и исцртавање
importances_rf = model_eff_rf.named_steps['regressor'].feature_importances_
importances_xgb = model_eff_xgb.named_steps['regressor'].feature_importances_

numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
all_feature_names = numeric_features

forest_importances = pd.Series(importances_rf, index=all_feature_names)
xgb_importances = pd.Series(importances_xgb, index=all_feature_names)

plt.figure(figsize=(12, 8))
forest_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importances - Random Forest (EFF)')
plt.show()

plt.figure(figsize=(12, 8))
xgb_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importances - XGBoost (EFF)')
plt.show()
