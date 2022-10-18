# Football-Match-Outcome-Prediction

> Football Match Prediction Classifier using data from over 30 years from the top European leagues to predict future match results. The project involves: EDA, data cleaning, feature engineering, feature selection, training and optimising multiple classification models for the best accuracy.

## EDA and Data Cleaning

- Used pandas and seaborn to analyse the dataset and perform some initial analysis, and found some trends in the dataset
- Used pandas to clean some corrupted data
- There is a clear relationship between the home team tending to score more goals per game across all the different leagues

> ![](./imgs/number_of_goals_per_game_in_each_league_for_home_and_away.png?raw=1)

## Feature Engineering

- Used pandas to join 3 separate datasets together
- Cleaned the dataset by filling in missing data
- Created a number of new features, ensuring no data leakage:
    - Form
    - Total goals
    - Discipline