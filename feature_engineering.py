import pandas as pd
import numpy as np

def generate_home_away_features(df, window_length = 3):
    '''
    This function takes in the cleaned dataset and generates a some new features:
        - home_team_home_form: Home points won over the last (window_length) home games
        - away_team_away_form: Away points won over the last (window_length) away games
        - home_team_home_total_goals: Cumulative sum of home team goals scored at home for the current season
        - away_team_away_total_goals: Cumulative sum of away team goals scored at away for the current season
    '''
    return (df
        .assign(home_team_home_form = df.groupby(['home_team', 'season_year']).home_points.transform(lambda df_: df_.rolling(window_length, min_periods=1).sum().shift(1).fillna(0)),
            away_team_away_form = df.groupby(['away_team', 'season_year']).away_points.transform(lambda df_: df_.rolling(window_length, min_periods=1).sum().shift(1).fillna(0)),
            home_team_home_total_goals = df.groupby(['home_team', 'season_year']).home_goals.transform(lambda df_: df_.cumsum().shift(1).fillna(0)),
            away_team_away_total_goals = df.groupby(['home_team', 'season_year']).away_goals.transform(lambda df_: df_.cumsum().shift(1).fillna(0)))
    )

def generate_team_season_features(df, window_length=5, cards_multiplier_red = 1, cards_multiplier_yellow = 0.2):
    '''
    This function returns a tuple of new home and away features for the input dataframe, df:
    Each element in the tuple is a pandas dataframe with the following new columns:
        - form: Points won over the last (window_length) games (both home & away)
        - total_goals: Cumulative sum of goals scored for the current season (both home & away)
        - discipline: (cards_multiplier_red) * red_cards + (cards_multiplier_yellow) * yellow_cards for the last
            (window_length) games (both home & away)
    '''

    home_data = (df[['home_team', 'home_goals', 'home_yellow', 'home_red', 'home_points', 'match_round', 'season_year', 'league']]
                    .rename(columns=lambda col_name: col_name[5:] if col_name[:4] == 'home' else col_name))
    away_data = (df[['away_team', 'away_goals', 'away_yellow', 'away_red', 'away_points', 'match_round', 'season_year', 'league']]
                    .rename(columns=lambda col_name: col_name[5:] if col_name[:4] == 'away' else col_name))

    scores_data_long_format = (pd.concat([home_data, away_data])
                                .reset_index()
                                .assign(idx = lambda df_: np.arange(df_.shape[0]))
                                .set_index('idx'))

    scores_data_long_new_features = (scores_data_long_format
        .sort_values(['league', 'season_year', 'team', 'match_round'])
        .reset_index()
        .assign(form = lambda df_: df_.groupby(['team', 'season_year']).points.transform(lambda df: df.rolling(window_length, min_periods=1).sum().shift(1).fillna(0)),
            total_goals = lambda df_: df_.groupby(['team', 'season_year']).goals.transform(lambda df: df.cumsum().shift(1).fillna(0)))
        .assign(cards_temp = lambda df_: df_.red.mul(cards_multiplier_red).add(df_.yellow.mul(cards_multiplier_yellow)))
        .assign(discipline = lambda df_: df_.groupby(['team', 'season_year']).cards_temp.transform(lambda df: df.rolling(window_length, min_periods=1).sum().shift(1).fillna(0)))
        .set_index('idx').sort_index()
    )

    home_data_transformed = (scores_data_long_new_features[:home_data.shape[0]]
                                .set_index('match_id')
                                .drop(columns=['team', 'goals', 'yellow', 'red', 'points', 'match_round',
                                    'season_year', 'league', 'cards_temp'])
                                .rename(columns = lambda col: 'home_' + col))

    away_data_transformed = (scores_data_long_new_features[home_data.shape[0]:]
                                .set_index('match_id')
                                .drop(columns=['team', 'goals', 'yellow', 'red', 'points', 'match_round',
                                    'season_year', 'league', 'cards_temp'])
                                .rename(columns = lambda col: 'away_' + col))

    return home_data_transformed, away_data_transformed