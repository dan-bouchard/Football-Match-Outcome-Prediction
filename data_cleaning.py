import pandas as pd
import numpy as np
import pickle
from zipfile import ZipFile


def import_leagues(original_dataset = True):
    '''
    This function is used to import the results dataset from the zip file Football-Dataset.zip into a pandas dataframe

    NOTE: if original_dataset parameter is True, the score column will be cleaned for the corrupted data

    Returns:
        A pandas dataframe with columns: home_team, away_team, score, link, season_year, match_round and league
    '''
    
    league_names = ["premier_league", "championship", "primera_division", "segunda_division",
                "bundesliga", "2_liga", "serie_a", "serie_b", "ligue_1", "ligue_2", 
                "eredivisie", "eerste_divisie"]
    
    def import_league(league_name, df=None):
        if df is None:
            df = pd.DataFrame()
        years = list(range(1990,2022))
        
        for year in years:
            with ZipFile("Football-Dataset.zip") as myzip:
                data = myzip.open(f"Football-Dataset/{league_name}/Results_{year}_{league_name}.csv")
            season = pd.read_csv(data)
            season.columns = season.columns.str.lower()
            df = pd.concat([df, season])
        
        return df
    
    def clean_score_col(data):   
        score_idx_to_fix = data[~data.score.str.contains('^\d+-\d+$')].index.to_list()
        data.loc[score_idx_to_fix[0],'score'] = '0-0'
        data.loc[score_idx_to_fix[1],'score'] = '3-2'
        data.loc[score_idx_to_fix[2],'score'] = '0-1'
        # Fourth match postponed due to Covid pandemic so need to remove row
        data.loc[score_idx_to_fix[4],'score'] = '0-0'

        # Remove incompleted match
        data = (data
            .drop(score_idx_to_fix[3])
            .reset_index()
            .drop('index', axis=1))

        return data

    df = pd.DataFrame()
    for league in league_names:
        df = pd.concat([df, import_league(league)])
    
    df = (df.reset_index()
        .drop(columns='index')
        .rename(columns={'season': 'season_year', 'result': 'score', 'round': 'match_round'})
        .assign(season_year = lambda df_: df_.season_year.astype('int'),
            match_round = lambda df_: df_.match_round.astype('int'))
    )

    if original_dataset:
        df = clean_score_col(df)

    return df

def tweak_scores_df(data_df):
    '''
    This function is used to create the home and away goals columns from the imported scores dataframe by using the scores column. 
    The function assumes that the scores column is not corrupted.
    This function then also calculates the result column which can be: home_win, draw or away_win

    Returns:
        An altered pandas dataframe with new columns: home_goals, away_goals, result, home_points and away_points
    '''

    def split_home_away_goals(df):
        return (df.join(df.score.str.split('-', expand=True)
            .astype('int')
            .rename(columns={0: 'home_goals', 1: 'away_goals'})))

    return (data_df
            .pipe(split_home_away_goals)
            .assign(result = lambda df_: np.select([df_.home_goals > df_.away_goals, df_.home_goals == df_.away_goals], ['home_win', 'draw'], 'away_win'))
            .assign(result = lambda df_: df_.result.astype('category'))
            .assign(home_points = lambda df_: np.select([df_.result == 'home_win', df_.result == 'draw'], [3, 1], 0),
                    away_points = lambda df_: np.select([df_.result == 'home_win', df_.result == 'draw'], [0, 1], 3))
            .drop_duplicates(subset=['home_team', 'away_team', 'season_year'], keep='first'))
            

def create_match_id_col_from_link(data):
    '''
    This function is used to create the match_id column from the imported scores dataframe by using the link and year columns. 

    Returns:
        An altered pandas dataframe with new column match_id
    '''

    return data.assign(match_id = data.link.str.extract('match/([\w-]+\/[\w-]+\/)')[0].str.cat(data.season_year.astype('str')))

def import_match_info_data():
    '''
    This function is used to import the match details dataset from the csv file Match_Info.csv into a pandas dataframe

    Returns:
        A pandas dataframe with columns: link, date, referee, home_yellow, home_red, away_yellow, away_red
    '''

    df = pd.read_csv('Match_Info.csv', na_values=np.nan)
    return (df.rename(columns = dict(zip(df.columns, df.columns.str.lower())))
                .assign(referee = lambda df_: df_.referee.str.extract('\r\nReferee: ([\w -.]+)'),
                    date_new = lambda df_: pd.to_datetime(df_.date_new))
                .rename(columns={'date_new': 'date'})
            )

def create_match_id_col(data):
    '''
    This function is used to create the match_id column from the imported match info dataframe by using the link and date columns. 
    This function then removes the redundant link column

    Returns:
        An altered pandas dataframe with new column match_id and removes the redundant link column
    '''

    return (data
        .assign(match_id = 
                    data.link.str.extract('/match[_\d]*/([\w-]+/[\w-]+/[\d]+)')[0])
                    # .str.cat(data.date.dt.year
                    #             .mask(data.date.dt.day_of_year > 196, data.date.dt.year.add(1)).astype('str'))) # day 196 is 15th July
        .drop(columns='link')
        )

def import_team_info_data():
    '''
    This function is used to import the team details dataset from the csv file Team_Info.csv into a pandas dataframe

    Returns:
        A pandas dataframe with columns: team, city, country, stadium, capacity and pitch
    '''  

    df = pd.read_csv('Team_Info.csv', na_values=np.nan)
    # drop indexes with Portugal
    portugal_idxs = df.query("Country == 'Portugal'").index.to_list()
    return (df.rename(columns = dict(zip(df.columns, df.columns.str.lower())))
                .assign(capacity = lambda df_: df_.capacity.str.replace(',','').astype('int'))
                .drop(portugal_idxs)
            )

def import_elo_data():
    '''
    This function is used to import the elo data from the pickle file

    Returns:
        A pandas data frame with columns: link, home_elo and home_away
    '''

    elo_dict = pickle.load(open('elo_dict.pkl', 'rb'))
    elo_link_list = []
    elo_home_list = []
    elo_away_list = []
    for key, value in elo_dict.items():
        elo_link_list.append(key)
        elo_home_list.append(value['Elo_home'])
        elo_away_list.append(value['Elo_away'])

    elo_df = pd.DataFrame({'link': elo_link_list, 'home_elo': elo_home_list, 'away_elo': elo_away_list})

    return elo_df

def merge_data_into_one_df(scores_df, match_df, team_df, elo_df):
    '''
    This function merges the four dataframes together into a single dataframe
    
    Returns:
        A pandas dataframe with columns:

        home_team, away_team, score, link, season_year, match_round,
        league, home_goals, away_goals, result, home_points,
        away_points, date, referee, home_yellow, home_red,
        away_yellow, away_red, capacity, home_elo, away_elo

    '''

    scores_match_info_df = pd.merge(scores_df, match_df, how='left', on="match_id")
    team_reduced_df = team_df.copy()
    team_reduced_df = team_reduced_df[['team', 'capacity']]

    scores_match_team_info_df = (pd.merge(scores_match_info_df, team_reduced_df, how='left', left_on='home_team', right_on='team')
                                    .drop(columns = 'team'))
    
    scores_match_team_info_elo_df = pd.merge(scores_match_team_info_df, elo_df, how='left', on="link").set_index('match_id')

    return scores_match_team_info_elo_df

def import_and_merge_data_pipeline():
    '''
    This function is a pipeline for importing and merging all the initial data together.

    Returns:
        A pandas dataframe with columns:

        home_team, away_team, score, link, season_year, match_round,
        league, home_goals, away_goals, result, home_points,
        away_points, date, referee, home_yellow, home_red,
        away_yellow, away_red, capacity, home_elo, away_elo

    '''
    elo_df = import_elo_data()

    scores_df = import_leagues()
    scores_df = tweak_scores_df(scores_df)
    scores_df = create_match_id_col_from_link(scores_df)

    match_info_df = import_match_info_data()
    match_info_df = create_match_id_col(match_info_df)

    team_info_df = import_team_info_data()

    merged_data_df = merge_data_into_one_df(scores_df, match_info_df, team_info_df, elo_df)
    
    return merged_data_df

def calculate_median_yellow_red_cards_for_each_year(merged_df):
    ''''
    Given the merged dataframe, return the median: home_yellow, home_red, away_yellow, away_red for each year required

    Returns:
        A pandas dataframe with columns: years with missing cards data
        A list of tuples with (year, league) where the full season has missing cards data
    '''

    missing_cards_df = (merged_df
                            .assign(missing_yellow = merged_df.home_yellow.isna())
                            .groupby(['season_year', 'league'])
                            .agg(missing_cards = ('missing_yellow', 'sum'),
                                total_games = ('missing_yellow', 'count'))
                        )
    
    # so there only certain seasons where the full season has missing cards data
    year_season_full_missing_cards = (missing_cards_df[missing_cards_df.missing_cards > 0]
                                    [(missing_cards_df[missing_cards_df.missing_cards > 0].missing_cards) == 
                                        (missing_cards_df[missing_cards_df.missing_cards > 0].total_games)]
                                    ).index.to_list()
    
    # Going to calculate the median value for each year and replace the missing season value with that value
    if len(year_season_full_missing_cards) > 0:
        years_missing, _ = tuple(map(list, zip(*year_season_full_missing_cards)))
        median_values_for_missing_years_df = pd.DataFrame()
        for year in set(years_missing):
            median_values_for_missing_years_df[year] = (merged_df
                                                            .query("season_year == @year")
                                                            [['home_yellow', 'home_red', 'away_yellow', 'away_red']]
                                                            .median()
                                                        )
    else:
        median_values_for_missing_years_df = pd.DataFrame()
    
    return median_values_for_missing_years_df, year_season_full_missing_cards

def fill_median_yellow_red_cards_for_each_year(merged_df):
    '''
    Given the merged dataframe, return the same dataframe with the missing values for:
        - home_yellow
        - home_red
        - away_yellow
        - away_red
    
    filled in only when the full season data is missing (i.e. year and league)
    '''
    median_values, year_league = calculate_median_yellow_red_cards_for_each_year(merged_df)
    if len(year_league) > 0:
        for year, league in year_league:
        
            selection_idx = (merged_df.season_year == year) & (merged_df.league == league)

            merged_df.loc[selection_idx, ['home_yellow', 'home_red', 'away_yellow', 'away_red']] = (merged_df
                .query("(season_year == @year) and (league == @league)")
                [['home_yellow', 'home_red', 'away_yellow', 'away_red']]
                .fillna(median_values.to_dict()[year])
            )
    
    return merged_df

def calculate_year_league_teams_missing_full_season_cards_data(merged_df):
    '''
    A function which calculates all the tuple's (year, league, team) for when a single team has got a full season worth
    of missing data

    Returns:
        A list of tuples (year, league, team)
    '''

    missing_cards_teams_df = (((pd.concat([(merged_df
                [['season_year', 'league', 'home_team', 'home_yellow']]
                .rename(columns = lambda col_name: col_name[5:] if col_name[:4] == 'home' else col_name)), 
                (merged_df
                [['season_year', 'league', 'away_team', 'away_yellow']]
                .rename(columns = lambda col_name: col_name[5:] if col_name[:4] == 'away' else col_name))], axis=0))
            .assign(idx = lambda df_: np.arange(df_.shape[0]))
            .set_index('idx'))
        .assign(missing_yellow = lambda df_: df_.yellow.isna())
        .groupby(['season_year', 'league', 'team'])
        .agg(missing_cards=('missing_yellow', 'sum'),
            total_games = ('missing_yellow', 'count'))
    )
    
    # so there certain seasons where some teams have no cards data for the full season
    missing_cards_teams_full_season_idx = missing_cards_teams_df[missing_cards_teams_df.missing_cards == missing_cards_teams_df.total_games].index.to_list()
    return missing_cards_teams_full_season_idx

def calculate_median_cards_value_for_year_league_teams_missing_full_season_cards_data(merged_df):
    ''''
    Given the merged dataframe, return the median: home_yellow, home_red, away_yellow, away_red for each year, league, team required

    Returns:
        A dictionary with keys: tuple (year, league) with missing cards data
            and values: dictionary median value for each of home_yellow, home_red, away_yellow, away_red 
    '''
    
    missing_year_league_teams = calculate_year_league_teams_missing_full_season_cards_data(merged_df)
    
    cards_league_year = dict()
    if len(missing_year_league_teams) > 0:
        year, league, _ = tuple(zip(*missing_year_league_teams))
        unique_year_league_set = list(set(zip(year, league)))
        

        for year_league in unique_year_league_set:
            year, league = year_league
            cards_league_year[year_league] = (merged_df
                .query("(season_year == @year) and (league == @league)")
                [['home_yellow', 'home_red', 'away_yellow', 'away_red']]
                .median()
                .to_dict()
            )
    return cards_league_year

def fill_median_yellow_red_cards_for_each_year_league(merged_df):
    '''
    Given the merged dataframe, return the same dataframe with the missing values for:
        - home_yellow
        - home_red
        - away_yellow
        - away_red
    
    filled in only when a full team's season data is missing (i.e. year, league and team)
    '''

    missing_year_league_teams = calculate_year_league_teams_missing_full_season_cards_data(merged_df)
    cards_for_each_league_year = calculate_median_cards_value_for_year_league_teams_missing_full_season_cards_data(merged_df)
    
    for year_league_team in missing_year_league_teams:
        year, league, team = year_league_team
        year_league = (year, league)
        home_selection_idx = ((merged_df.season_year == year) & (merged_df.league == league)
            & (merged_df.home_team == team))
        away_selection_idx = ((merged_df.season_year == year) & (merged_df.league == league)
            & (merged_df.away_team == team))
        
        home_dict = {'home_yellow': cards_for_each_league_year[year_league]['home_yellow'], 
            'home_red': cards_for_each_league_year[year_league]['home_red']}
        away_dict = {'away_yellow': cards_for_each_league_year[year_league]['away_yellow'], 
            'away_red': cards_for_each_league_year[year_league]['away_red']}
        
        merged_df.loc[home_selection_idx, ['home_yellow', 'home_red']] = (merged_df
            .query("(season_year == @year) and (league == @league) and (home_team == @team)")
            [['home_yellow', 'home_red']]
            .fillna(home_dict))
        
        merged_df.loc[away_selection_idx, ['away_yellow', 'away_red']] = (merged_df
            .query("(season_year == @year) and (league == @league) and (away_team == @team)")
            [['away_yellow', 'away_red']]
            .fillna(away_dict))
        
    return merged_df

def calculate_median_value_for_each_year_league(merged_df, cols):
    ''''
    Given the merged dataframe, return the median cols for each (year, league) required

    Returns:
        A dictionary with keys: tuple (year, league) with missing cols data
            and values: dictionary median (cols) for that (year, league) 
    '''
    missing_cols_df = (merged_df
                            .assign(missing_col = merged_df[cols[0]].isna())
                            .groupby(['season_year', 'league'])
                            .agg(missing_col = ('missing_col', 'sum'),
                                    total_games = ('missing_col', 'count'))
    )

    seasons_with_some_missing_cols = missing_cols_df[missing_cols_df.missing_col > 0].index.to_list()

    median_values_for_missing_cols = {}
    for year_league in seasons_with_some_missing_cols:
        year, league = year_league
        median_values_for_missing_cols[year_league] = (merged_df
            .query("(season_year == @year) and (league == @league)")
            .loc[:, cols]
            .median()
            .to_dict()
        )

    return median_values_for_missing_cols

def fill_median_value_for_each_year_league(merged_df, cols):
    '''
    Given the merged dataframe, return the same dataframe with the missing values for cols    
    filled in with median for that season (i.e. year and league)
    '''
    
    median_values_for_missing_cols = calculate_median_value_for_each_year_league(merged_df, cols)
    for key, cols_values in median_values_for_missing_cols.items():
        
        year, league = key
        
        selection_idx = (merged_df.season_year == year) & (merged_df.league == league)

        merged_df.loc[selection_idx, cols] = (merged_df
            .query("(season_year == @year) and (league == @league)")
            [cols]
            .fillna(cols_values)
        )
    return merged_df