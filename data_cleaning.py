import pandas as pd
import numpy as np
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