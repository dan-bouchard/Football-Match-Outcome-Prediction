import pandas as pd
import numpy as np
from zipfile import ZipFile


def import_leagues(original_dataset = True):
    '''
    This function is used to import the results dataset from the zip file Football-Dataset.zip into a pandas dataframe

    NOTE: if original_dataset parameter is True, the score column will be cleaned for the corrupted data

    Returns:
        A pandas dataframe with columns: home_team, away_team, score, link, year, round and league
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
        home_goals_numeric = data['score'].apply(lambda x: x.split('-')[0].isnumeric())
        list_of_err_idxs = home_goals_numeric[~home_goals_numeric].index.to_list()

        data.loc[list_of_err_idxs[0],'score'] = '0-0'
        data.loc[list_of_err_idxs[1],'score'] = '3-2'
        data.loc[list_of_err_idxs[2],'score'] = '0-1'
        # Fourth match postponed due to Covid pandemic so need to remove row
        data.loc[list_of_err_idxs[4],'score'] = '0-0'

        # Remove incompleted match
        data.drop(list_of_err_idxs[3], inplace=True)

        data.reset_index(inplace=True)
        data.drop(['index'], axis=1, inplace=True)
        return data

    df = pd.DataFrame()
    for league in league_names:
        df = pd.concat([df, import_league(league)])
    
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    df.rename(columns={'season': 'year', 'result': 'score'}, inplace=True)

    if original_dataset:
        df = clean_score_col(df)

    return df

def create_home_away_goals_and_result_attributes(data_df):
    '''
    This function is used to create the home and away goals columns from the imported scores dataframe by using the scores column. 
    The function assumes that the scores column is not corrupted.
    This function then also calculates the result column which can be: home_win, draw or away_win

    Returns:
        An altered pandas dataframe with new columns: home_goals, away_goals and result
    '''

    def calculate_result(df):
        if df['home_goals'] > df['away_goals']:
            return 'home_win' # Home Win
        elif df['home_goals'] == df['away_goals']:
            return 'draw' # Draw
        else:
            return 'away_win' # Away Win

    data_df['home_goals'] = data_df['score'].apply(lambda x: int(x.split('-')[0]))
    data_df['away_goals'] = data_df['score'].apply(lambda x: int(x.split('-')[1]))

    # New result attribute for each match
    data_df['result'] = data_df[['home_goals', 'away_goals']].apply(calculate_result, axis=1)

    return data_df

def import_match_info_data():
    '''
    This function is used to import the match details dataset from the csv file Match_Info.csv into a pandas dataframe

    Returns:
        A pandas dataframe with columns: link, date, referee, home_yellow, home_red, away_yellow, away_red
    '''

    def clean_referee_col(item):
        try:
            remove_separators = item.split('\r\n')[1]
            ref_name = remove_separators.split(':')[1]
            ref_name_out = ref_name[1:]
            return ref_name_out
        except:
            if item == '\r\n':
                return np.nan
            else:
                return item

    df = pd.read_csv('Match_Info.csv', na_values=np.nan)
    df.columns = df.columns.str.lower()
    df.rename(columns={'date_new': 'date'}, inplace=True)

    # Need to clean the referee column
    df['referee'] = df['referee'].apply(clean_referee_col)

    return df

def create_match_id_col(data):
    '''
    This function is used to create the match_id column from the imported match info dataframe by using the link and date columns. 
    The function assumes that the scores column is not corrupted.
    This function then removes the redundant link column

    Returns:
        An altered pandas dataframe with new column match_id and removes the redundant link column
    '''

    data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)

    def create_match_id_teams_match_info(x):
        split_str = x.split('/')
        return split_str[2] + '/' + split_str[3]

    def convert_to_season_year(x):
        year = x.year
        month = x.month
        if month >= 7:
            year += 1
        return year
    
    data['match_id'] = data['link'].apply(create_match_id_teams_match_info)  + '/' + data['date'].apply(convert_to_season_year).astype(str)
    data.drop('link', axis=1, inplace=True)
    return data

def import_team_info_data():
    '''
    This function is used to import the team details dataset from the csv file Team_Info.csv into a pandas dataframe

    Returns:
        A pandas dataframe with columns: team, city, country, stadium, capacity and pitch
    '''

    df = pd.read_csv('Team_Info.csv', na_values=np.nan)
    df.columns = df.columns.str.lower()
    df['capacity'] = df['capacity'].apply(lambda x: int(x.replace(",","")))

    return df