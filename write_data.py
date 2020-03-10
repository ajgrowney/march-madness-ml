import sys
import pandas as pd
import numpy as np
from statistics import mean
# Teams Data
teamsconf_df = pd.read_csv('Data/MTeamConferences.csv')
teams_df = pd.read_csv('Data/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
teamscoach_df = pd.read_csv('Data/MTeamCoaches.csv')

# Regular Szn Data
regularseasonresults_df = pd.read_csv('Data/MRegularSeasonDetailedResults.csv')

# Conference Tourney Data
conferencetourney_df = pd.read_csv('Data/MConferenceTourneyGames.csv')

# NCAA Tourney Data
ncaatourneyresults_df = pd.read_csv('Data/MNCAATourneyDetailedResults.csv')

class TeamSeason:
    def __init__(self, name: str, id, year: int ):
        self.name, self.id, self.year = name, id, year
        self.wins, self.losses = [], []
        self.stats = {
            "Points": [], "FGM": [], "FGA": [], "FGM3": [], "FGA3": [], "FTM": [], "FTA": [], "OR": [], "DR": [], "Ast": [], "TO": [], "Stl": [], "Blk": [], "Fouls": [],
            "OppPoints": [], "OppFGM": [], "OppFGA": [], "OppFGM3": [], "OppFGA3": [], "OppFTM": [], "OppFTA": [], "OppOR": [], "OppDR": [], "OppAst": [], "OppTO": [], "OppStl": [], "OppBlk": [], "OppFouls": []
        }
        # Filled in after populating the stats
        self.averages = {}

    def calculate_season_avgs(self):
        for k, v in self.stats.items():
            self.averages[k] = mean([val for (opp, val) in v])

    def get_data_columns(self):
        return list(self.averages.keys())

    def get_data(self):
        return np.array(list(self.averages.values()))

    def fill_game(self, game_row):
        if(game_row[3] == self.id):
            TeamPoints = game_row[4]
            OppID, OppPoints = game_row[5], game_row[6]
            self.wins.append(OppID)
            Loc, NumOT, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls = game_row[7:]
        elif(game_row[5] == self.id):
            TeamPoints = game_row[6]
            OppID, OppPoints = game_row[3], game_row[4]
            self.losses.append(OppID)
            Loc, NumOT, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls = game_row[7:]
        else:
            print("Error: TeamID not in game row")
            exit(0)

        self.stats["Points"].append((OppID, TeamPoints))
        self.stats["FGM"].append((OppID, FGM)), self.stats["FGA"].append((OppID, FGA)), self.stats["FGM3"].append((OppID, FGM3)), self.stats["FGA3"].append((OppID, FGA3)), self.stats["FTM"].append((OppID, FTM)), self.stats["FTA"].append((OppID, FTA)), self.stats["OR"].append((OppID, OR)), self.stats["DR"].append((OppID, DR)), self.stats["Ast"].append((OppID, Ast)), self.stats["TO"].append((OppID, TO)), self.stats["Stl"].append((OppID, Stl)), self.stats["Blk"].append((OppID, Blk)), self.stats["Fouls"].append((OppID, Fouls))
        
        self.stats["OppPoints"].append((OppID, OppPoints))
        self.stats["OppFGM"].append((OppID, OppFGM)), self.stats["OppFGA"].append((OppID, OppFGA)), self.stats["OppFGM3"].append((OppID, OppFGM3)), self.stats["OppFGA3"].append((OppID, OppFGA3)), self.stats["OppFTM"].append((OppID, OppFTM)), self.stats["OppFTA"].append((OppID, OppFTA)), self.stats["OppOR"].append((OppID, OppOR)), self.stats["OppDR"].append((OppID, OppDR)), self.stats["OppAst"].append((OppID, OppAst)), self.stats["OppTO"].append((OppID, OppTO)), self.stats["OppStl"].append((OppID, OppStl)), self.stats["OppBlk"].append((OppID, OppBlk)), self.stats["OppFouls"].append((OppID, OppFouls))
        
    # Param: rs_df { Pandas Dataframe } - Dataframe containing only games that team has played in
    def fill_regularseason(self, rs_df):
        ## All Conference Tournament Data is phase 2
        ## ct_df = conferencetourney_df.loc[conferencetourney_df['Season'] == year]
        ## ct_gamedays = list(ct_df[(ct_df['WTeamID'] == self.id) | (ct_df['LTeamID'] == self.id)]['DayNum'].values)
        ## conf_tournament_mask = rs_df['DayNum'].isin(ct_gamedays)

        ## Split into Regular Season and Conference Tournament Dataframes
        ## team_ct, team_rs = rs_df[conf_tournament_mask], rs_df[~conf_tournament_mask]
        ## win_ct_mask = team_ct["WTeamID"] == self.id
        ## team_ct_wins_df, team_ct_losses_df = team_ct[win_ct_mask], team_ct[~win_ct_mask]
        
        team_rs = rs_df

        # Fill Regular Season Data
        [self.fill_game(row) for row in team_rs.itertuples()]

        

class Team_Historical:
    def __init__(self, name:str, years_to_fill: list = []):
        self.name = name
        self.id = teams_df.loc[teams_df['TeamName'] == self.name]['TeamID'].values[0]
        self.conference, self.coaches = {}, {}
        
        # Dictionary: Key { int } - Year, Value { TeamSeason } - Season Stats
        self.team_seasons = {}

        for i, season, conf in teamsconf_df.loc[teamsconf_df['TeamID'] == self.id][['Season', 'ConfAbbrev']].itertuples(): self.conference[season] = conf
        
        for i, season, coach in teamscoach_df.loc[teamscoach_df['TeamID'] == self.id][['Season', 'CoachName']].itertuples():
            if season not in self.coaches: self.coaches[season] = [coach]
            else: self.coaches[season].append(coach)

        self.fill_years(years_to_fill)
    
    # Description: Call fill_year upon a list of years
    def fill_years(self, years: list):
        for year in years:
            self.fill_year(year)

    # Description: Fill the TeamSeason object of that year
    # Param: year { int } - year to fill with data
    # Return: { None }
    def fill_year(self, year: int):
        # Setup Team's Dataframes for: Regular Season (team_rs), Conference Tournament (team_ct), and March Madness (team_mm)
        rs_df = regularseasonresults_df.loc[regularseasonresults_df['Season'] == year]
        mm_df = ncaatourneyresults_df.loc[ncaatourneyresults_df['Season'] == year]
        team_mm = mm_df[(mm_df['WTeamID'] == self.id) | (mm_df['LTeamID'] == self.id)]
        team_rs = rs_df[(rs_df['WTeamID'] == self.id) | (rs_df['LTeamID'] == self.id)]
        
        t = TeamSeason(self.name, self.id, year)
        t.fill_regularseason(team_rs)
        t.calculate_season_avgs()
        self.team_seasons[year] = t
    
    # Description: Get the ML data for that TeamSeason
    def get_season_data(self, year):
        return self.team_seasons[year].get_data()

    def get_data_columns(self, year):
        return self.team_seasons[year].get_data_columns()




teams2018 = [
    "Virginia", "Cincinnati", "Tennessee", "Arizona", "Kentucky", "Miami FL", "Nevada", "Creighton", "Kansas St", "Texas", "Loyola-Chicago", "Davidson", "Buffalo", "Wright St", "Georgia St", "UMBC",
    "Xavier", "North Carolina", "Michigan", "Gonzaga", "Ohio St", "Houston", "Texas A&M", "Missouri", "Florida St", "Providence", "San Diego St", "S Dakota St", "UNC Greensboro", "Montana", "NC Central", "TX Southern",
    "Villanova", "Purdue", "Texas Tech", "Wichita St", "West Virginia", "Florida", "Arkansas", "Virginia Tech", "Alabama", "Butler", "St Bonaventure", "UCLA", "Murray St", "Marshall", "SF Austin", "CS Fullerton", "LIU Brooklyn", "Radford",
    "Kansas", "Duke", "Michigan St", "Auburn", "Clemson", "TCU", "Rhode Island", "Seton Hall", "NC State", "Oklahoma", "Arizona St", "Syracuse", "New Mexico St", "Col Charleston", "Bucknell", "Iona", "Penn"
]
df_columns, df_rows = None, []

for tm in teams2018:
    team = Team_Historical(tm, [2018, 2019])
    
    # Initialize Columns of DataFrame based on what is coming from data
    if(not df_columns):
        df_columns = ['TeamID'] + team.get_data_columns(2019)
    
    # Fill rows of DataFrame
    df_rows.append(np.append(team.id, team.get_season_data(2018)))


df = pd.DataFrame(df_rows, columns=df_columns)
df.to_csv('./Data/phase1.csv', index=False)