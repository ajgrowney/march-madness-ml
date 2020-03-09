import sys
import pandas as pd

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




class Team_Historical:
    def __init__(self, name:str):
        self.name = name
        self.id = teams_df.loc[teams_df['TeamName'] == self.name]['TeamID'].values[0]
        self.conference, self.coaches, self.full_stats = {}, {}, {}
        
        for i, season, conf in teamsconf_df.loc[teamsconf_df['TeamID'] == self.id][['Season', 'ConfAbbrev']].itertuples(): self.conference[season] = conf
        
        for i, season, coach in teamscoach_df.loc[teamscoach_df['TeamID'] == self.id][['Season', 'CoachName']].itertuples():
            if season not in self.coaches: self.coaches[season] = [coach]
            else: self.coaches[season].append(coach)

def fill_team_stats(team: Team_Historical, df, win: bool):
    return


def fill_team_year(team: Team_Historical, year: int):
    # Setup Team's Dataframes for: Regular Season (team_rs), Conference Tournament (team_ct), and March Madness (team_mm)
    rs_df = regularseasonresults_df.loc[regularseasonresults_df['Season'] == year]
    ct_df = conferencetourney_df.loc[conferencetourney_df['Season'] == year]
    mm_df = ncaatourneyresults_df.loc[ncaatourneyresults_df['Season'] == year]
    team_mm = mm_df[(mm_df['WTeamID'] == team.id) | (mm_df['LTeamID'] == team.id)]
    team_rs = rs_df[(rs_df['WTeamID'] == team.id) | (rs_df['LTeamID'] == team.id)]
    ct_gamedays = list(ct_df[(ct_df['WTeamID'] == team.id) | (ct_df['LTeamID'] == team.id)]['DayNum'].values)
    conf_tournament_mask = team_rs['DayNum'].isin(ct_gamedays)
    team_ct, team_rs = team_rs[conf_tournament_mask], team_rs[~conf_tournament_mask]
    
    # Analyze Regular Season Data
    win_rs_mask, win_ct_mask = team_rs["WTeamID"] == team.id, team_ct["WTeamID"] == team.id
    team_rs_wins_df, team_rs_losses_df = team_rs[win_rs_mask], team_rs[~win_rs_mask]
    team_ct_wins_df, team_ct_losses_df = team_ct[win_ct_mask], team_ct[~win_ct_mask]

    fill_team_stats(team, team_rs_wins_df, win=True)
    fill_team_stats(team, team_rs_losses_df, win=False)

    

year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
team = Team_Historical(sys.argv[1])

fill_team_year(team,year)
