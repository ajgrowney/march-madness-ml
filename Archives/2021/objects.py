import numpy as np
import pandas as pd
import re
DATA_ROOT = "/Users/andrewgrowney/Data/kaggle/marchmadness-2021"

# Teams Data
teamsconf_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamConferences.csv')
teams_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
teamscoach_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamCoaches.csv')
ordinals_df = pd.read_csv(f"{DATA_ROOT}/Stage2/MMasseyOrdinals.csv")
# Regular Season Data
regularseasonresults_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MRegularSeasonDetailedResults.csv')
# Conference Tourney Data
conferencetourney_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MConferenceTourneyGames.csv')

# NCAA Tourney Data
tourney_seeds = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv')
ncaatourneyresults_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneyDetailedResults.csv')

class TeamSeason:
    def __init__(self, id, year:int, tournament_seed:int):
        self.id, self.year = id, year
        self.wins, self.losses, self.win_pct, self.opp_win_pct = [], [], None, None
        self.sos, self.sov = None, None
        self.tourney_seed = tournament_seed
        try:
            self.conf = teamsconf_df[(teamsconf_df['TeamID'] == self.id) & (teamsconf_df['Season'] == self.year)]['ConfAbbrev'].values[0]
            self.coach = teamscoach_df[(teamscoach_df['TeamID'] == self.id) & (teamscoach_df['Season'] == self.year)]['CoachName'].values[0]
        except Exception as e:
            raise Exception(e)

        self.stats = {
            "Points": [], "Poss": [], "OE": [], "DE": [], "FGM": [], "FGA": [], "FGM3": [], "FGA3": [], "FTM": [], "FTA": [], "OR": [], "DR": [], "Ast": [], "TO": [], "Stl": [], "Blk": [], "Fouls": [],
            "OppPoints": [], "OppFGM": [], "OppFGA": [], "OppFGM3": [], "OppFGA3": [], "OppFTM": [], "OppFTA": [], "OppOR": [], "OppDR": [], "OppAst": [], "OppTO": [], "OppStl": [], "OppBlk": [], "OppFouls": []
        }
        # Filled in after populating the stats
        self.means, self.averages, self.stdev = {}, {}, {}



    def calculate_season_stats(self):
        # Calculate Win Pct
        self.win_pct = len(self.wins) / (len(self.wins) + len(self.losses))
        # Calculate distributions for status
        for k, v in self.stats.items():
            stat_vals = [val for (opp_id, val) in v]
            self.means[k] = np.mean(stat_vals)
            self.averages[k] = np.average(stat_vals)
            self.stdev[k] = np.std(stat_vals)

    
    def calculate_post_season_stats(self, league_season_data):
        """ Calculate statistics that you need other teams info for """
        """ Param: league_season_data: dict<id,TeamSeason> """
        
        opp_wp = []
        opp_opp_wp = []
        for o in self.wins:
            opponent = league_season_data[o]
            opp_wp.append(opponent.win_pct)
            opp_games = opponent.wins + opponent.losses
            opp_opp_wp.append(np.average([league_season_data[oo].win_pct for oo in opp_games]))

        self.sov = np.average(opp_wp)
        
        for o in self.losses:
            opponent = league_season_data[o]
            opp_wp.append(opponent.win_pct)
            opp_games = opponent.wins + opponent.losses
            opp_opp_wp.append(np.average([league_season_data[oo].win_pct for oo in opp_games]))


        ow = np.average(opp_wp)
        oow = np.average(opp_opp_wp)
        self.sos = (2*ow + oow) / 3
        return
            

    def get_data_columns(self):
        return list([k+"_mean" for k in self.means.keys()]) + \
            list([k+"_stdev" for k in self.stdev.keys()]) + \
             ["WinPct", "SOS", "SOV", "Seed"]

    def get_data(self, columns:list = None):
        if columns is None:
            return np.array(
                list(self.means.values()) + 
                list(self.stdev.values()) + 
                [self.win_pct, self.sos, self.sov, self.tourney_seed]
            )
        else:
            vals = []
            for c in columns:
                vals.append(self.averages[c])
            return np.array(vals)

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

        Poss = (FGA - OR) + TO + (.475 * FTA)
        self.stats["Poss"].append((OppID, Poss))
        self.stats["OE"].append((OppID, (TeamPoints*100)/Poss))
        self.stats["DE"].append((OppID, (OppPoints*100)/Poss))

        self.stats["Points"].append((OppID, TeamPoints))
        self.stats["FGM"].append((OppID, FGM)), self.stats["FGA"].append((OppID, FGA)), self.stats["FGM3"].append((OppID, FGM3)), self.stats["FGA3"].append((OppID, FGA3)), self.stats["FTM"].append((OppID, FTM)), self.stats["FTA"].append((OppID, FTA)), self.stats["OR"].append((OppID, OR)), self.stats["DR"].append((OppID, DR)), self.stats["Ast"].append((OppID, Ast)), self.stats["TO"].append((OppID, TO)), self.stats["Stl"].append((OppID, Stl)), self.stats["Blk"].append((OppID, Blk)), self.stats["Fouls"].append((OppID, Fouls))
        
        self.stats["OppPoints"].append((OppID, OppPoints))
        self.stats["OppFGM"].append((OppID, OppFGM)), self.stats["OppFGA"].append((OppID, OppFGA)), self.stats["OppFGM3"].append((OppID, OppFGM3)), self.stats["OppFGA3"].append((OppID, OppFGA3)), self.stats["OppFTM"].append((OppID, OppFTM)), self.stats["OppFTA"].append((OppID, OppFTA)), self.stats["OppOR"].append((OppID, OppOR)), self.stats["OppDR"].append((OppID, OppDR)), self.stats["OppAst"].append((OppID, OppAst)), self.stats["OppTO"].append((OppID, OppTO)), self.stats["OppStl"].append((OppID, OppStl)), self.stats["OppBlk"].append((OppID, OppBlk)), self.stats["OppFouls"].append((OppID, OppFouls))
        
    # Param: team_rs_df { Pandas Dataframe } - Dataframe containing only games that team has played in
    def fill_regularseason(self, team_rs_df):
        # Fill Regular Season Data
        [self.fill_game(row) for row in team_rs_df.itertuples()]


class TeamHistorical:
    def __init__(self, id:int, years_to_fill: list = []):
        self.id = id
        self.valid_years = set(years_to_fill)
        self.name = teams_df.loc[teams_df['TeamID'] == self.id]['TeamName'].values[0]
        self.conference, self.coaches = {}, {}
        
        # Dictionary: Key { int } - Year, Value { TeamSeason } - Season Stats
        self.team_seasons = {}
        self.fill_years(years_to_fill)
    
    # Description: Call fill_year upon a list of years
    def fill_years(self, years: list):
        for year in years:
            self.fill_year(year)
        return

    # Description: Fill the TeamSeason object of that year
    # Param: year { int } - year to fill with data
    # Return: { None }
    def fill_year(self, year: int):
        # Setup Team's Dataframes for: Regular Season (team_rs), Conference Tournament (team_ct), and March Madness (team_mm)
        rs_df = regularseasonresults_df.loc[regularseasonresults_df['Season'] == year]
        team_rs = rs_df[(rs_df['WTeamID'] == self.id) | (rs_df['LTeamID'] == self.id)]
        raw_tournament_seed = tourney_seeds[(tourney_seeds['Season'] == year) & (tourney_seeds['TeamID'] == self.id)]['Seed'].values
        if len(raw_tournament_seed) == 0:
            seed = None
        else:
            seed = int(re.sub("[^0-9^.]", "",raw_tournament_seed[0]).lstrip('0'))
        try:
            t = TeamSeason(self.id, year, seed)
            t.fill_regularseason(team_rs)
            t.calculate_season_stats()
            self.team_seasons[year] = t
        except:
            print(f"Unable to make season: {self.id}, {year}")
            self.valid_years.remove(year)
    
    # Description: Get the ML data for that TeamSeason
    def get_season_data(self, year):
        return self.team_seasons[year].get_data()

    def get_data_columns(self, year):
        return self.team_seasons[year].get_data_columns()


