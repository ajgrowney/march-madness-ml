from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import re
from mm_analytics.utilities import DATA_ROOT

# Teams Data
teamsconf_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamConferences.csv')
teams_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
teamscoach_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamCoaches.csv')
ORDINALS_DF = pd.read_csv(f"{DATA_ROOT}/Stage2/MMasseyOrdinals_thru_Season2023_Day128.csv")
# Regular Season Data
regularseasonresults_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MRegularSeasonDetailedResults.csv')

# Conference Tourney Data
conferencetourney_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MConferenceTourneyGames.csv')

# NCAA Tourney Data
SEEDS_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv')
ncaatourneyresults_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneyDetailedResults.csv')

class TeamSeasonOrdinals:
    def __init__(self, id, year) -> None:
        self.id = id
        self.year = year
        self.rpi = {"count": 0, "average": None, "last": None}
    
    def fill_data(self, ordinals_df):
        """Use the MMasseyOrdinals data to fill the RPI data for the team
        :param ordinals_df { pd.DataFrame }: DataFrame containing the MMasseyOrdinals data
        """
        rpi_data = ordinals_df.loc[
            (ordinals_df["SystemName"] == "RPI") & 
            (ordinals_df["TeamID"] == self.id) & 
            (ordinals_df["Season"] == self.year)][["RankingDayNum","OrdinalRank"]]

        if len(rpi_data) > 0:
            rpi_season_ranks = {}
            for _, daynum, rank in rpi_data.itertuples():
                self.rpi["count"] += 1
                rpi_season_ranks[daynum] = rank
            self.rpi["average"] = np.average(list(rpi_season_ranks.values()))
            self.rpi["last"] = rpi_season_ranks[max(rpi_season_ranks.keys())]
        else:
            self.rpi["count"] = 0
            self.rpi["average"] = 999
            self.rpi["last"] = 999

        return
    
    def get_data(self):
        return [self.rpi["average"], self.rpi["count"], self.rpi["last"]]
    
    def get_data_columns(self):
        return ["rpi_avg", "rpi_count", "rpi_last"]

class TeamSeason:
    def __init__(self, id, year:int, tournament_seed:int, regular_season_df:pd.DataFrame = None,
                 teamsconf_df:pd.DataFrame = None, teamscoach_df:pd.DataFrame = None, ordinals_df:pd.DataFrame = None):
        self.id, self.year = id, year
        self.wins, self.losses, self.win_pct, self.opp_win_pct = [], [], None, None
        self.sos, self.sov = None, None
        self.tourney_seed = tournament_seed
        # Ordinal Info
        self.ordinal_data = TeamSeasonOrdinals(id, year)
        self.conf, self.coach = None, None
        if teamsconf_df is not None:
            self.conf = teamsconf_df[(teamsconf_df['TeamID'] == self.id) & (teamsconf_df['Season'] == self.year)]['ConfAbbrev'].values[0]
        if teamscoach_df is not None:
            self.coach = teamscoach_df[(teamscoach_df['TeamID'] == self.id) & (teamscoach_df['Season'] == self.year)]['CoachName'].values[0]

        self.stats = {
            "Points": [], "Poss": [], "OE": [], "DE": [], "FGM": [], "FGA": [], "FGM3": [], "FGA3": [], "FTM": [], "FTA": [], "OR": [], "DR": [], "Ast": [], "TO": [], "Stl": [], "Blk": [], "Fouls": [],
            "OppPoints": [], "OppFGM": [], "OppFGA": [], "OppFGM3": [], "OppFGA3": [], "OppFTM": [], "OppFTA": [], "OppOR": [], "OppDR": [], "OppAst": [], "OppTO": [], "OppStl": [], "OppBlk": [], "OppFouls": []
        }
        # Filled in after populating the stats
        self.means, self.averages, self.stdev = {}, {}, {}
        # Fill the regular season stats
        for row in regular_season_df.itertuples():
            self.fill_game(row)
        self.calculate_season_stats(ordinals_df)

    def calculate_season_stats(self, ordinals_df):
        # Calculate Win Pct
        self.win_pct = len(self.wins) / (len(self.wins) + len(self.losses))
        # Calculate distributions for status
        for k, v in self.stats.items():
            stat_vals = [val for (opp_id, val) in v]
            self.means[k] = np.mean(stat_vals)
            self.averages[k] = np.average(stat_vals)
            self.stdev[k] = np.std(stat_vals)
        
        # Calculate ordinals info
        if ordinals_df is not None:
            self.ordinal_data.fill_data(ordinals_df)

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
             self.ordinal_data.get_data_columns() + \
             ["WinPct", "SOS", "SOV", "Seed"]

    def get_data(self, columns:list = None):
        if columns is None:
            return np.array(
                list(self.means.values()) + 
                list(self.stdev.values()) + 
                self.ordinal_data.get_data() +
                [self.win_pct, self.sos, self.sov, self.tourney_seed]
            )
        else:
            vals = []
            for c in columns:
                vals.append(self.averages[c])
            return np.array(vals)

    def fill_game(self, game_row:Tuple):
        """Take a dataframe row and fill in the stats for that game
        :param game_row: Tuple - Row of the Regular Season DataFrame
        """
        if(game_row[3] == self.id):
            # Team Win Stats
            TeamPoints, OppID, OppPoints = game_row[4], game_row[5], game_row[6]
            self.wins.append(OppID)
            _, _, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls = game_row[7:]
        elif(game_row[5] == self.id):
            # Team Loss Stats
            OppID, OppPoints, TeamPoints = game_row[3], game_row[4], game_row[6]
            self.losses.append(OppID)
            _, _, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls = game_row[7:]
        else:
            print("Error: TeamID not in game row")
            exit(0)

        poss = (FGA - OR) + TO + (.475 * FTA)
        self.stats["Poss"].append((OppID, poss))
        self.stats["OE"].append((OppID, (TeamPoints*100)/poss))
        self.stats["DE"].append((OppID, (OppPoints*100)/poss))

        self.stats["Points"].append((OppID, TeamPoints))
        self.stats["FGM"].append((OppID, FGM)), self.stats["FGA"].append((OppID, FGA)), self.stats["FGM3"].append((OppID, FGM3)), self.stats["FGA3"].append((OppID, FGA3)), self.stats["FTM"].append((OppID, FTM)), self.stats["FTA"].append((OppID, FTA)), self.stats["OR"].append((OppID, OR)), self.stats["DR"].append((OppID, DR)), self.stats["Ast"].append((OppID, Ast)), self.stats["TO"].append((OppID, TO)), self.stats["Stl"].append((OppID, Stl)), self.stats["Blk"].append((OppID, Blk)), self.stats["Fouls"].append((OppID, Fouls))
        
        self.stats["OppPoints"].append((OppID, OppPoints))
        self.stats["OppFGM"].append((OppID, OppFGM)), self.stats["OppFGA"].append((OppID, OppFGA)), self.stats["OppFGM3"].append((OppID, OppFGM3)), self.stats["OppFGA3"].append((OppID, OppFGA3)), self.stats["OppFTM"].append((OppID, OppFTM)), self.stats["OppFTA"].append((OppID, OppFTA)), self.stats["OppOR"].append((OppID, OppOR)), self.stats["OppDR"].append((OppID, OppDR)), self.stats["OppAst"].append((OppID, OppAst)), self.stats["OppTO"].append((OppID, OppTO)), self.stats["OppStl"].append((OppID, OppStl)), self.stats["OppBlk"].append((OppID, OppBlk)), self.stats["OppFouls"].append((OppID, OppFouls))


def get_season_seeds(year, seeds_df) -> Dict[int, int]:
    """Fetch a mapping for TeamID to Seed for that year
    :param int year: year to fill with data
    :param seeds_df: Pandas DataFrame - Tournament Seeds of that season
    :return: Dict<int, int> - Dictionary of TeamID to Seed
    """
    team_seeds = {}
    for team_id, seed in seeds_df[seeds_df['Season'] == year][['TeamID', 'Seed']].values:
        team_seeds[team_id] = int(re.sub("[^0-9^.]", "",seed).lstrip('0'))
    return team_seeds

def get_team_seasons(year, regular_season_df, seeds_df: pd.DataFrame = None, teams_conf_df = None,
                teams_coach_df = None, ordinals_df = None) -> Dict[int, TeamSeason]:
    """From a set of regular season, conference tournament, and march madness games
    fill in all possible TeamSeasons
    :param int year: year to fill with data
    :param regular_season_df: Pandas DataFrame - Regular Season Games of that season
    :param seeds_df: Pandas DataFrame - Tournament Seeds of that season
    :param teams_conf_df: Pandas DataFrame - Conference Information of that season
    :param teams_coach_df: Pandas DataFrame - Coach Information of that season
    :param ordinals_df: Pandas DataFrame - Ordinals Information of that season
    :return: Dict<int, TeamSeason> - Dictionary of TeamSeasons
    """
    team_seasons: Dict[int, TeamSeason] = {}
    # Fetch Team IDs to Evaluate for the Season
    teams = set(regular_season_df['WTeamID'].values).union(set(regular_season_df['LTeamID'].values))
    seeds = get_season_seeds(year, seeds_df)
    for team_id in teams:
        team_seed = seeds.get(team_id)
        print(f"Team: {team_id}, Seed: {team_seed}, Year: {year}")
        team_games = regular_season_df[(regular_season_df['WTeamID'] == team_id) | (regular_season_df['LTeamID'] == team_id)]
        team_seasons[team_id] = TeamSeason(team_id, year, team_seed, team_games, teams_conf_df, teams_coach_df, ordinals_df)
    
    for team_season in team_seasons.values():
        team_season.calculate_post_season_stats(team_seasons)
    
    return team_seasons


class TeamHistorical:
    def __init__(self, id:int, years_to_fill: list = []):
        self.id = id
        self.valid_years = set(years_to_fill)
        self.name = teams_df.loc[teams_df['TeamID'] == self.id]['TeamName'].values[0]
        self.conference, self.coaches = {}, {}
        
        # Dictionary: Key { int } - Year, Value { TeamSeason } - Season Stats
        self.team_seasons = {}
        for year in years_to_fill:
            self.team_seasons[year] = self.fill_team_season(year)

    def fill_team_season(self, year: int) -> TeamSeason:
        """Fill the TeamSeason object of that year
        :param int year: year to fill with data
        :return: None
        """
        # Setup Team's Dataframes for: Regular Season (team_rs), Conference Tournament (team_ct), and March Madness (team_mm)
        rs_df = regularseasonresults_df.loc[regularseasonresults_df['Season'] == year]
        team_rs = rs_df[(rs_df['WTeamID'] == self.id) | (rs_df['LTeamID'] == self.id)]
        raw_tournament_seed = tourney_seeds[(tourney_seeds['Season'] == year) & (tourney_seeds['TeamID'] == self.id)]['Seed'].values
        seed = None if len(raw_tournament_seed) == 0 else int(re.sub("[^0-9^.]", "",raw_tournament_seed[0]).lstrip('0'))
        try:
            return TeamSeason(self.id, year, seed, team_rs)
        except:
            print(f"Unable to make season: {self.id}, {year}")
            self.valid_years.remove(year)
    
    # Description: Get the ML data for that TeamSeason
    def get_season_data(self, year):
        return self.team_seasons[year].get_data()

    def get_data_columns(self, year):
        return self.team_seasons[year].get_data_columns()


if __name__ == "__main__":
    for year in [2021]:
        year_reg_season = regularseasonresults_df[regularseasonresults_df["Season"] == year]
        teams_conf_season = teamsconf_df[teamsconf_df["Season"] == year]
        teams_coach_season = teamscoach_df[teamscoach_df["Season"] == year]
        
        
        ts = get_team_seasons(year, year_reg_season, SEEDS_DF, teams_conf_season, teams_coach_season)
        for k,v in ts.items():
            print(year, k, v.get_data())