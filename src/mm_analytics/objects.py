from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Union
import json
import re

import numpy as np
import pandas as pd

from mm_analytics.utilities import get_historical_similarity, DATA_ROOT, NpEncoder, ROUND_DAYS

# Teams Data
TEAM_CONF_DF = pd.read_csv(f'{DATA_ROOT}/MTeamConferences.csv')
TEAM_DF = pd.read_csv(f'{DATA_ROOT}/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
TEAM_NAMES = { int(k):v for k,v in TEAM_DF.set_index('TeamID')['TeamName'].to_dict().items()}
TEAM_COACH_DF = pd.read_csv(f'{DATA_ROOT}/MTeamCoaches.csv')
ORDINALS_DF = pd.read_csv(f"{DATA_ROOT}/MMasseyOrdinals.csv")
SEASONS_DF = pd.read_csv(f'{DATA_ROOT}/MSeasons.csv')
# Load v from m/d/yyyy to date
SEASON_DAY_ZEROES = {k:datetime.strptime(v, '%m/%d/%Y').date() for k, v in SEASONS_DF.set_index('Season')['DayZero'].to_dict().items() }
# Regular Season Data
REGULAR_SZN_DF = pd.read_csv(f'{DATA_ROOT}/MRegularSeasonDetailedResults.csv')

# Conference Tourney Data
conferencetourney_df = pd.read_csv(f'{DATA_ROOT}/MConferenceTourneyGames.csv')

# NCAA Tourney Data
SEEDS_DF = pd.read_csv(f'{DATA_ROOT}/MNCAATourneySeeds.csv')
TOURNEY_RESULTS_DF = pd.read_csv(f'{DATA_ROOT}/MNCAATourneyDetailedResults.csv')

def to_pct(val1, val2) -> Union[float, None]:
    """Calculate the percentage of two values
    :param val1: int - Value 1
    :param val2: int - Value 2
    :return: float - Percentage
    """
    return val1 / val2 if val2 != 0 else None

class TeamGame:
    def __init__(self, opp_id: int, opp_name:str, team_score: int, opp_score: int,
                 team_loc: str, date_int:int = None, date_str:str = None, conf: bool = None,
                 poss: int = None) -> None:
        self.opponent_id = opp_id
        self.opponent_name = opp_name
        self.team_score = team_score
        self.opp_score = opp_score
        self.team_loc = team_loc
        self.date_int = date_int
        self.date_str = date_str
        self.conf = conf
        self.poss = poss
    
    def get_efficiency(self) -> Tuple[float, float]:
        """Get Offensive and Defensive Efficiency for the game
        """
        return (100 * (self.team_score / self.poss), 100 * (self.opp_score / self.poss))
    
    def is_win(self) -> bool:
        return self.team_score > self.opp_score
    
    def to_json(self) -> dict:
        return {
            "opp_id": self.opponent_id,
            "team_score": self.team_score,
            "opp_score": self.opp_score,
            "team_loc": self.team_loc,
            "date_int": self.date_int,
            "date_str": self.date_str,
            "conf": self.conf
        }

class TeamSeasonOrdinals:
    """Store the ordinal data for a team in a season
    """
    def __init__(self) -> None:
        self._data = {}
    
    def valid_system(self, system_id:str) -> bool:
        return system_id in self._data.keys()

    @staticmethod
    def valid_statistic(stat:str) -> bool:
        if stat not in ["average", "last"]:
            raise ValueError(f"Statistic {stat} not in list of valid statistics")
    
    def add_system(self, system_id:str, system_data:pd.DataFrame):
        """Use the MasseyOrdinals data to fill the system data for the team
        :param system_data { pd.DataFrame }:
            DataFrame containing that system's data for the season
        """
        if len(system_data) == 0:
            self._data[system_id] = {"average": None, "last": None}
            return
        values, last_day, last_val = [], 0, 0
        for _, daynum, rank in system_data[["RankingDayNum","OrdinalRank"]].itertuples():
            values.append(rank)
            if daynum > last_day:
                last_val = rank
                last_day = daynum
        self._data[system_id] = {"average": np.average(values), "last": last_val}

    def list_systems(self):
        return list(self._data.keys())

    def get_system_data(self, system_id:str, stat:str = None):
        if stat is not None:
            self.valid_statistic(stat)
            return self._data[system_id][stat]
        return self._data[system_id]

    def get_data(self, statistics:List[str] = None):
        if statistics is not None:
            for stat in statistics:
                self.valid_statistic(stat)
            data = []
            for system in self.list_systems():
                for stat in statistics:
                    data.append(self._data[system][stat])
            return data
        return [self._data[system][stat] for system in self.list_systems() for stat in ["average", "last"]]

    def get_data_columns(self, statistics:List[str] = None):
        if statistics is not None:
            for stat in statistics:
                self.valid_statistic(stat)
            return [f"{system}_{stat}" for system in self.list_systems() for stat in statistics]
        return [f"{system}_{stat}" for system in self.list_systems() for stat in ["average", "last"]]
    def to_json(self, statistics:str = None):
        if statistics is not None:
            self.valid_statistic(statistics)
            return {system: self._data[system][statistics] for system in self.list_systems()}
        return {system: self._data[system] for system in self.list_systems()}

def get_year_system(year):
    return "NET" if year > 2018 else "RPI"

def get_season_ordinals(season_ordinals_df, systems:List[str] = None) -> Dict[int, TeamSeasonOrdinals]:
    """Get the ordinal data for each team in a season
    :param season_ordinals_df: Pandas DataFrame - Ordinals Data for a Season
    :param systems: List[str] - List of systems to include in the data
    """
    results = {}
    for tid, tdf in season_ordinals_df.groupby("TeamID"):
        team_season_ordinals = TeamSeasonOrdinals()
        for sys in systems:
            team_season_ordinals.add_system(sys, tdf[tdf["SystemName"] == sys])
        results[tid] = team_season_ordinals
    return results


class TeamSeason:
    data_col_attrs = {
        "WinPct": "win_pct",
        "SOS": "sos",
        "SOV": "sov",
        "Seed": "tourney_seed",
        "ExitRound": "tourney_exit_round"
    }
    def __init__(self, id, year:int, name:str, tournament_seed:int, regular_season_df:pd.DataFrame = None,
                 season_conf_df:pd.DataFrame = None, team_coach:str = None, post_season_df:pd.DataFrame = None):
        self.id, self.year, self.name = id, year, name
        self.games: List[TeamGame] = []
        self.wins: List[TeamGame] = []
        self.losses: List[TeamGame] = []
        self.win_pct, self.opp_win_pct = None, None
        self.sos, self.sov = None, None
        self.tourney_seed = tournament_seed
        self.coach = team_coach
        self.conf = season_conf_df[(season_conf_df['TeamID'] == self.id)]['ConfAbbrev'].values[0]

        self.stat_values = {
            "Points": [], "Poss": [], "OE": [], "DE": [], "NE": [], "FGM": [], "FGA": [], "FGM3": [], "FGA3": [], "FTM": [], "FTA": [], "OR": [], "DR": [], "Ast": [], "TO": [], "Stl": [], "Blk": [], "Fouls": [],
            "FG%": [], "FG3%": [], "EFG%": [], "FT%": [],
            "OppPoints": [], "OppFGM": [], "OppFGA": [], "OppFGM3": [], "OppFGA3": [], "OppFTM": [], "OppFTA": [], "OppOR": [], "OppDR": [], "OppAst": [], "OppTO": [], "OppStl": [], "OppBlk": [], "OppFouls": [],
            "OppFG%": [], "OppFG3%": [], "OppEFG%": [], "OppFT%": []
        }

        # Rankings that can be filled in after the season stats are calculated
        self.stat_rankings = {}
        self.adjusted_stats = {}
        # Fill in the calculate_season_stats method
        self.means, self.stdev = {}, {}

        # Fill the regular season stats
        conf_opponents = season_conf_df[season_conf_df['ConfAbbrev'] == self.conf]['TeamID'].unique().tolist()
        for row in regular_season_df.itertuples():
            self.fill_game(row, conf_opponents, TEAM_NAMES, SEASON_DAY_ZEROES[year])
        self.wins = [g for g in self.games if g.is_win()]
        self.losses = [g for g in self.games if not g.is_win()]
        # Generate means, averages, and stdevs for each stat
        self.calculate_season_stats()

        # Fill Ordinal Info in the calculate_post_season_stats method
        self.ordinal_data: TeamSeasonOrdinals = None
        self.quad_wins      = {1:[], 2:[], 3:[], 4:[]}
        self.quad_losses    = {1:[], 2:[], 3:[], 4:[]} 
        # Similarity Data
        self.similar_teams = []
        # Tournament Stats / Results
        self.tourney_games: List[TeamGame] = []
        self.tourney_exit_round: str = None
        if post_season_df is not None:
            for row in post_season_df.itertuples():
                self.fill_postseason_game(row)

            for game in self.tourney_games:
                if not game.is_win():
                    self.tourney_exit_round = ROUND_DAYS[game.date_int]
                    break
            if len(self.tourney_games) > 0 and self.tourney_exit_round is None:
                self.tourney_exit_round = "Champion"

    def calculate_season_stats(self):
        # Calculate Win Pct
        self.win_pct = len(self.wins) / (len(self.wins) + len(self.losses))
        # Calculate distributions for status
        for k, v in self.stat_values.items():
            stat_vals        = [val for (_, val) in v]
            self.means[k]    = np.mean([x for x in stat_vals if x is not None])
            self.stdev[k]    = np.std([x for x in stat_vals if x is not None])

    def calculate_post_season_adjusted_stats(self, league_season_data: Dict[int, 'TeamSeason'], season_averages: Dict[str, float]):
        """
        Calculate the adjusted stats by using the season averages
        """
        avg_sos, avg_oe, avg_de = season_averages["SOS"], season_averages["OE"], season_averages["DE"]
        self.stat_values["AdjOE"], self.stat_values["AdjDE"], self.stat_values["AdjNE"] = [], [], []
        for game in self.games:
            game_oe, game_de = game.get_efficiency()
            opp_season = league_season_data[game.opponent_id]
            # Adjusted Offensive Efficiency = Raw OE * (Strength of OppDefense)
            # Strength of OppDefense = (Opponent DE / Average DE) * (Opponent SOS / Average SOS)
            adj_oe = (game_oe) * (opp_season.means["DE"] / avg_de) * (opp_season.sos / avg_sos)
            self.stat_values["AdjOE"].append(adj_oe)
            # Adjusted Defensive Efficiency = Raw DE * (Strength of OppOffense)
            # Strength of OppOffense = (Opponent OE / Average OE) * (Opponent SOS / Average SOS)
            adj_de = (game_de) * (opp_season.means["OE"] / avg_oe) * (opp_season.sos / avg_sos)
            self.stat_values["AdjDE"].append(adj_de)
            # Adjusted Net Efficiency = Adjusted OE - Adjusted DE
            self.stat_values["AdjNE"].append(adj_oe - adj_de)
        self.means["AdjOE"] = np.mean(self.stat_values["AdjOE"])
        self.stdev["AdjOE"] = np.std(self.stat_values["AdjOE"])
        self.means["AdjDE"] = np.mean(self.stat_values["AdjDE"])
        self.stdev["AdjDE"] = np.std(self.stat_values["AdjDE"])
        self.means["AdjNE"] = np.mean(self.stat_values["AdjNE"])
        self.stdev["AdjNE"] = np.std(self.stat_values["AdjNE"])

    def calculate_post_season_stats(self, league_season_data: Dict[int, 'TeamSeason'], season_ordinals: Dict[int, TeamSeasonOrdinals] = None):
        """ Calculate statistics that you need other teams info for 
        - statistical rankings for the season
        - strength of schedule, strength of victory
        Param: league_season_data: dict<id,TeamSeason> 
        """
        
        opp_wp = []
        opp_opp_wp = []
        for o in self.wins:
            opponent = league_season_data[o.opponent_id]
            opp_wp.append(opponent.win_pct)
            opp_games = opponent.wins + opponent.losses
            opp_opp_wp.append(np.average([league_season_data[oo.opponent_id].win_pct for oo in opp_games]))

        self.sov = np.average(opp_wp)
        
        for o in self.losses:
            opponent = league_season_data[o.opponent_id]
            opp_wp.append(opponent.win_pct)
            opp_games = opponent.wins + opponent.losses
            opp_opp_wp.append(np.average([league_season_data[oo.opponent_id].win_pct for oo in opp_games]))


        ow = np.average(opp_wp)
        oow = np.average(opp_opp_wp)
        self.sos = (2*ow + oow) / 3
        if season_ordinals is not None:
            # Set the RPI, NET, and other ordinal rankings
            self.ordinal_data = season_ordinals[self.id]
            # Calculate the teams record per quad
            for game in self.wins:
                game_quad = get_game_quad(game, season_ordinals, self.year)
                self.quad_wins[game_quad].append(game)
            for game in self.losses:
                game_quad = get_game_quad(game, season_ordinals, self.year)
                self.quad_losses[game_quad].append(game)
        return

    def get_data_columns(self):
        return list([k+"_mean" for k in self.means.keys()]) + \
            list([k+"_stdev" for k in self.stdev.keys()]) + \
            self.ordinal_data.get_data_columns(["last"]) + \
            ["Q1_WinPct", "Q2_WinPct", "Q3_WinPct", "Q4_WinPct"] + \
            list(self.data_col_attrs.keys())

    def get_data(self, columns:list = None):
        """Return the data for the team season
        :param columns: List[str] - List of columns to return
            accepted values: "WinPct", "SOS", "SOV", "Seed",
            "{system}_{stat}", "{stat}_mean", "{stat}_stdev"
        """
        qpct = lambda qw, ql: (len(qw) / (len(qw) + len(ql))) if len(qw) + len(ql) > 0 else None
        q_winpct = [qpct(self.quad_wins[q], self.quad_losses[q]) for q in range(1,5)]
        if columns is None:
            data = ( 
                list(self.means.values()) + 
                list(self.stdev.values()) + 
                self.ordinal_data.get_data(["last"]) +
                q_winpct,
                [getattr(self, self.data_col_attrs[k]) for k in self.data_col_attrs.keys()]
            )
            return np.array([round(v, 4) if v is not None else None for v in data])
        else:
            vals = []
            for c in columns:
                if c.endswith("_mean"):
                    vals.append(self.means[c[:-5]])
                elif c.endswith("_stdev"):
                    vals.append(self.stdev[c[:-6]])
                elif c in self.data_col_attrs.keys():
                    vals.append(getattr(self, self.data_col_attrs[c]))
                elif c in self.ordinal_data.get_data_columns():
                    sys, stat = c.split("_")
                    vals.append(self.ordinal_data.get_system_data(sys, stat))
                elif c in ["Q1_WinPct", "Q2_WinPct", "Q3_WinPct", "Q4_WinPct"]:
                    q = int(re.sub(r"\D", "", c))
                    vals.append(q_winpct[q-1])
            return np.array(vals)

    def fill_game(self, game_row:Tuple, conf_opponents:List[int], team_names:Dict[int, str], season_day_zero:date):
        """Take a dataframe row and fill in the stats for that game
        :param game_row: Tuple - Row of the Regular Season DataFrame
        """
        game_day = game_row[2]
        game_day_str = (season_day_zero + timedelta(days=game_day)).strftime("%m/%d")
        if(game_row[3] == self.id):
            # Team Win Stats
            TeamPoints, OppID, OppPoints = game_row[4], game_row[5], game_row[6]
            WLoc, _, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls = game_row[7:]
            team_loc = WLoc
        elif(game_row[5] == self.id):
            # Team Loss Stats
            OppID, OppPoints, TeamPoints = game_row[3], game_row[4], game_row[6]
            WLoc, _, OppFGM, OppFGA, OppFGM3, OppFGA3, OppFTM, OppFTA, OppOR, OppDR, OppAst, OppTO, OppStl, OppBlk, OppFouls, FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, Fouls = game_row[7:]
            team_loc = "H" if WLoc == "A" else "A" if WLoc == "H" else "N"
        else:
            print("Error: TeamID not in game row")
            exit(0)
        # ---- Computed Statistics ----
        team_poss = (FGA - OR) + TO + (.475 * FTA) # KenPom: https://kenpom.com/blog/national-efficiency/
        opp_poss = (OppFGA - OppOR) + OppTO + (.475 * OppFTA) # KenPom: https://kenpom.com/blog/national-efficiency/
        game_poss = (team_poss + opp_poss) / 2
        self.games.append(TeamGame(OppID, team_names[OppID], TeamPoints, OppPoints, team_loc, game_day, game_day_str, OppID in conf_opponents, game_poss))
        game_oe = 100 * (TeamPoints/game_poss)
        game_de = 100 * (OppPoints/game_poss)
        eff_fgpct = (FGM + 0.5 * FGM3) / FGA
        opp_eff_fgpct = (OppFGM + 0.5 * OppFGM3) / OppFGA

        self.stat_values["OE"].append((OppID, game_oe))
        self.stat_values["DE"].append((OppID, game_de))
        self.stat_values["NE"].append((OppID, (game_oe - game_de)))
        self.stat_values["Poss"].append((OppID, game_poss))
        self.stat_values["Points"].append((OppID, TeamPoints))
        self.stat_values["FGM"].append((OppID, FGM)), self.stat_values["FGA"].append((OppID, FGA)), self.stat_values["FGM3"].append((OppID, FGM3)), self.stat_values["FGA3"].append((OppID, FGA3)), self.stat_values["FTM"].append((OppID, FTM)), self.stat_values["FTA"].append((OppID, FTA)), self.stat_values["OR"].append((OppID, OR)), self.stat_values["DR"].append((OppID, DR)), self.stat_values["Ast"].append((OppID, Ast)), self.stat_values["TO"].append((OppID, TO)), self.stat_values["Stl"].append((OppID, Stl)), self.stat_values["Blk"].append((OppID, Blk)), self.stat_values["Fouls"].append((OppID, Fouls))
        self.stat_values["FG%"].append((OppID, to_pct(FGM,FGA))), self.stat_values["FG3%"].append((OppID, to_pct(FGM3,FGA3))), self.stat_values["FT%"].append((OppID, to_pct(FTM, FTA)))
        self.stat_values["EFG%"].append((OppID, eff_fgpct))
        self.stat_values["OppPoints"].append((OppID, OppPoints))
        self.stat_values["OppFGM"].append((OppID, OppFGM)), self.stat_values["OppFGA"].append((OppID, OppFGA)), self.stat_values["OppFGM3"].append((OppID, OppFGM3)), self.stat_values["OppFGA3"].append((OppID, OppFGA3)), self.stat_values["OppFTM"].append((OppID, OppFTM)), self.stat_values["OppFTA"].append((OppID, OppFTA)), self.stat_values["OppOR"].append((OppID, OppOR)), self.stat_values["OppDR"].append((OppID, OppDR)), self.stat_values["OppAst"].append((OppID, OppAst)), self.stat_values["OppTO"].append((OppID, OppTO)), self.stat_values["OppStl"].append((OppID, OppStl)), self.stat_values["OppBlk"].append((OppID, OppBlk)), self.stat_values["OppFouls"].append((OppID, OppFouls))
        self.stat_values["OppFG%"].append((OppID, to_pct(OppFGM,OppFGA))), self.stat_values["OppFG3%"].append((OppID, to_pct(OppFGM3, OppFGA3))), self.stat_values["OppFT%"].append((OppID, to_pct(OppFTM,OppFTA)))
        self.stat_values["OppEFG%"].append((OppID, opp_eff_fgpct))

    def fill_postseason_game(self, game_row:Tuple):
        """
        """
        game_day = game_row[2]
        game_day_str = ROUND_DAYS[game_day]
        if(game_row[3] == self.id):
            # Team Win Stats
            TeamPoints, OppID, OppPoints = game_row[4], game_row[5], game_row[6]
        elif (game_row[5] == self.id):
            # Team Loss Stats
            OppID, OppPoints, TeamPoints = game_row[3], game_row[4], game_row[6]
        else:
            print("Error: TeamID not in game row")
        t1_fga, t1_or, t1_to, t1_fta = game_row[10], game_row[15], game_row[18],  game_row[14]
        t2_fga, t2_or, t2_to, t2_fta = game_row[23], game_row[28], game_row[31],  game_row[27]
        game_poss = (((t1_fga - t1_or) + t1_to + (.475 * t1_fta)) + ((t2_fga - t2_or) + t2_to + (.475 * t2_fta))) / 2
        self.tourney_games.append(TeamGame(OppID, TEAM_NAMES[OppID], TeamPoints, OppPoints, "N", game_day, game_day_str, poss = game_poss))

    def to_web_json(self):
        """Return a JSON representation of the TeamSeason
        that can be used in the march madness web app
        """
        tournament_data = {
            "seed": self.tourney_seed,
            "games": [g.to_json() for g in self.tourney_games],
            "exit_round": self.tourney_exit_round
        } if (len(self.tourney_games) > 0 or self.tourney_seed is not None) else None
        return {
            "id": self.id,
            "year": self.year,
            "name": self.name,
            "record": {
                "overall": (len(self.wins), len(self.losses)),
                "conf": (len([g for g in self.wins if g.conf]), len([g for g in self.losses if g.conf])),
                "home": (len([g for g in self.wins if g.team_loc == "H"]), len([g for g in self.losses if g.team_loc == "H"])),
                "road": (len([g for g in self.wins if g.team_loc in {"A", "N"}]), len([g for g in self.losses if g.team_loc in {"A", "N"}])),
                "quad_1": (len(self.quad_wins[1]), len(self.quad_losses[1])),
                "quad_2": (len(self.quad_wins[2]), len(self.quad_losses[2])),
                "quad_3": (len(self.quad_wins[3]), len(self.quad_losses[3])),
                "quad_4": (len(self.quad_wins[4]), len(self.quad_losses[4]))
            },
            "quad_wins": {q: [g.to_json() for g in games] for q,games in self.quad_wins.items()},
            "quad_losses": {q: [g.to_json() for g in games] for q,games in self.quad_losses.items()},
            "conf": self.conf,
            "stat_rankings": self.stat_rankings,
            "stats": {
                **{k:round(v, 3) for k,v in self.means.items() }
            },
            "coach": self.coach,
            "win_pct": round(self.win_pct, 3),
            "sos": round(self.sos, 3),
            "sov": round(self.sov, 3),
            "ordinal_data": self.ordinal_data.to_json(statistics="last"),
            "tournament": tournament_data,
            "similar_teams": [{
                "id": tid, "year": ty, "avg": avgs, "res": rs, "st": ss, "er": er
            } for (tid, ty, avgs, rs, ss, er) in self.similar_teams]
        }

QUAD_THRESHOLDS = {
    "H": [30, 75, 160],
    "N": [50, 100, 200],
    "A": [75, 135, 240]
}

def get_game_quad(game: TeamGame, season_ordinals: Dict[int, TeamSeasonOrdinals], season:int) -> int:
    """Evaluate the quadrant of a game based on the NET or RPI system
    Quadrant 1: Home 1-30, Neutral 1-50, Away 1-75
    Quadrant 2: Home 31-75, Neutral 51-100, Away 76-135
    Quadrant 3: Home 76-160, Neutral 101-200, Away 135-240
    Quadrant 4: Home 161-353, Neutral 201-353, Away 241-353
    """
    quad_ordinal = get_year_system(season)
    opp_ordinals = season_ordinals[game.opponent_id]
    if not opp_ordinals.valid_system(quad_ordinal):
        raise ValueError(f"System {quad_ordinal} not in opponent's ordinal data")
    opp_rank = opp_ordinals._data[quad_ordinal]["last"]
    if opp_rank is None:
        return 4
    game_thresholds = QUAD_THRESHOLDS[game.team_loc]
    if opp_rank <= game_thresholds[0]:
        return 1
    elif opp_rank <= game_thresholds[1]:
        return 2
    elif opp_rank <= game_thresholds[2]:
        return 3
    else:
        return 4

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

def get_team_seasons_and_rankings(year, regular_season_df, seeds_df: pd.DataFrame = None, teams_conf_df = None,
                teams_coach_df = None, season_ordinals:Dict[int, TeamSeasonOrdinals] = None,
                tourney_results_df = None) -> Tuple[Dict[int, TeamSeason], Dict[str, Tuple[int, float]]]:
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
    team_ids_in_season = set(regular_season_df['WTeamID'].values).union(set(regular_season_df['LTeamID'].values))
    seeds: Dict[int, int] = get_season_seeds(year, seeds_df)
    season_conf_df = teams_conf_df[teams_conf_df['Season'] == year]
    
    for team_id in team_ids_in_season:
        team_seed = seeds.get(team_id)
        team_games = regular_season_df[(regular_season_df['WTeamID'] == team_id) | (regular_season_df['LTeamID'] == team_id)]
        team_coach = teams_coach_df[(teams_coach_df['TeamID'] == team_id) & (teams_coach_df['Season'] == year)]['CoachName'].values[0]
        team_name = TEAM_DF[TEAM_DF['TeamID'] == team_id]['TeamName'].values[0]
        team_tourney = tourney_results_df[(tourney_results_df['WTeamID'] == team_id) | (tourney_results_df['LTeamID'] == team_id)]
        print(f"Team: {team_id}, Seed: {team_seed}, Year: {year}, Tourney Games: {len(team_tourney)}")
        team_seasons[team_id] = TeamSeason(team_id, year, team_name, team_seed, team_games, season_conf_df, team_coach, team_tourney)
    
    
    # Fill in the post season stats
    for team_season in team_seasons.values():
        team_season.calculate_post_season_stats(team_seasons, season_ordinals)
    (_, season_avgs) = calculate_season_rankings_and_averages(team_seasons, rank = False)
    # Fill in the adjusted stats and rankings
    for team_season in team_seasons.values():
        team_season.calculate_post_season_adjusted_stats(team_seasons, season_avgs)
    (season_ranks, season_avgs) = calculate_season_rankings_and_averages(team_seasons)
    
    return team_seasons, season_ranks

STATS_WITH_INVERTED_RANKS = ["DE", "AdjDE", "TO", "Fouls"]
def calculate_season_rankings_and_averages(team_seasons: Dict[int, TeamSeason],
        add_to_season:bool = True, rank:bool = True) -> Tuple[Dict[str, Tuple[int, float]], Dict[str, float]]:
    """Calculate the rankings in each statistic for each team in the season
    from their season averages
    :param team_seasons: Dict<int, TeamSeason> - Dictionary of TeamSeasons
    :param season_averages: Dict<int, float> - Dictionary of Season Averages
    :param add_to_season: bool - Add the rankings to the TeamSeason object instances
    :param rank: bool - Rank the statistics
    :return: Tuple of Dict<str, List<Tuple<int, float>>, Dict<str, float>>
        Index 0: Dict<str, List<Tuple[int, float]>> 
        Mapping of Statistic to ordered list of TeamID and Statistic Value
        ex: { "OE": [(1242, 91.2), (1234, 87.1), ...], ...}
        Index 1: Dict<str, float>
        Mapping of Statistic to average value
        ex: { "OE": 89.2, "DE": 78.1, ...}
    """
    rankings, averages = {}, {}
    stats_to_rank = list(team_seasons.values())[0].means.keys()
    for stat in stats_to_rank:
        stat_vals = {}
        for team_id, team_season in team_seasons.items():
            stat_vals[team_id] = team_season.means[stat]
        averages[stat] = np.average(list(stat_vals.values()))
        if rank:
            sorted_vals = sorted(stat_vals.items(), key=lambda x: x[1], reverse=True)
            rankings[stat] = []
            for i, (team_id, team_val) in enumerate(sorted_vals):
                rankings[stat].append((team_id, team_val))
                if add_to_season:
                    team_seasons[team_id].stat_rankings[stat] = i
    # Rank the sos and sov
    for stat in ["SOS", "SOV"]:
        stat_vals = {}
        for team_id, team_season in team_seasons.items():
            stat_vals[team_id] = getattr(team_season, stat.lower())
        averages[stat] = np.average(list(stat_vals.values()))
        if rank:
            sorted_vals = sorted(stat_vals.items(), key=lambda x: x[1], reverse=True)
            rankings[stat] = []
            for i, (team_id, team_val) in enumerate(sorted_vals):
                rankings[stat].append((team_id, team_val))
                if add_to_season:
                    team_seasons[team_id].stat_rankings[stat] = i
    return rankings, averages

def team_seasons_to_df(team_seasons:Dict[Tuple[int, int], TeamSeason], columns:List[str],
                       add_season_ordinal:bool = True) -> pd.DataFrame:
    """Convert a dictionary of TeamSeasons to a DataFrame
    :param team_seasons: Dict<int, TeamSeason> - Dictionary of TeamSeasons
    :param columns: List<str> - List of columns to include in the DataFrame
    :param add_season_ordinal: bool - Add the season ordinal to the DataFrame
    :return: pd.DataFrame - DataFrame of TeamSeasons
    """
    # Configure columns to use
    result_columns = columns + ["NET_last"] if add_season_ordinal else columns
    team_seasons_df = pd.DataFrame(columns = ["TeamID", "Season"] + result_columns)
    for (team_id, _), team in team_seasons.items():
        season_columns = columns + [f"{get_year_system(team.year)}_last"] if add_season_ordinal else columns
        team_row = np.array([team_id, team.year] + team.get_data(columns=season_columns).tolist())
        team_seasons_df = pd.concat([team_seasons_df, pd.DataFrame([team_row], columns = team_seasons_df.columns)], ignore_index=True)
    return team_seasons_df

if __name__ == "__main__":

    for year in [2022]:
        year_reg_season     = REGULAR_SZN_DF[REGULAR_SZN_DF["Season"] == year]
        teams_conf_season   = TEAM_CONF_DF[TEAM_CONF_DF["Season"] == year]
        teams_coach_season  = TEAM_COACH_DF[TEAM_COACH_DF["Season"] == year]
        year_tourney        = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"] == year]

        so = get_season_ordinals(ORDINALS_DF[ORDINALS_DF["Season"] == year], ["NET"] if year >= 2019 else ["RPI"])
        (ts, sr) = get_team_seasons_and_rankings(year, year_reg_season, SEEDS_DF, teams_conf_season, teams_coach_season, so, year_tourney)
        
        ts_df = team_seasons_to_df(ts)
        ts_df.to_csv(f"TeamSeasons_{year}.csv", index=False)
