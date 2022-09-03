from multiprocessing.spawn import prepare
import pandas as pd
import numpy as np
import requests


def importData():
    api_path = "https://fantasy.premierleague.com/api/bootstrap-static/"

    data_raw = requests.get(api_path)
    data_json = data_raw.json()

    elements = pd.DataFrame(data_json["elements"])
    element_types = pd.DataFrame(data_json["element_types"])
    teams = pd.DataFrame(data_json["teams"])
    events = pd.DataFrame(data_json["events"])

    return elements, element_types, teams, events


def preProc(importedData):
    elements = importedData[0]
    element_types = importedData[1]
    teams = importedData[2]
    events = importedData[3]

    # get gameweek info:
    gameweek_core = events[
        [
            "id",
            "name",
            "deadline_time",
            "average_entry_score",
            "highest_score",
            "is_previous",
            "is_current",
            "is_next",
            "most_selected",
            "most_transferred_in",
            "top_element",
            "top_element_info",
            "most_captained",
            "most_vice_captained",
        ]
    ].copy()

    # get top players in current and prior gameweek:
    top_players = pd.json_normalize(
        gameweek_core.loc[
            (gameweek_core["is_current"] == True)
            | (gameweek_core["is_previous"] == True)
        ]["top_element_info"]
    )

    top_players = pd.merge(
        top_players,
        elements[["id", "second_name", "team"]],
        how="left",
        left_on="id",
        right_on="id",
    )

    top_players = pd.merge(
        top_players,
        teams[["id", "name"]],
        how="left",
        left_on="team",
        right_on="id",
        suffixes=("", "_del"),
    )

    top_players = top_players[["id", "points", "second_name", "name"]].copy()

    # get list of players and core info:
    player_core = elements[
        [
            "id",
            "first_name",
            "second_name",
            "element_type",
            "chance_of_playing_next_round",
            "chance_of_playing_this_round",
            "form",
            "now_cost",
            "points_per_game",
            "selected_by_percent",
            "team",
            "team_code",
            "total_points",
            "transfers_in_event",
            "transfers_out_event",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "influence",
            "creativity",
            "threat",
            "corners_and_indirect_freekicks_order",
            "direct_freekicks_order",
            "penalties_order",
        ]
    ].copy()

    # get list of teams and core info:
    teams_core = teams[
        [
            "id",
            "name",
            "short_name",
            "strength_overall_home",
            "strength_overall_away",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ]
    ].copy()

    # add basic calcs:
    player_core = pd.merge(
        player_core,
        element_types[["id", "singular_name_short"]],
        how="left",
        left_on="element_type",
        right_on="id",
    )

    player_core = pd.merge(
        player_core,
        teams[["id", "name"]],
        how="left",
        left_on="team",
        right_on="id",
    )

    # points per minute:
    player_core["pointsPerMin"] = (
        player_core["total_points"] / player_core["minutes"]
    ).fillna(0)

    # goals scored per minute:
    player_core["goalsPerMin"] = (
        player_core["goals_scored"] / player_core["minutes"]
    ).fillna(0)

    # assists per minute:
    player_core["assistsPerMin"] = (
        player_core["assists"] / player_core["minutes"]
    ).fillna(0)

    # clean sheets per minute:
    player_core["cleanSheetsPerMin"] = (
        player_core["clean_sheets"] / player_core["minutes"]
    ).fillna(0)

    # goals conceded per minute:
    player_core["goalsConcededPerMin"] = (
        player_core["goals_conceded"] / player_core["minutes"]
    ).fillna(0)

    # penalties saved per minute:
    player_core["penSavedPerMin"] = (
        player_core["penalties_saved"] / player_core["minutes"]
    ).fillna(0)

    # penalties missed per minute:
    player_core["penMissPerMin"] = (
        player_core["penalties_missed"] / player_core["minutes"]
    ).fillna(0)

    # yellow cards per minute:
    player_core["yelCardsPerMin"] = (
        player_core["yellow_cards"] / player_core["minutes"]
    ).fillna(0)

    # red cards per minute:
    player_core["redCardsPerMin"] = (
        player_core["red_cards"] / player_core["minutes"]
    ).fillna(0)

    # saves per minute:
    player_core["savesPerMin"] = (player_core["saves"] / player_core["minutes"]).fillna(
        0
    )

    # bonus points per minute:
    player_core["bonusPerMin"] = (player_core["bonus"] / player_core["minutes"]).fillna(
        0
    )

    player_core["gkCombinedIndex"] = np.where(
        player_core["singular_name_short"] == "GKP",
        player_core["pointsPerMin"]
        + (player_core["cleanSheetsPerMin"] * 4)
        - player_core["goalsConcededPerMin"]
        + (player_core["penSavedPerMin"] * 5)
        + player_core["savesPerMin"]
        + (player_core["bonusPerMin"] * 2),
        0,
    )

    player_core["defCombinedIndex"] = np.where(
        player_core["singular_name_short"] == "DEF",
        player_core["pointsPerMin"]
        + (player_core["cleanSheetsPerMin"] * 4)
        + (player_core["goalsPerMin"] * 6)
        + (player_core["assistsPerMin"] * 3)
        - player_core["goalsConcededPerMin"]
        - player_core["redCardsPerMin"]
        - (player_core["redCardsPerMin"] * 3)
        + (player_core["bonusPerMin"] * 2),
        0,
    )

    player_core["midCombinedIndex"] = np.where(
        player_core["singular_name_short"] == "MID",
        player_core["pointsPerMin"]
        + (player_core["goalsPerMin"] * 5)
        + (player_core["assistsPerMin"] * 3)
        - player_core["redCardsPerMin"]
        - (player_core["redCardsPerMin"] * 3)
        + (player_core["bonusPerMin"] * 2)
        - (player_core["penMissPerMin"] * 2),
        0,
    )

    player_core["fwdCombinedIndex"] = np.where(
        player_core["singular_name_short"] == "FWD",
        player_core["pointsPerMin"]
        + (player_core["goalsPerMin"] * 4)
        + (player_core["assistsPerMin"] * 3)
        - player_core["redCardsPerMin"]
        - (player_core["redCardsPerMin"] * 3)
        + (player_core["bonusPerMin"] * 2)
        - (player_core["penMissPerMin"] * 2),
        0,
    )

    top_gk = (
        player_core.loc[
            (player_core["singular_name_short"] == "GKP")
            & (player_core["minutes"] > 90)
        ]
        .copy()
        .sort_values(["gkCombinedIndex"], ascending=False)[
            [
                "first_name",
                "second_name",
                "name",
                "form",
                "selected_by_percent",
                "total_points",
                "minutes",
                "influence",
                "creativity",
                "threat",
                "gkCombinedIndex",
            ]
        ]
        .copy()
    )

    top_def = (
        player_core.loc[
            (player_core["singular_name_short"] == "DEF")
            & (player_core["minutes"] > 90)
        ]
        .copy()
        .sort_values(["defCombinedIndex"], ascending=False)[
            [
                "first_name",
                "second_name",
                "name",
                "form",
                "selected_by_percent",
                "total_points",
                "minutes",
                "influence",
                "creativity",
                "threat",
                "defCombinedIndex",
            ]
        ]
        .copy()
    )

    top_mid = (
        player_core.loc[
            (player_core["singular_name_short"] == "MID")
            & (player_core["minutes"] > 90)
        ]
        .copy()
        .sort_values(["midCombinedIndex"], ascending=False)[
            [
                "first_name",
                "second_name",
                "name",
                "form",
                "selected_by_percent",
                "total_points",
                "minutes",
                "influence",
                "creativity",
                "threat",
                "midCombinedIndex",
            ]
        ]
        .copy()
    )

    top_fwd = (
        player_core.loc[
            (player_core["singular_name_short"] == "FWD")
            & (player_core["minutes"] > 90)
        ]
        .copy()
        .sort_values(["fwdCombinedIndex"], ascending=False)[
            [
                "first_name",
                "second_name",
                "name",
                "form",
                "selected_by_percent",
                "total_points",
                "minutes",
                "influence",
                "creativity",
                "threat",
                "fwdCombinedIndex",
            ]
        ]
        .copy()
    )

    pass


if __name__ == "__main__":
    imported_data = importData()
    out = preProc(imported_data)
