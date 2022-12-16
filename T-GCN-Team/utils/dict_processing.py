import numpy as np

def dict_processing(team_2_player: dict, num_t: int, num_p: int):
    assert len(team_2_player) == num_p
    team_to_player = dict.fromkeys(range(num_t))
    for player in team_2_player:
        if team_2_player[player] is not None:
            if team_to_player[team_2_player[player]] is None:
                team_to_player[team_2_player[player]] = [player]
            else:
                team_to_player[team_2_player[player]].append(player)
    return team_to_player
