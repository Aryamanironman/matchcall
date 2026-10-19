import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')
from app.commentary_gpt2 import event_line
import json

teams = {"home": "Argentina", "away": "France"}

# Test a few events
features_pass = {
    "minute": 15,
    "ball_x_frac": 0.5,
    "goal_zone_time": 0.0,
    "has_ball": 1.0,
    "n_players": 4.0,
    "ball_speed": 0.2,
    "motion": 3.0,
    "phase": "midfield",
    "scoreline": "0-0",
    "_pos_i": 0
}
features_shot = {
    "minute": 22,
    "ball_x_frac": 0.8,
    "goal_zone_time": 0.1,
    "has_ball": 0.0,
    "n_players": 3.0,
    "ball_speed": 0.8,
    "motion": 5.0,
    "phase": "attacking",
    "scoreline": "0-0",
    "_pos_i": 0
}
features_goal = {
    "minute": 35,
    "ball_x_frac": 0.9,
    "goal_zone_time": 0.2,
    "has_ball": 0.0,
    "n_players": 2.0,
    "ball_speed": 0.9,
    "motion": 4.0,
    "phase": "goal",
    "scoreline": "0-0",
    "_pos_i": 0
}
features_tackle = {
    "minute": 40,
    "ball_x_frac": 0.3,
    "goal_zone_time": 0.0,
    "has_ball": 0.0,
    "n_players": 5.0,
    "ball_speed": 0.1,
    "motion": 6.0,
    "phase": "defensive",
    "scoreline": "0-0",
    "_pos_i": 0
}
features_pass2 = {
    "minute": 50,
    "ball_x_frac": 0.4,
    "goal_zone_time": 0.0,
    "has_ball": 1.0,
    "n_players": 4.0,
    "ball_speed": 0.3,
    "motion": 2.0,
    "phase": "build-up",
    "scoreline": "1-0",
    "_pos_i": 0
}

for label, feats in [("pass", features_pass), ("shot", features_shot), ("goal", features_goal), ("tackle", features_tackle), ("pass", features_pass2)]:
    line, excitement = event_line(label, feats, teams)
    print(f"{label.upper()}: {line} (excitement: {excitement})")