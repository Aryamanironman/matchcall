import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')
from app.commentary_gpt2 import event_line, Gpt2Announcer
import json

# Dummy teams and features
teams = {"home": "Argentina", "away": "France"}
features = {
    "minute": 10,
    "ball_x_frac": 0.6,
    "goal_zone_time": 0.0,
    "has_ball": 1.0,
    "n_players": 5.0,
    "ball_speed": 0.0,
    "motion": 5.0,
    "phase": "midfield",
    "scoreline": "0-0",
    "_pos_i": 0
}
label = "pass"
try:
    line, excitement = event_line(label, features, teams)
    print(f"Line: {line}")
    print(f"Excitement: {excitement}")
except Exception as e:
    print(f"Error: {e}")

# Also test the announcer class
announcer = Gpt2Announcer(teams)
try:
    line2, exc2 = announcer.event_line(label, features)
    print(f"Announcer line: {line2}")
    print(f"Announcer excitement: {exc2}")
except Exception as e:
    print(f"Announcer error: {e}")