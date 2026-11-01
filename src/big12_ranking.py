import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = "data/Big12_Match_Data.xlsx"
SHEET = "Match Data"
OUTPUT_DIR = "results"
GAMES_PER_TEAM = 9          # every Big 12 team plays 9 conference games

# ---------- 1. load and check the data ----------
all_games = pd.read_excel(DATA_FILE, sheet_name=SHEET)
all_games = all_games.dropna(subset=["Team_A", "Team_B", "Score_A", "Score_B"])

assert (all_games["Team_A"] != all_games["Team_B"]).all(), "A game lists the same team twice"
assert (all_games[["Score_A", "Score_B"]] >= 0).all().all(), "Negative score found"
dup = all_games.duplicated(subset=["Date", "Team_A", "Team_B", "Score_A", "Score_B"])
assert not dup.any(), f"Duplicate game rows: {list(all_games.index[dup])}"

df = all_games[all_games["Include_In_Model"] == 1].reset_index(drop=True)     # games used to fit
test = all_games[all_games["Include_In_Model"] == 0].reset_index(drop=True)   # title game, held out

teams = sorted(set(df["Team_A"]) | set(df["Team_B"]))
team_index = {team: i for i, team in enumerate(teams)}

counts = pd.concat([df["Team_A"], df["Team_B"]]).value_counts()
for team in teams:
    if counts[team] != GAMES_PER_TEAM:
        print(f"WARNING: {team} has {counts[team]} games in the fit (expected {GAMES_PER_TEAM})")

# all teams must be linked through the games, otherwise ratings cannot be compared
links = {t: set() for t in teams}
for a, b_ in zip(df["Team_A"], df["Team_B"]):
    links[a].add(b_)
    links[b_].add(a)
seen, todo = set(), [teams[0]]
while todo:
    t = todo.pop()
    if t not in seen:
        seen.add(t)
        todo.extend(links[t] - seen)
connected = len(seen) == len(teams)
if not connected:
    print("WARNING: the games do not connect all teams; ratings are not comparable across groups")

print(f"{len(all_games)} games read: {len(df)} used to fit, {len(test)} held out; {len(teams)} teams")

# ---------- 2. build the system A r = b ----------
A = np.zeros((len(df), len(teams)))
b = (df["Score_A"] - df["Score_B"]).to_numpy(dtype=float)

for i, (_, game) in enumerate(df.iterrows()):
    A[i, team_index[game["Team_A"]]] = 1
    A[i, team_index[game["Team_B"]]] = -1

A_games = A.copy()                       # keep the game rows for the error checks
A = np.vstack([A, np.ones(len(teams))])  # extra row: ratings add up to 0
b_full = np.append(b, 0)

# ---------- 3. solve by least squares ----------
ratings = np.linalg.lstsq(A, b_full, rcond=None)[0]
ranking = sorted(zip(teams, ratings), key=lambda item: item[1], reverse=True)

print("\nBig 12 Football Rankings")
print("------------------------")
for rank, (team, rating) in enumerate(ranking, start=1):
    print(f"{rank:2}. {team:<32} Rating: {rating:7.2f}")

# ---------- 4. errors ----------
predicted = A_games @ ratings           # model's predicted score gap for each game
residual = predicted - b                # predicted minus real

mae = np.mean(np.abs(residual))
rmse = np.sqrt(np.mean(residual ** 2))
correct = int(np.sum((predicted > 0) == (b > 0)))
rank_without = np.linalg.matrix_rank(A_games)
rank_with = np.linalg.matrix_rank(A)

print("\nModel errors (games used to fit)")
print("--------------------------------")
print(f"Average miss (MAE):        {mae:.2f} points")
print(f"Root mean square error:    {rmse:.2f} points")
print(f"Winner picked correctly:   {correct} of {len(df)} ({100 * correct / len(df):.1f}%)  [in-sample, optimistic]")
print(f"Largest single miss:       {np.max(np.abs(residual)):.1f} points")
print(f"Sum of ratings:            {ratings.sum():.6f}  (should be 0)")
print(f"Matrix rank:               {rank_without} without the sum row, {rank_with} with it ({len(teams)} unknowns)")

# ---------- 5. title game test (not used to fit) ----------
print("\nHeld-out test: Big 12 Championship Game")
print("---------------------------------------")
test_lines = []
for _, g in test.iterrows():
    pred = ratings[team_index[g["Team_A"]]] - ratings[team_index[g["Team_B"]]]
    real = g["Score_A"] - g["Score_B"]
    line = (f"{g['Team_A']} {int(g['Score_A'])} - {int(g['Score_B'])} {g['Team_B']}: "
            f"model gap {pred:+.2f}, real gap {real:+.0f}, winner correct: {bool((pred > 0) == (real > 0))}")
    test_lines.append(line)
    print(line)

# ---------- 6. save the results ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
table = pd.DataFrame(ranking, columns=["Team", "Rating"])
table.insert(0, "Rank", range(1, len(table) + 1))
table["Rating"] = table["Rating"].round(2)
table.to_csv(f"{OUTPUT_DIR}/Team_Ratings.csv", index=False)

games_out = df[["Date", "Team_A", "Score_A", "Team_B", "Score_B"]].copy()
games_out["Real_Gap"] = b
games_out["Predicted_Gap"] = predicted.round(2)
games_out["Miss_Predicted_minus_Real"] = residual.round(2)
games_out.to_csv(f"{OUTPUT_DIR}/Game_Residuals.csv", index=False)

with open(f"{OUTPUT_DIR}/Model_Metrics.txt", "w", encoding="utf-8") as f:
    f.write("Ratings are model estimates, not official rankings.\n")
    f.write(f"games_used_to_fit: {len(df)}\nteams: {len(teams)}\n")
    f.write(f"mean_absolute_error_points: {mae:.4f}\nroot_mean_squared_error_points: {rmse:.4f}\n")
    f.write(f"winners_correct: {correct} of {len(df)} (in-sample)\n")
    f.write(f"matrix_rank_without_sum_row: {rank_without}\nmatrix_rank_with_sum_row: {rank_with}\n")
    f.write(f"connected_games_network: {connected}\nsum_of_ratings: {ratings.sum():.6f}\n")
    f.write("\nHeld-out games (not used to fit):\n" + "\n".join("  " + l for l in test_lines) + "\n")

plot = table.sort_values("Rating")
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(plot["Team"], plot["Rating"])
ax.axvline(0, color="k", linewidth=1)
ax.set_xlabel("Estimated team rating (points better than an average Big 12 team)")
ax.set_title("Big 12 Football 2025 - Model-Based Team Ratings")
ax.grid(axis="x", linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/Team_Ranking_Chart.png", dpi=200)
plt.close(fig)

print(f"\nSaved Team_Ratings.csv, Game_Residuals.csv, Model_Metrics.txt and Team_Ranking_Chart.png to {OUTPUT_DIR}/")
