# %%
from config import DATA_PATH, WIDTH_ORIGINAL, HEIGHT_ORIGINAL
import pandas as pd

# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
all_player_labels_path = match_path + "/all_player_labels.csv"
all_player_labels = pd.read_csv(all_player_labels_path)
all_player_labels = all_player_labels[all_player_labels["label"].isin([0])] 

# %%
all_player_labels

# %%
court_coordinates = pd.read_csv(match_path + "court_coordinates.csv")
court_coordinates

# %%
# boundary 밖 사람은 제거


# %%
# check frames that has less than 4

player_labels_grouped = player_labels_file.groupby(["clip_number", "frame_number"]).count().label
player_labels_grouped
# %%

player_labels_grouped[player_labels_file.groupby(["clip_number", "frame_number"]).count().label < 4]

# %%
