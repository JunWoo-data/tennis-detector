# %%
from config import DATA_PATH
import pandas as pd

# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
player_labels_file_path = match_path + "/all_player_labels.csv"
player_labels_file = pd.read_csv(player_labels_file_path)
player_labels_file = player_labels_file[player_labels_file["label"].isin([0])] 
player_labels_file

# %%
# boundary 밖 사람은 제거


# %%
# check frames that has less than 4

player_labels_grouped = player_labels_file.groupby(["clip_number", "frame_number"]).count().label
player_labels_grouped
# %%

player_labels_grouped[player_labels_file.groupby(["clip_number", "frame_number"]).count().label < 4]

# %%
