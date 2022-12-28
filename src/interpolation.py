# %%
from config import DATA_PATH, WIDTH_ORIGINAL, HEIGHT_ORIGINAL, DEFAULT_NORMALIZED_WIDTH_FOR_BOX, DEFAULT_NORMALIZED_HEIGHT_FOR_BOX
from utils import visualize_point_on_image
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import cv2
from google.colab.patches import cv2_imshow


# %%
# TODO: Delete
def visualize_point_on_image(img, coordinates):
    img_copy = img.copy()
    output = cv2.circle(img_copy, (int(coordinates[0]), int(coordinates[1])), 15, (0,255,0), 4)
    
    cv2_imshow(output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
all_player_labels.reset_index(inplace = True, drop = True)

# %%
all_player_labels

# %%
court_coordinates = pd.read_csv(match_path + "court_coordinates.csv")
court_coordinates

# %%
# boundary 밖 사람은 제거
tl_x = court_coordinates[court_coordinates["type"] == "top_outer"]["x1"].values[0] 
tl_y = court_coordinates[court_coordinates["type"] == "top_outer"]["y1"].values[0] 
tl = (tl_x, tl_y)

tr_x = court_coordinates[court_coordinates["type"] == "top_outer"]["x2"].values[0] 
tr_y = court_coordinates[court_coordinates["type"] == "top_outer"]["y2"].values[0] 
tr = (tr_x, tr_y)

bl_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["x1"].values[0] 
bl_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["y1"].values[0] 
bl = (bl_x, bl_y)

br_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["x2"].values[0] 
br_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["y2"].values[0] 
br = (br_x, br_y)

# %%
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# %%
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# %%
dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

# %%
rect = np.float32([tl, tr, br, bl])

# %%
M = cv2.getPerspectiveTransform(rect, dst)

# %%
img_path = match_path + "clip1/frames/frame_0.jpg"
img = cv2.imread(img_path)

# %%
cv2_imshow(img)

# %%
img.shape

# %%
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
cv2_imshow(warped)

# %%
warped_with_margine = cv2.copyMakeBorder(warped, 500, 500, 500, 500, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])
cv2_imshow(warped_with_margine)

# %%
M

# %%
all_player_labels

# %%
all_player_labels["Rxf"] = all_player_labels["xc"] * WIDTH_ORIGINAL
all_player_labels["Ryf"] = (all_player_labels["yc"] + all_player_labels["h"] / 2) * HEIGHT_ORIGINAL
all_player_labels

# %%
visualize_point_on_image(img, (all_player_labels.iloc[3]["Rxf"], all_player_labels.iloc[3]["Ryf"]))

# %%
all_player_labels

# %%
list_downoids = all_player_labels[["Rxf", "Ryf"]].values.tolist()
list_downoids

# %%
len(list_downoids)

# %%
list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
list_points_to_detect.shape

# %%
transformed_points = cv2.perspectiveTransform(list_points_to_detect, M)
transformed_points

# %%
transformed_points.shape

# %%
transformed_points

# %%
transformed_points_df = pd.DataFrame(transformed_points.reshape(-1, 2), columns = ["RTxf", "RTyf"])
transformed_points_df.shape

# %%
all_player_labels["RTxf"] = transformed_points_df["RTxf"]
all_player_labels["RTyf"] = transformed_points_df["RTyf"]
all_player_labels



# %%
court_coordinates

# %%
Tx1y1 = pd.DataFrame(cv2.perspectiveTransform(np.float32(court_coordinates[["x1", "y1"]].values.tolist()).reshape(-1, 1, 2), M).reshape(-1, 2), columns = ["Tx1", "Ty1"])

# %%
Tx2y2 = pd.DataFrame(cv2.perspectiveTransform(np.float32(court_coordinates[["x2", "y2"]].values.tolist()).reshape(-1, 1, 2), M).reshape(-1, 2), columns = ["Tx2", "Ty2"])

# %%
court_coordinates["Tx1"] = Tx1y1["Tx1"]
court_coordinates["Ty1"] = Tx1y1["Ty1"]
court_coordinates["Tx2"] = Tx2y2["Tx2"]
court_coordinates["Ty2"] = Tx2y2["Ty2"]
court_coordinates

# %%
warped.shape

# %%
reduced_player_labels = all_player_labels[(all_player_labels["RTxf"] >= -100) & (all_player_labels["RTxf"] <= warped.shape[1] + 100) & (all_player_labels.frame_number == 0)]
reduced_player_labels

# %%
reduced_player_labels_grouped = reduced_player_labels.groupby(["clip_number", "frame_number"]).count().label
reduced_player_labels_grouped

# %%
gt_4 = reduced_player_labels_grouped[reduced_player_labels_grouped > 4].reset_index()
gt_4

# %%
gt_4.shape

# %%
reduced_player_labels

# %%
# for i in range(gt_4.shape[0]):
i = 0

# %%
target_clip_number = gt_4.iloc[i].clip_number
target_frame_number = gt_4.iloc[i].frame_number

target_labels = reduced_player_labels[(reduced_player_labels.clip_number == target_clip_number) & (reduced_player_labels.frame_number == target_frame_number)]
target_labels 
    
# %%
temp_df = target_labels[["RTxf", "RTyf"]]
temp_df

# %%
temp_distance_df = pd.DataFrame(distance_matrix(temp_df.values, temp_df.values), index=temp_df.index, columns=temp_df.index)
temp_distance_df

# %%
temp_distance_df < 10


# %%
temp_distance_df.iloc[0, 1]
# %%
temp_distance_df.iloc[0].index

# %%
temp_distance_df.index

# %%
duplicated_index = {}
for i in range(temp_distance_df.shape[0]):
    duplicated_index_list = []
    for j in range(i + 1, temp_distance_df.shape[0]):
        if temp_distance_df.iloc[i, j] < 10: duplicated_index_list.append(temp_distance_df.iloc[0].index[j])
    
    duplicated_index[temp_distance_df.index[i]] = duplicated_index_list

# %%
duplicated_index

# %%
index_to_delete = []
for v in duplicated_index.values():
    index_to_delete.append(v)

index_to_delete

# %%
index_to_delete = sum(index_to_delete, [])

# %%
index_to_delete

# %%
reduced_player_labels = reduced_player_labels[~reduced_player_labels.index.isin(index_to_delete)]

# %%
np.sum(reduced_player_labels.groupby(["clip_number", "frame_number"]).count().label > 4)

##### lt_4 처리하기
# %%
lt_4 = reduced_player_labels_grouped[reduced_player_labels_grouped < 4].reset_index()
lt_4

# %%
cv2_imshow(warped)

# %%
reduced_player_labels

# %%
player1_info = court_coordinates.loc[court_coordinates["type"] == "top_outer", ["x1", "y1"]]
player2_info = court_coordinates.loc[court_coordinates["type"] == "top_outer", ["x2", "y2"]]
player3_info = court_coordinates.loc[court_coordinates["type"] == "bottom_outer", ["x2", "y2"]]
player4_info = court_coordinates.loc[court_coordinates["type"] == "bottom_outer", ["x1", "y1"]]

# %%

empty_player_assign_info = pd.DataFrame([player1_info.values[0], player2_info.values[0], player3_info.values[0], player4_info.values[0]], columns = ["Rxf", "Ryf"]) 
empty_player_assign_info["player"] = ["player1", "player2", "player3", "player4"]
empty_player_assign_info

# %%
empty_player_assign_info.loc[empty_player_assign_info["Rxf"] < 0, "Rxf"] = 0
empty_player_assign_info.loc[empty_player_assign_info["Rxf"] > WIDTH_ORIGINAL, "Rxf"] = WIDTH_ORIGINAL
empty_player_assign_info.loc[empty_player_assign_info["Ryf"] < 0, "Ryf"] = 0
empty_player_assign_info.loc[empty_player_assign_info["Ryf"] > HEIGHT_ORIGINAL, "Ryf"] = HEIGHT_ORIGINAL
empty_player_assign_info

# %%
list_downoids = empty_player_assign_info[["Rxf", "Ryf"]].values.tolist()
list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
transformed_points = cv2.perspectiveTransform(list_points_to_detect, M)
transformed_points_df = pd.DataFrame(transformed_points.reshape(-1, 2), columns = ["RTxf", "RTyf"])
empty_player_assign_info["RTxf"] = transformed_points_df["RTxf"]
empty_player_assign_info["RTyf"] = transformed_points_df["RTyf"]
empty_player_assign_info

# %%
empty_player_assign_info.iloc[0]["RTxf"]

# %%
visualize_point_on_image(warped, (empty_player_assign_info.iloc[3]["RTxf"], empty_player_assign_info.iloc[3]["RTyf"]))

# %%
all_player_labels.describe()

# %%
DEFAULT_NORMALIZED_WIDTH_FOR_BOX = 0.038
DEFAULT_NORMALIZED_HEIGHT_FOR_BOX = 0.17

# %%
empty_player_assign_info["xc"] = empty_player_assign_info["Rxf"] / WIDTH_ORIGINAL
empty_player_assign_info["yc"] = empty_player_assign_info["Ryf"] / HEIGHT_ORIGINAL - DEFAULT_NORMALIZED_HEIGHT_FOR_BOX / 2 
empty_player_assign_info["w"] = DEFAULT_NORMALIZED_WIDTH_FOR_BOX
empty_player_assign_info["h"] = DEFAULT_NORMALIZED_HEIGHT_FOR_BOX
empty_player_assign_info = empty_player_assign_info[["player", "xc", "yc", "w", "h", "Rxf", "Ryf", "RTxf", "RTyf"]]
empty_player_assign_info


# %%
# for i in range(lt_4.shape[0])
i = 0 

# %%
target_clip_number = lt_4.iloc[i].clip_number
target_frame_number = lt_4.iloc[i].frame_number

target_labels = reduced_player_labels[(reduced_player_labels.clip_number == target_clip_number) & (reduced_player_labels.frame_number == target_frame_number)]
target_labels 

# %%
court_coordinates

# %%
court_coordinates.loc[court_coordinates["type"] == "middle", "Tx1"].values[0]
# %%
target_labels["RTxf"] < 


