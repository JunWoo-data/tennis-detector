# %%
from config import DATA_PATH, WIDTH_ORIGINAL, HEIGHT_ORIGINAL
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

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
court_coordinates["type"].isin(["top_outer"])
left_threshold = 

# %%
def compute_perspective_transform(corner_points,width,height,image):
	""" Compute the transformation matrix
	@ corner_points : 4 corner points selected from the image
	@ height, width : size of the image
	return : transformation matrix and the transformed image
	"""
	# Create an array out of the 4 corner points
	corner_points_array = np.float32(corner_points)
 
	# Create an array with the parameters (the dimensions) required to build the matrix
	img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
	
    # Compute and return the transformation matrix
	matrix = cv2.getPerspectiveTransform(corner_points_array, img_params) 
	img_transformed = cv2.warpPerspective(image,matrix,(width,height))
 
	return matrix,img_transformed

# %%
tl_x = court_coordinates[court_coordinates["type"] == "top_outer"]["x1"]
tl_y = court_coordinates[court_coordinates["type"] == "top_outer"]["y1"]
tl = (tl_x, tl_y)

tr_x = court_coordinates[court_coordinates["type"] == "top_outer"]["x2"]
tr_y = court_coordinates[court_coordinates["type"] == "top_outer"]["y2"]
tr = (tr_x, tr_y)

bl_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["x1"]
bl_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["y1"]
bl = (bl_x, bl_y)

br_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["x2"]
br_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["y2"]
br = (br_x, br_y)

# %%
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# %%



# %%
corner_points = [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)]
corner_points_array = np.float32(corner_points)
corner_points_array

# %%
width = WIDTH_ORIGINAL
height = HEIGHT_ORIGINAL

# %%
img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])

# %%
matrix = cv2.getPerspectiveTransform(corner_points_array, img_params) 

# %%
img_path = match_path + "clip1/frames/frame_0.jpg"
img = cv2.imread(img_path)

# %%
img_transformed = cv2.warpPerspective(img, matrix, (width,height))

# %%
cv2_imshow(img_transformed)

# %%
# check frames that has less than 4

player_labels_grouped = player_labels_file.groupby(["clip_number", "frame_number"]).count().label
player_labels_grouped
# %%

player_labels_grouped[player_labels_file.groupby(["clip_number", "frame_number"]).count().label < 4]

# %%
