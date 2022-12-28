# %%
from config import DATA_PATH, WIDTH_ORIGINAL, HEIGHT_ORIGINAL
from utils import visualize_point_on_image
import pandas as pd
import numpy as np
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
all_player_labels.reset_index(inplace = True)

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
temp_target_labels = all_player_labels[(all_player_labels["RTxf"] >= -100) & (all_player_labels["RTxf"] <= warped.shape[1] + 100) & (all_player_labels.frame_number == 0)]
temp_target_labels

# %%
temp_target_labels_grouped = temp_target_labels.groupby(["clip_number", "frame_number"]).count().label
temp_target_labels_grouped

# %%
temp_target_labels_grouped[temp_target_labels_grouped > 4]


# %%
temp_target_labels_grouped[temp_target_labels_grouped < 4]


# %%
all_player_labels[(all_player_labels.clip_number == 9) & (all_player_labels.frame_number == 0)]

# %%
# check frames that has less than 4

player_labels_grouped = player_labels_file.groupby(["clip_number", "frame_number"]).count().label
player_labels_grouped
# %%

player_labels_grouped[player_labels_file.groupby(["clip_number", "frame_number"]).count().label < 4]

# %%
