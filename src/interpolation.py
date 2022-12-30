# %%
from config import DATA_PATH, WIDTH_ORIGINAL, HEIGHT_ORIGINAL, DEFAULT_NORMALIZED_WIDTH_FOR_BOX, DEFAULT_NORMALIZED_HEIGHT_FOR_BOX
from utils import visualize_point_on_image, visualize_labels_of_frame, get_maximum_clip, get_maximum_frame
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt 
from google.colab.patches import cv2_imshow
import os


# %%
# TODO: Delete
# def visualize_point_on_image(img, coordinates):
#     img_copy = img.copy()
#     output = cv2.circle(img_copy, (int(coordinates[0]), int(coordinates[1])), 15, (0,255,0), 4)
    
#     cv2_imshow(output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# %%
# TODO: delete
# import glob

# def get_maximum_clip(season, match_date, court_number, match_number):
#     base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
#     return len(os.listdir(base_path)) - len(glob.glob(base_path + "*.*"))

# def get_maximum_frame(season, match_date, court_number, match_number, clip_number):
#     base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/clip" + str(clip_number) + "/frames/"
    
#     return len(glob.glob(base_path + "*.jpg"))

# %%
def find_coordinate_transform_matrix(season, match_date, court_number, match_number):
    match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    court_coordinates = pd.read_csv(match_path + "court_coordinates.csv")
    
    tl_x = court_coordinates[court_coordinates["type"] == "top_outer"]["Rx1"].values[0] 
    tl_y = court_coordinates[court_coordinates["type"] == "top_outer"]["Ry1"].values[0] 
    tl = (tl_x, tl_y)

    tr_x = court_coordinates[court_coordinates["type"] == "top_outer"]["Rx2"].values[0] 
    tr_y = court_coordinates[court_coordinates["type"] == "top_outer"]["Ry2"].values[0] 
    tr = (tr_x, tr_y)

    bl_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["Rx1"].values[0] 
    bl_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["Ry1"].values[0] 
    bl = (bl_x, bl_y)

    br_x = court_coordinates[court_coordinates["type"] == "bottom_outer"]["Rx2"].values[0] 
    br_y = court_coordinates[court_coordinates["type"] == "bottom_outer"]["Ry2"].values[0] 
    br = (br_x, br_y)
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0],
    		        [maxWidth - 1, 0],
    		        [maxWidth - 1, maxHeight - 1],
    		        [0, maxHeight - 1]], dtype = "float32")
    
    rect = np.float32([tl, tr, br, bl])
    
    M = cv2.getPerspectiveTransform(rect, dst)
    
    frame_0_path = match_path + "clip1/frames/frame_0.jpg"
    frame_0 = cv2.imread(frame_0_path)

    warped = cv2.warpPerspective(frame_0, M, (maxWidth, maxHeight))
    
    return M, warped

# %%
def Rxy2RTxy(Rxy_array, M):
    Rxy_array = Rxy_array.reshape(-1, 1, 2)
    RTxy_array = cv2.perspectiveTransform(Rxy_array, M).reshape(-1, 2)

    return RTxy_array
    
    
# %%
def make_all_player_labels_enrichment(season, match_date, court_number, match_number):
    match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    all_player_labels_path = match_path + "/all_player_labels.csv"
    all_player_labels = pd.read_csv(all_player_labels_path)
    all_player_labels = all_player_labels[all_player_labels["label"].isin([0])] 
    all_player_labels.reset_index(inplace = True, drop = True)
    
    M, warped = find_coordinate_transform_matrix(season, match_date, court_number, match_number)
    
    all_player_labels["Rxf"] = all_player_labels["xc"] * WIDTH_ORIGINAL
    all_player_labels["Ryf"] = (all_player_labels["yc"] + all_player_labels["h"] / 2) * HEIGHT_ORIGINAL
    
    Rxy_array = np.float32(all_player_labels[["Rxf", "Ryf"]].values.tolist())
    RTxy_array = Rxy2RTxy(Rxy_array, M)
    
    RTxy_df = pd.DataFrame(RTxy_array, columns = ["RTxf", "RTyf"])
    all_player_labels["RTxf"] = RTxy_df["RTxf"]
    all_player_labels["RTyf"] = RTxy_df["RTyf"]
    
    return all_player_labels

# %%
def make_court_coordinates_enrichment(season, match_date, court_number, match_number):
    match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    court_coordinates = pd.read_csv(match_path + "court_coordinates.csv")
    
    M, warped = find_coordinate_transform_matrix(season, match_date, court_number, match_number)
    
    Rx1y1_array = np.float32(court_coordinates[["Rx1", "Ry1"]].values.tolist())
    Rx2y2_array = np.float32(court_coordinates[["Rx2", "Ry2"]].values.tolist())
    
    RTx1y1_array = Rxy2RTxy(Rx1y1_array, M)
    RTx2y2_array = Rxy2RTxy(Rx2y2_array, M)
    
    RTx1y1_df = pd.DataFrame(RTx1y1_array, columns = ["RTx1", "RTy1"])
    RTx2y2_df = pd.DataFrame(RTx2y2_array, columns = ["RTx2", "RTy2"])
    
    court_coordinates["RTx1"] = RTx1y1_df["RTx1"]
    court_coordinates["RTy1"] = RTx1y1_df["RTy1"]
    court_coordinates["RTx2"] = RTx2y2_df["RTx2"]
    court_coordinates["RTy2"] = RTx2y2_df["RTy2"]
    
    return court_coordinates

# %%
def make_empty_player_assign_info(season, match_date, court_number, match_number):
    match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    court_coordinates = pd.read_csv(match_path + "court_coordinates.csv")
    
    M, warped = find_coordinate_transform_matrix(season, match_date, court_number, match_number)
    
    tl_player_info = court_coordinates.loc[court_coordinates["type"] == "top_outer", ["Rx1", "Ry1"]]
    tr_player_info = court_coordinates.loc[court_coordinates["type"] == "top_outer", ["Rx2", "Ry2"]]
    br_player_info = court_coordinates.loc[court_coordinates["type"] == "bottom_outer", ["Rx2", "Ry2"]]
    bl_player_info = court_coordinates.loc[court_coordinates["type"] == "bottom_outer", ["Rx1", "Ry1"]]
    
    empty_player_assign_info = pd.DataFrame([tl_player_info.values[0], tr_player_info.values[0], br_player_info.values[0], bl_player_info.values[0]], 
                                            columns = ["Rxf", "Ryf"]) 
    empty_player_assign_info["player_by_location"] = ["tl", "tr", "br", "bl"]
    
    empty_player_assign_info.loc[empty_player_assign_info["Rxf"] < 0, "Rxf"] = 0
    empty_player_assign_info.loc[empty_player_assign_info["Rxf"] > WIDTH_ORIGINAL, "Rxf"] = WIDTH_ORIGINAL
    empty_player_assign_info.loc[empty_player_assign_info["Ryf"] < 0, "Ryf"] = 0
    empty_player_assign_info.loc[empty_player_assign_info["Ryf"] > HEIGHT_ORIGINAL, "Ryf"] = HEIGHT_ORIGINAL
    
    Rxy_array = np.float32(empty_player_assign_info[["Rxf", "Ryf"]].values.tolist())
    RTxy_array = Rxy2RTxy(Rxy_array, M)
    
    RTxy_df = pd.DataFrame(RTxy_array, columns = ["RTxf", "RTyf"])
    empty_player_assign_info["RTxf"] = RTxy_df["RTxf"]
    empty_player_assign_info["RTyf"] = RTxy_df["RTyf"]
    
    empty_player_assign_info["xc"] = empty_player_assign_info["Rxf"] / WIDTH_ORIGINAL
    empty_player_assign_info["yc"] = empty_player_assign_info["Ryf"] / HEIGHT_ORIGINAL - DEFAULT_NORMALIZED_HEIGHT_FOR_BOX / 2 
    empty_player_assign_info["w"] = DEFAULT_NORMALIZED_WIDTH_FOR_BOX
    empty_player_assign_info["h"] = DEFAULT_NORMALIZED_HEIGHT_FOR_BOX
    empty_player_assign_info = empty_player_assign_info[["player_by_location", "xc", "yc", "w", "h", "Rxf", "Ryf", "RTxf", "RTyf"]]
    
    return empty_player_assign_info
    
# %%
def make_reduced_player_labels(season, match_date, court_number, match_number):
    print(f"== season: {season} / match date: {match_date} / court number: {court_number} / match_number: {match_number} ==")
    
    ## 1) make reduced_player_labels_frame_0
    print("")
    print("== (Step1) Make all frame 0 for each clip to have exactly 4 players ==")
    
    M, warped = find_coordinate_transform_matrix(season, match_date, court_number, match_number)
    all_player_labels_enriched = make_all_player_labels_enrichment(season, match_date, court_number, match_number)
    court_coordinates_enriched = make_court_coordinates_enrichment(season, match_date, court_number, match_number)
    
    reduced_player_labels_frame_0 = all_player_labels_enriched[(all_player_labels_enriched["RTxf"] >= -100) & 
                                                               (all_player_labels_enriched["RTxf"] <= warped.shape[1] + 100) & 
                                                               (all_player_labels_enriched.frame_number == 0)]
    
    print("- all_player_labels_enriched shape: ", all_player_labels_enriched.shape)
    print("- reduced_player_labels_frame_0 shape: ", reduced_player_labels_frame_0.shape)
    
    temp_grouped = reduced_player_labels_frame_0.groupby(["clip_number", "frame_number"]).count().label
    gt_4 = temp_grouped[temp_grouped > 4].reset_index() # clip, frames list that have more than 4 players -> we will delete detected player that is not real player
    lt_4 = temp_grouped[temp_grouped < 4].reset_index() # clip, frames list that have less than 4 players -> we will fill arbitrary player at emty corner
    
    print("- gt_4 cases: ", gt_4.shape[0])
    print("- lt_4 cases: ", lt_4.shape[0])
    print("")
    
    # handle gt_4 cases
    print("-- Handling greater than 4 players cases...")
    for i in range(gt_4.shape[0]):
        target_clip_number = gt_4.iloc[i].clip_number
        target_frame_number = gt_4.iloc[i].frame_number
        
        print(f"- clip number: {target_clip_number} / frame number: {target_frame_number}")

        target_labels = reduced_player_labels_frame_0[(reduced_player_labels_frame_0.clip_number == target_clip_number) & 
                                                      (reduced_player_labels_frame_0.frame_number == target_frame_number)]
        
        temp_df = target_labels[["RTxf", "RTyf"]]
        temp_distance_df = pd.DataFrame(distance_matrix(temp_df.values, temp_df.values), index=temp_df.index, columns=temp_df.index)
        
        duplicated_index = {}
        for j in range(temp_distance_df.shape[0]):
            duplicated_index_list = []
            for k in range(j + 1, temp_distance_df.shape[0]):
                if temp_distance_df.iloc[j, k] < 10: duplicated_index_list.append(temp_distance_df.iloc[0].index[k])

            duplicated_index[temp_distance_df.index[j]] = duplicated_index_list
        
        print("- duplicated cases: ", duplicated_index)
        
        index_to_delete = []
        
        for v in duplicated_index.values():
            index_to_delete.append(v)
        
        index_to_delete = sum(index_to_delete, [])

        reduced_player_labels_frame_0 = reduced_player_labels_frame_0[~reduced_player_labels_frame_0.index.isin(index_to_delete)]
    
    print("- reduced_player_labels_frame_0 shape after handling gt_4 cases: ", reduced_player_labels_frame_0.shape)
    
    temp_grouped = reduced_player_labels_frame_0.groupby(["clip_number", "frame_number"]).count().label
    gt_4 = temp_grouped[temp_grouped > 4].reset_index()
    
    print("- gt_4 cases after handle gt_ cases: ", gt_4.shape[0])
    print("")
    
    # handle lt_4 cases
    print("-- Handling less than 4 players cases... --")
    
    empty_player_assign_info = make_empty_player_assign_info(season, match_date, court_number, match_number)
    
    for i in range(lt_4.shape[0]):
        target_clip_number = lt_4.iloc[i].clip_number
        target_frame_number = lt_4.iloc[i].frame_number
        
        print(f"- clip number: {target_clip_number} / frame number: {target_frame_number}")

        target_labels = reduced_player_labels_frame_0[(reduced_player_labels_frame_0.clip_number == target_clip_number) & 
                                                      (reduced_player_labels_frame_0.frame_number == target_frame_number)]
        
        target_empty_player_assign_info = empty_player_assign_info.copy()
        target_empty_player_assign_info["clip_number"] = target_clip_number
        target_empty_player_assign_info["frame_number"] = target_frame_number
        target_empty_player_assign_info["label"] = 0
        target_empty_player_assign_info = target_empty_player_assign_info[["player_by_location", "clip_number", "frame_number", "label", "xc", "yc", "w", "h", "Rxf", "Ryf", "RTxf", "RTyf"]]
        
        x_threshold = court_coordinates_enriched.loc[court_coordinates_enriched["type"] == "middle", "RTx1"].values[0]
        y_threshold = court_coordinates_enriched.loc[court_coordinates_enriched["type"] == "net", "RTy1"].values[0]
        
        player_to_be_added = []
        if target_labels[(target_labels["RTxf"] < x_threshold) & (target_labels["RTyf"] < y_threshold)].shape[0] == 0: player_to_be_added.append("tl")
        elif target_labels[(target_labels["RTxf"] > x_threshold) & (target_labels["RTyf"] < y_threshold)].shape[0] == 0: player_to_be_added.append("tr")
        elif target_labels[(target_labels["RTxf"] > x_threshold) & (target_labels["RTyf"] > y_threshold)].shape[0] == 0: player_to_be_added.append("br")
        elif target_labels[(target_labels["RTxf"] < x_threshold) & (target_labels["RTyf"] > y_threshold)].shape[0] == 0: player_to_be_added.append("bl")
        
        print("- empty player: ", player_to_be_added)
        
        df_to_be_added = target_empty_player_assign_info[target_empty_player_assign_info["player_by_location"].isin(player_to_be_added)].iloc[:, 1:]
        
        reduced_player_labels_frame_0 = reduced_player_labels_frame_0.append(df_to_be_added).reset_index(drop = True)
        
    print("- reduced_player_labels_frame_0 shape after handling lt_4 cases: ", reduced_player_labels_frame_0.shape)
    print("")
    
    print("!!! (check) count labeld players for each frame for every clip is not 4", np.sum(reduced_player_labels_frame_0.groupby(["clip_number", "frame_number"]).count().label != 4))
    print("== (step1) finish ==")
    print("")
    
    ## 2) make reduced_player_labels
    print("== (step2) Make all other frames for each clip to have exactly 4 players ==")
    
    reduced_player_labels_frame_0.loc[(reduced_player_labels_frame_0["RTxf"] < x_threshold) & 
                                      (reduced_player_labels_frame_0["RTyf"] < y_threshold), "player_by_location"] = "tl"
    reduced_player_labels_frame_0.loc[(reduced_player_labels_frame_0["RTxf"] > x_threshold) & 
                                      (reduced_player_labels_frame_0["RTyf"] < y_threshold), "player_by_location"] = "tr"
    reduced_player_labels_frame_0.loc[(reduced_player_labels_frame_0["RTxf"] > x_threshold) & 
                                      (reduced_player_labels_frame_0["RTyf"] > y_threshold), "player_by_location"] = "br"
    reduced_player_labels_frame_0.loc[(reduced_player_labels_frame_0["RTxf"] < x_threshold) & 
                                      (reduced_player_labels_frame_0["RTyf"] > y_threshold), "player_by_location"] = "bl"
    
    print("!!! (check) count player_by_location is null", np.sum(reduced_player_labels_frame_0.player_by_location.isnull()))
    
    
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    for i in range(1, max_clip_number + 1):
        print(f"-- clip: {i}")
        
        current_clip = all_player_labels_enriched[all_player_labels_enriched["clip_number"] == i]
        
        before_frame = reduced_player_labels_frame_0.loc[reduced_player_labels_frame_0["clip_number"] == i, 
                                                         ["clip_number", "frame_number", "Rxf", "Ryf", "RTxf", "RTyf", "player_by_location"]]

        max_frame_number = get_maximum_frame(season, match_date, court_number, match_number, i)

        for j in range(1, max_frame_number):
            print("-- frame: ", j)
            
            current_frame = current_clip[current_clip["frame_number"] == j]
            
            temp_distance_df = pd.DataFrame(distance_matrix(before_frame[["RTxf", "RTyf"]], current_frame[["RTxf", "RTyf"]]), 
                                            index = before_frame.player_by_location, columns=current_frame.index)
            
            player_to_be_added = {}
            
            for k in range(4):
                #print("temp_distance_df shape: ", temp_distance_df.shape)
                #display(temp_distance_df)
                min_index = np.argmin(temp_distance_df)
                #print("min_index: ", min_index)
                r, c = divmod(min_index, temp_distance_df.shape[1])
                print("min distance value: ", temp_distance_df.iloc[r, c])
                #print("r, c: ", (r, c))
                
                player_to_be_added[temp_distance_df.columns[c]] = temp_distance_df.index[r]
                temp_distance_df = temp_distance_df.loc[temp_distance_df.index != temp_distance_df.index[r], 
                                                        ~temp_distance_df.columns.isin([temp_distance_df.columns[c]])]

    
    
            current_frame.loc[player_to_be_added.keys(), "player_by_location"] = list(player_to_be_added.values())
            current_frame = current_frame.loc[player_to_be_added.keys(), :]
            reduced_player_labels_frame_0 = reduced_player_labels_frame_0.append(current_frame)
            
            temp_4_lag = reduced_player_labels_frame_0[reduced_player_labels_frame_0.clip_number == i].sort_values(["clip_number", "player_by_location", "frame_number"])
            temp_4_lag["xc_lag_1"] = temp_4_lag["xc"].shift(1)
            temp_4_lag["yc_lag_1"] = temp_4_lag["yc"].shift(1)
            temp_4_lag["w_lag_1"] = temp_4_lag["w"].shift(1)
            temp_4_lag["h_lag_1"] = temp_4_lag["h"].shift(1)
            temp_4_lag["Rxf_lag_1"] = temp_4_lag["Rxf"].shift(1)
            temp_4_lag["Ryf_lag_1"] = temp_4_lag["Ryf"].shift(1)
            temp_4_lag["RTxf_lag_1"] = temp_4_lag["RTxf"].shift(1)
            temp_4_lag["RTyf_lag_1"] = temp_4_lag["RTyf"].shift(1)
            temp_4_lag["distance"] = np.sqrt(np.square(temp_4_lag["RTxf"] - temp_4_lag["RTxf_lag_1"]) + np.square(temp_4_lag["RTyf"] - temp_4_lag["RTyf_lag_1"]))
            temp_4_lag.loc[temp_4_lag["frame_number"] == 0, "distance"] = 0

            target_index = temp_4_lag[temp_4_lag.distance > 300].sort_values("distance").index

            print("- Number of frames that have labels farther than 300 from labels of before frame: ", len(target_index))  
            print("- target_index: ", target_index)

            target_values = temp_4_lag.loc[temp_4_lag.index.isin(target_index), 
                                          ["xc_lag_1", "yc_lag_1", "w_lag_1", "h_lag_1", "Rxf_lag_1", "Ryf_lag_1", "RTxf_lag_1", "RTyf_lag_1"]]      
    
            reduced_player_labels_frame_0.loc[reduced_player_labels_frame_0.index.isin(target_index), 
                                              ["xc", "yc", "w", "h", "Rxf", "Ryf", "RTxf", "RTyf"]] = target_values.values
            
            
            before_frame = reduced_player_labels_frame_0.loc[(reduced_player_labels_frame_0["clip_number"] == i) & 
                                                 (reduced_player_labels_frame_0["frame_number"] == j), 
                                                 ["clip_number", "frame_number", "Rxf", "Ryf", "RTxf", "RTyf", "player_by_location"]]

            
        print("!!! (check) count player_by_location is null", np.sum(reduced_player_labels_frame_0.player_by_location.isnull()))
        print("== (step2) finish ==")
        print("")
        
    reduced_player_labels_frame_0.to_csv(match_path + "reduced_player_labels.csv", index = False)
    print("== Reduced player labels saved for " + season + " / " + match_date + " / " + court_number + " / " + match_number + " :")
    print("- save path: " + match_path + "reduced_player_labels.csv")
    print("- file name: reduced_player_labels.csv")
    print("- size: " + str(reduced_player_labels_frame_0.shape))
    print(" ")
        
    return reduced_player_labels_frame_0

# %%
def check_player_by_location(season, match_date, court_number, match_number):
    match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    reduced_player_labels = pd.read_csv(match_path + "reduced_player_labels.csv")

    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    if not os.path.exists("check/"):
        os.mkdir("check/")
        
    for i in range(1, max_clip_number + 1):
        frame_0_path = match_path + f"clip{i}/frames/frame_0.jpg"
        frame_0 = cv2.imread(frame_0_path)
        
        current_clip = reduced_player_labels[reduced_player_labels.clip_number == i]
        current_clip = current_clip.sort_values(["player_by_location", "frame_number"])[["player_by_location", "frame_number", "Rxf", "Ryf"]]
        
        tl = current_clip.loc[current_clip.player_by_location == "tl", ["frame_number", "Rxf", "Ryf"]]
        tr = current_clip.loc[current_clip.player_by_location == "tr", ["frame_number", "Rxf", "Ryf"]]
        bl = current_clip.loc[current_clip.player_by_location == "bl", ["frame_number", "Rxf", "Ryf"]]
        br = current_clip.loc[current_clip.player_by_location == "br", ["frame_number", "Rxf", "Ryf"]]
        
        tl.index = tl.frame_number
        tr.index = tr.frame_number
        bl.index = bl.frame_number
        br.index = br.frame_number
        
        fig = plt.figure(figsize = (20, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(tl["Rxf"], tl["Ryf"], c = tl.index, cmap = plt.cm.Blues, s=100, zorder=1)
        plt.imshow(cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Top-Left Player movement during the clip", fontsize = 20)

        plt.subplot(2, 2, 2)
        plt.scatter(tr["Rxf"], tr["Ryf"], c = tr.index, cmap = plt.cm.Reds, s=100, zorder=1)
        plt.imshow(cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Top-Right Player movement during the clip", fontsize = 20)


        plt.subplot(2, 2, 3)
        plt.scatter(bl["Rxf"], bl["Ryf"], c = bl.index, cmap = plt.cm.Greens, s=100, zorder=1)
        plt.imshow(cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Bottom-Left Player movement during the clip", fontsize = 20)

        plt.subplot(2, 2, 4)
        plt.scatter(br["Rxf"], br["Ryf"], c = br.index, cmap = plt.cm.Greys, s=100, zorder=1)
        plt.imshow(cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Bottom-Right Player movement during the clip", fontsize = 20)

        fig.tight_layout()
        fig.savefig(f"check/{season}_{match_date}_{court_number}_{match_number}_clip{i}.png")
        plt.close(fig)
        
# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
frame_0_path = match_path + "clip1/frames/frame_0.jpg"
frame_0 = cv2.imread(frame_0_path)

# %%
M, warped = find_coordinate_transform_matrix(season, match_date, court_number, match_number)

# %%
all_player_labels_enriched = make_all_player_labels_enrichment(season, match_date, court_number, match_number)
all_player_labels_enriched

# %%
court_coordinates_enriched = make_court_coordinates_enrichment(season, match_date, court_number, match_number)
court_coordinates_enriched

# %%
reduced_player_labels_frame_0 = make_reduced_player_labels(season, match_date, court_number, match_number)
reduced_player_labels_frame_0

# %%
np.sum(reduced_player_labels_frame_0.isnull())

# %%
reduced_player_labels_frame_0.to_csv(match_path + "reduced_player_labels.csv", index = False)

# %%
reduced_player_labels = pd.read_csv(match_path + "reduced_player_labels.csv")
reduced_player_labels

# %%
check_player_by_location(season, match_date, court_number, match_number)
