# %%
# !sudo pip3 install imageio==2.4.1

# %%
# !pip install pytube

# %%
from moviepy.editor import VideoFileClip
import pytube
from pytube import YouTube
import cv2
import os
from google.colab.patches import cv2_imshow
from google.colab import drive
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.pyplot import figure
from config import DATA_PATH

# %%
def video_to_clip(youtube_link, season, match_date, court_number, match_number, clip_info_list):
    # make folder if not exists
    base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # download youtube video from link to the folder
    yt = YouTube(youtube_link)
    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download(base_path, "match_video.mp4")

    
    # make clip from downloaded video
    video_path = base_path + "match_video.mp4"
    clip = VideoFileClip(video_path)

    for i in range(len(clip_info_list)):
        clip_info = clip_info_list[i]

        start_string = clip_info[0]
        start_sec = int(start_string.split(":")[0]) * 60 + int(start_string.split(":")[1]) * 1

        end_string = clip_info[1]
        end_sec = int(end_string.split(":")[0]) * 60 + int(end_string.split(":")[1]) * 1

        
        clip_path = base_path + "clip" + str(i + 1) + "/"

        if not os.path.exists(clip_path):
          os.makedirs(clip_path)
        
        clip_file_path = clip_path + "clip" + str(i + 1) + ".mp4"
        
        if os.path.exists(clip_file_path):
            continue
        
        else:
            clip.subclip(start_sec, end_sec).write_videofile(clip_file_path)
    
    # delete downloaded video because we only need frames and clips from video
    os.remove(video_path)
    
# %%
def get_maximum_clip(season, match_date, court_number, match_number):
    base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    return len(os.listdir(base_path))

# %%
def clips_to_frames(season, match_date, court_number, match_number):
    # get maximum clip number to iterate all clips
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    for i in range(1, max_clip_number + 1):
        clip_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" + "clip" + str(i) + "/"
        frame_path = clip_path + "frames/"
        
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        
        clip_file_path = clip_path + "clip" + str(i) + ".mp4"
        clip = cv2.VideoCapture(clip_file_path)   
    
        print("== Make clips for: " + season + " / " + match_date + " / " + court_number + " / " + match_number + " / clip " + str(i) + " ==")
    
        current_frame = 0
    
        while (True):
            ret, frame = clip.read()
            if ret:
                name = frame_path + "frame_" + str(current_frame) + ".jpg"
                cv2.imwrite(name, frame)
                current_frame += 1
            else:
                break


        if (current_frame != 0): 
            print("0 ~ ", current_frame, " frames created.")

        clip.release()
        cv2.destroyAllWindows()
    
# %%
# clip_info_list = [["0:9", "0:21"], ["0:43", "0:58"], ["1:11", "1:23"], ["1:38", "1:50"], ["2:05", "2:10"], ["2:25", "2:31"], ["2:48", "2:56"], ["3:09", "3:17"],
# ["3:33", "3:43"], ["3:55", "4:00"], ["4:22", "4:27"], ["4:45", "4:53"], ["5:09", "5:14"], ["5:31", "5:35"], ["5:59", "6:07"], ["6:21", "6:27"],
# ["6:40", "6:46"], ["6:57", "7:10"], ["7:22", "7:36"], ["7:54", "8:08"], ["8:30", "8:56"], ["9:13", "9:35"], ["9:47", "10:04"], ["10:33", "10:48"],
# ["11:00", "11:15"], ["11:31", "11:40"], ["11:53", "12:01"], ["13:06", "13:17"], ["13:30", "13:42"], ["13:52", "14:25"], ["14:39", "14:56"], ["15:09", "15:15"],
# ["15:36", "15:56"], ["16:15", "16:23"], ["16:41", "16:47"], ["17:00", "17:14"], ["17:37", "17:42"], ["17:57", "18:08"], ["18:31", "18:37"], ["18:49", "18:55"],
# ["19:06", "19:25"], ["19:40", "19:54"], ["20:10", "20:18"], ["20:36", "20:40"], ["20:53", "21:05"], ["21:16", "21:22"], ["21:32", "21:48"], ["22:00", "22:10"],
# ["22:23", "22:37"], ["23:13", "23:32"], ["24:00", "24:07"], ["24:22", "24:29"], ["24:43", "24:57"], ["25:10", "25:18"], ["25:33", "25:51"], ["26:58", "27:20"],
# ["27:50", "27:59"], ["28:13", "28:35"], ["28:53", "28:59"], ["29:16", "29:21"], ["29:39", "29:46"], ["29:57", "30:07"], ["30:27", "30:34"]]

# %%
# video_to_clip("https://youtu.be/AdmCmegtgc8", "22F", "20220908", "court1", "match1", clip_info_list)

# %%
# clips_to_frames( "22F", "20220908", "court1", "match1")

# %%
def combine_player_detect_labels(season, match_date, court_number, match_number):
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    all_player_labels = pd.DataFrame(columns = ["clip_number", "frame_number", "label", "x1", "y1", "x2", "y2"])

    for i in range(1, max_clip_number + 1):
        match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
        labels_path = match_path + "clip" + str(i) + "/player_detect/labels/"
        label_list = glob.glob(labels_path + "*.txt")

        for label in label_list:
            frame_number = int(label.split("/")[-1].split("_")[-1].split(".")[0])

            player_labels = pd.read_csv(label, sep = " ", header = None, names = ["label", "x1", "y1", "x2", "y2"])
            player_labels["clip_number"] = i
            player_labels["frame_number"] = frame_number
            player_labels = player_labels[["clip_number", "frame_number", "label", "x1", "y1", "x2", "y2"]]

            all_player_labels = all_player_labels.append(player_labels)

    all_player_labels.to_csv(match_path + "all_player_labels.csv", index = False)
    print("== All player labels saved for " + season + " / " + match_date + " / " + court_number + " / " + match_number + " :")
    print("- save path: " + match_path + "all_player_labels.csv")
    print("- file name: all_player_labels.csv")
    print("- size: " + str(all_player_labels.shape))
    print(" ")

# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
combine_player_detect_labels(season, match_date, court_number, match_number)


# %%
match_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
labels_file_path = match_path + "/all_player_labels.csv"
labels_file = pd.read_csv(labels_file_path)
clip_number = 1
frames_path = match_path + "/clip" + str(clip_number) + "/frames/"
frame_range = [8, 10]
label = [0]

# %%
os.listdir(frames_path)
# %%
frame_start = frame_range[0]
frame_end = frame_range[1]

# %%
labels_file = labels_file[(labels_file["label"].isin(label)) & 
                          (labels_file["clip_number"] == 1) & 
                          (labels_file["frame_number"].isin(range(frame_start, frame_end + 1)))].sort_values("frame_number")

labels_file
# %%

for fn in range(frame_start, frame_end + 1):
    img_path = frames_path + "frame_" + str(fn) + ".jpg"
    frame = cv2.imread(img_path)
    
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.imshow(frame)
    
    target_labels_file = labels_file[labels_file["frame_number"] == fn]
    
    for ri in range(target_labels_file.shape[0]):
        xmin = target_labels_file.iloc[ri]["x1"]
        ymin = target_labels_file.iloc[ri]["y1"]
        xmax = target_labels_file.iloc[ri]["x2"]
        ymax = target_labels_file.iloc[ri]["y2"]
        
        xy = (int(xmin), int(ymin))
        height = int(ymax - ymin)
        width = int(xmax - xmin)
        
        rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = "r", facecolor = "none")
        ax.add_patch(rect)
    
    plt.show

# %%
target_labels_file = labels_file[labels_file["frame_number"] == 8]
target_labels_file

# %%
target_labels_file.iloc[0]["x1"]

# %%
# def visualize_labels(season, match_date, court_number, match_number, clip_number, frame_range, label = [0, 1, 38])

# %%

temp["label"].isin([0])
# %%
temp[(temp.clip_number == 1) & (temp.frame_number == 0)]