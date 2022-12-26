# %%
# !sudo pip3 install imageio==2.4.1

# %%
from moviepy.editor import VideoFileClip
import pytube
from pytube import YouTube
import cv2
import os
from google.colab.patches import cv2_imshow
from google.colab import drive
import numpy as np
import glob.glob
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
        
    os.remove(video_path)
# %%
def video_to_frames(youtube_link, season, match_date, court_number, match_number):
    # make folder if not exists
    base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # download youtube video from link to the folder
    yt = YouTube(youtube_link)
    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download(base_path, "match_video.mp4")

    # makke frames from downloaded video
    frame_path = base_path + "/original_frames/"
    
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    video_path = base_path + "match_video.mp4"
    clip = cv2.VideoCapture(video_path)
    
    print("== Make clips for: " + season + " / " + match_date + " / " + court_number + " / " + match_number + " ==")
    
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
    
    # delete downloaded video because we only need frames from video
    os.remove(video_path)

# %%
clip_info_list = [["0:9", "0:21"], ["0:43", "0:58"], ["1:11", "1:23"], ["1:38", "1:50"], ["2:05", "2:10"], ["2:25", "2:31"], ["2:48", "2:56"], ["3:09", "3:17"],
["3:33", "3:43"], ["3:55", "4:00"], ["4:22", "4:27"], ["4:45", "4:53"], ["5:09", "5:14"], ["5:31", "5:35"], ["5:59", "6:07"], ["6:21", "6:27"],
["6:40", "6:46"], ["6:57", "7:10"], ["7:22", "7:36"], ["7:54", "8:08"], ["8:30", "8:56"], ["9:13", "9:35"], ["9:47", "10:04"], ["10:33", "10:48"],
["11:00", "11:15"], ["11:31", "11:40"], ["11:53", "12:01"], ["13:06", "13:17"], ["13:30", "13:42"], ["13:52", "14:25"], ["14:39", "14:56"], ["15:09", "15:15"],
["15:36", "15:56"], ["16:15", "16:23"], ["16:41", "16:47"], ["17:00", "17:14"], ["17:37", "17:42"], ["17:57", "18:08"], ["18:31", "18:37"], ["18:49", "18:55"],
["19:06", "19:25"], ["19:40", "19:54"], ["20:10", "20:18"], ["20:36", "20:40"], ["20:53", "21:05"], ["21:16", "21:22"], ["21:32", "21:48"], ["22:00", "22:10"],
["22:23", "22:37"], ["23:13", "23:32"], ["24:00", "24:07"], ["24:22", "24:29"], ["24:43", "24:57"], ["25:10", "25:18"], ["25:33", "25:51"], ["26:58", "27:20"],
["27:50", "27:59"], ["28:13", "28:35"], ["28:53", "28:59"], ["29:16", "29:21"], ["29:39", "29:46"], ["29:57", "30:07"], ["30:27", "30:34"]]
video_to_clip("https://youtu.be/AdmCmegtgc8", "22F", "20220908", "court1", "match1", clip_info_list)


# %%
def get_maximum_clip(season, match_date, court_number, match_number):
    base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    return len(os.listdir(base_path))
   
# %%
clip_file_path = base_path + "clip" + str(1) + "/" + "clip" + str(1) + ".mp4"
os.path.exists(clip_file_path)
   
# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"
base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" 
base_path

# %%


# %%
for i in range(len(clip_info_list)):
    clip_info = clip_info_list[i]

    start_string = clip_info[0]
    start_sec = int(start_string.split(":")[0]) * 60 + int(start_string.split(":")[1]) * 1
    end_string = clip_info[1]
    end_sec = int(end_string.split(":")[0]) * 60 + int(end_string.split(":")[1]) * 1
    print("clip" + str(i) + ": " + str(start_sec) + " ~ " + str(end_sec))

# %%
youtube_link = "https://youtu.be/AdmCmegtgc8"
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
# make dir
base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
base_path

# %%
os.path.exists(base_path)
# %%
if not os.path.exists(base_path):
    os.makedirs(base_path)
    
# %%
# download youtube video to dir

yt = YouTube(youtube_link)
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download(base_path, "match_video.mp4")
# %%
# make clip from video
frame_path = base_path + "/original_frames/"
frame_path

# %%
if not os.path.exists(frame_path):
    os.makedirs(frame_path)
# %%
video_path = base_path + "match_video.mp4"
video_path
clip = cv2.VideoCapture(video_path)

# %%
print("== Make clips for: " + season + " / " + match_date + " / " + court_number + " / " + match_number + " ==")
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
# delete video
os.remove(video_path)
# %%


