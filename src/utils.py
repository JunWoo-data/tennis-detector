# %%
from moviepy.editor import VideoFileClip
import pytube
from pytube import YouTube
import cv2
import os
from google.colab.patches import cv2_imshow
from google.colab import drive
import numpy as np
from config import DATA_PATH

# %%
def video_to_clip(youtube_link, match_date, court_number, match_number, clip_list):
    # make folder if not exists
    base_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/"
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # download youtube video from link to the folder
    yt = YouTube(youtube_link)
    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download(base_path, "match_video.mp4")

    
    video_path = base_path + "match_video.mp4"
    clip = VideoFileClip(video_path)

    for i in range(len(clip_list)):
        clip = clip_list[i]

        start_string = clip[0]
        start_sec = int(start_string.split(":")[0]) * 60 + int(start_string.split(":")[1]) * 1

        end_string = clip[1]
        end_sec = int(end_string.split(":")[0]) * 60 + int(end_string.split(":")[1]) * 1

        clip_number = 1
        clip_path = base_path + "clip" + str(clip_number) + "/"

        if not os.path.exists(clip_path):
          os.makedirs(clip_path)

        clip.subclip(start_sec, end_sec).write_videofile(clip_path + "clip" + str(clip_number) + ".mp4")

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


