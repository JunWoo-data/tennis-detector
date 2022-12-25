# %%
import pytube
from pytube import YouTube
import cv2
import os
from google.colab.patches import cv2_imshow
from google.colab import drive
import numpy as np
from config import DATA_PATH

# %%
def video_to_frames(youtube_link, season, match_date, court_number, match_number):


# %%
youtube_link = "https://youtu.be/AdmCmegtgc8"
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"

# %%
os.listdir("/content/drive/My Drive/ALT+TAB/")
# %%
DATA_PATH
# %%
