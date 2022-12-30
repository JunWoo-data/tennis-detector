# %% 
# !sudo pip3 install imageio==2.4.1

# %%
# !pip install pytube

# %%
from utils import get_maximum_clip
from config import DATA_PATH
import glob
import time

# %%
# Download YOLOv7 repository and install requirements
# !git clone https://github.com/WongKinYiu/yolov7
# %cd yolov7
# !pip install -r requirements.txt

# %%
# download COCO starting checkpoint
# %cd yolov7
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt

# %%
def player_detect(season, match_date, court_number, match_number):
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    for i in range(1, max_clip_number + 1):
        print("== Detect players for: " + season + " / " + match_date + " / " + court_number + " / " + match_number + " / clip " + str(i) + " ==")
        
        start = time.time()
        
        clip_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" + "clip" + str(i)
        source_path = clip_path + "/frames"
        
        !python3 detect.py --weights yolov7_training.pt --conf 0.1 --source {source_path} --save-txt --classes 0 38 --project {clip_path} --name player_detect --exist-ok --nosave
        
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for clip {i}")
        
        

# %%
player_detect("22F", "20220908", "court1", "match1")

# %%
