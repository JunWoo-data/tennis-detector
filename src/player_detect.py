# %%
from utils import get_maximum_clip
from config import DATA_PATH
import glob

# %%
# Download YOLOv7 repository and install requirements
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt

# %%
# download COCO starting checkpoint
%cd /content/yolov7
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt

# %%
def player_detect(season, match_date, court_number, match_number):
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    for i in range(1, max_clip_number + 1):
        clip_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" + "clip" + str(i) + "/"
        source_list = glob.glob(clip_path + "*.jpg")
        !python detect.py --weights /yolov7/yolov7_training.pt --conf 0.1 --source source_path --save-txt --classes 0 38 --project project_path --name player_detect --exist-ok --nosave


# %%
season = "22F"
match_date = "20220908"
court_number = "court1"
match_number = "match1"
clip_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" + "clip" + str(1) + "/"
clip_path
# %%
source_list = glob.glob(clip_path + "*.jpg")[:10]
source_list
# %%
%cd /content/yolov7
!python detect.py --weights /yolov7/yolov7_training.pt --conf 0.1 --source source_list --save-txt --classes 0 38 --project clip_path --name player_detect --exist-ok --nosave
# %%
