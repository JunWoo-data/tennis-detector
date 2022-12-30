# %%
from utils import get_maximum_clip
from config import DATA_PATH
import time

# %%
# Download YOLOv7 repository and install requirements
# !git clone https://github.com/RizwanMunawar/yolov7-segmentation.git
# %cd yolov7-segmentation
# !pip install -r requirements.txt

# %%
# download COCO starting checkpoint
# !wget https://github.com/RizwanMunawar/yolov7-segmentation/releases/download/yolov7-segmentation/yolov7-seg.pt

# %%
# segment/predict.py line 189 change to:
# line = (cls, *segj, conf) if save_conf else (cls, *xyxy, *segj)  # label format

# %%
# segment/sort_count.py line 40 change to:
# #matplotlib.use('TkAgg')
# %%
def segment_detect(season, match_date, court_number, match_number):
    max_clip_number = get_maximum_clip(season, match_date, court_number, match_number)
    
    for i in range(1, max_clip_number + 1):
        print("== Detect players for: " + season + " / " + match_date + " / " + court_number + " / " + match_number + " / clip " + str(i) + " ==")
        
        start = time.time()
        
        clip_path = DATA_PATH + "detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/" + "clip" + str(i)
        source_path = clip_path + "/frames/frame_0.jpg"
        
        !python3 segment/predict.py --weights yolov7-seg.pt --conf 0.1 --source {source_path} --save-txt --classes 0 --project {clip_path} --name segmentation_detect --exist-ok

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for clip {i}")
# %%
segment_detect("22F", "20220908", "court1", "match1")

# %%
