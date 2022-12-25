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
    source_path = "/content/drive/MyDrive/ALT+TAB/data/detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number + "/original_frames/"
    project_path = "/content/drive/MyDrive/ALT+TAB/data/detect/" + season + "/" + match_date + "/" + court_number + "/" + match_number
    
    !python detect.py --weights /yolov7/yolov7_training.pt --conf 0.1 --source source_path --save-txt --classes 0 38 --project project_path --name player_detect --exist-ok --nosave
