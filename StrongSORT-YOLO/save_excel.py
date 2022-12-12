# Khai báo các thư viện cần thiết để sử dụng
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
from collections import deque
import warnings
import xlsxwriter
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # thêm ROOT vào PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # thêm thư mục yolov5 ROOT vào PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # thêm thư mục strong_sort ROOT vào PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import logging
from yolov5.models.common import DetectMultiBackend
try:
    from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
except:
    import sys
    sys.path.append('yolov5/utils')
    from dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# Loại bỏ luồng trùng lặp nếu có 1 luồng khác đang chạy, tránh bị ghi đề (xử lý đa luồng)
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

yolo_weights = "weights/yolov5/yolov5n.pt"
device = select_device("cpu")
model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
print(names)

line_excecl = []
for i , name in enumerate(names):
    line_excecl.append(0)
print(line_excecl)

name_excel = ["frames/names"] + names
workbook = xlsxwriter.Workbook('write_list.xlsx')
worksheet = workbook.add_worksheet()

# for col_num, data in enumerate(name_excel):
#     worksheet.write(1, col_num, data)
#
# workbook.close()
#
# df = pd.read_csv('demo_60.txt' , header=None, delim_whitespace=True)
# df = df.iloc[:,0:3]
# df.columns=["frameid" ,"class","trackid"]
# df = df[['class','trackid']]
# df = (df.groupby('trackid')['class'].apply(list).apply(lambda x:sorted(x))).reset_index()
#
# df.colums = ["trackid","class"]
# df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
# vc = df['class'].value_counts()
# vc = dict(vc)
#
# vc2 = {}
# for key, val in enumerate(names):
#     vc2[key] = val
# print(vc2)
# itemDict = dict((vc2[key], value) for (key, value) in vc.items())
# print(itemDict)
# itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0])) # kết quả số lượng vật thể
# print(itemDict)
#
# for i, item in enumerate(itemDict):
#     for i, name in enumerate(names):
#         if str(item) == str(name):
#             line_excecl[i] = itemDict[item]
# print(line_excecl)