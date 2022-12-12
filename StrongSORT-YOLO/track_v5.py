# Khai báo các thư viện cần thiết để sử dụng
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
from collections import deque
import xlsxwriter
import warnings
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


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # đường dẫn đến file weights của yolov5
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # đường dẫn đến file weights của strongsort
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # kích thước của ảnh khi chạy qua mô hình yolov5
        conf_thres=0.25,  # giá trị ngưỡng của vật thể - box
        iou_thres=0.45,  # giá trị ngưỡng IoU
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # hiển thị kết quả
        save_txt=False,  # lưu kết quả về dạng *.txt
        save_crop=False,  # crop vật thể nếu phát hiện được
        save_vid=False,  # lưu lại giá trị ngưỡng trong --save-txt labels
        save_conf=False,  # save confidences in --save-txt labels
        nosave=False,  # không lưu lại kết quả hình ảnh/ video
        classes=None,  # lọc kết quả theo class mà mình muốn
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # giá trị augment data (trong trường hợp để training mô hình)
        visualize=False,  # hiển thị các đặc trưng
        update=False,  # cập nhật mô hình
        project=ROOT / 'runs/track',  # lưu kết quả về đường dẫn
        name='exp',  # lưu kết quả về đường dẫn
        exist_ok=False,
        line_thickness=2,  # giá trị của viền bounding box (gtri càng to - viền càng dày)
        hide_labels=False,  # ẩn giá trị label của đối tượng - box
        hide_conf=False,  # ẩn giá trị ngưỡng của đối tượng
        hide_class=False,  # ẩn IDs của đối tượng
        half=False,  # sử dụng FP16 để chạy
        dnn=False,  # sử dụng OpenCV DNN để chạy ONNX (trong trường hợp convert code về ONNX)
        count=False,  # tính số lượng của mọi đối tượng
        count_frame=100, #Tính số lượng của mọi đối tượng trong 1s
        draw=False,  # vẽ đường quỹ đạo đối tượng

):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Thư mục
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model - load kiến trúc và trọng số của mô hình yolov5
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, namess, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    trajectory = {}

    #Tạo file excel có tên là Results.xlsx
    workbook = xlsxwriter.Workbook('Results.xlsx')
    worksheet = workbook.add_worksheet()
    #
    if isinstance(namess, list):
        names = namess
    elif isinstance(namess, dict):
        print(namess)
        names = []
        for item in namess.values():
            names.append(item)
    else:
        print("Kiểm tra lại class")
    #Tạo mảng toàn giá trị 0, độ dài tương ứng với số class của mô hình
    line_excecl = []
    for i, name in enumerate(names):
        line_excecl.append(0)
    # Ghi dòng thứ 1 bao gồm tên toàn bộ các class
    name_excel = ["frames/names"] + names
    for col_num, data in enumerate(name_excel):
        worksheet.write(1, col_num, data)
    int_line_excel = 2


    # Dataloader - load dữ liệu để đưa vào mô hình
    if webcam: # Nếu sử dụng webcam
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else: # Sử dụng các nguồn khác
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Khai báo các paramater và khởi tạo mô hình StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Tạo các luồng mô hình StrongSORT tương ứng với số lượng video/data khai báo ở trên
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Bắt đầu vào phần chính - tracking vật thể
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup - hàm sinh ra để khi khởi tạo model, sẽ tự động run 1 lần trước
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset): # chạy từng ảnh của tập datasets
        # tiền xử lý ảnh đầu vào
        t1 = time_sync()
        im = torch.from_numpy(im).to(device) # chuyển ảnh từ dang numpy về dạng tensor để đưa vào mô hình yolov5
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0 # đưa ảnh về dạng từ 0 đến 1 - giúp tính toán nhanh hơn

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference - chạy qua mô hình yolov5
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize) # pred là kết quả sau khi chạy qua yolov5
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #xử lý các box bị trùng, đè và giá trị ngưỡng thấp
        dt[2] += time_sync() - t3

        # Process detections - hậu xử lý kết quả của mô hình yolov5
        for i, det in enumerate(pred):  # chạy và xử lý cho từng bounding box
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count # khởi tạo đường dẫn - ảnh cần lưu
                p = Path(p)  # đến đường dẫn vừa khởi tạo
                s += f'{i}: '
                txt_file_name = p.name # tên file txt chuẩn bị lưu
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ... # tên file chuẩn bị lưu
            else: # tương tự như với webcam
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # nếu ta cần crop bounding box, ta sẽ copy thêm 1 ảnh nữa

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # cập nhật ảnh vào trong mô hình strongsort
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det): # nếu giá trị của box không rỗng - None
                # Trả lại tọa độ của box từ kích thước img_zise về kích thước ảnh thật ban đầu
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4]) # chuyển tọa độ từ [x1,y1,x2,y2] sang [x,y,w,h]
                confs = det[:, 4] # lấy giá trị ngưỡng của bounding box
                clss = det[:, 5] # lấy class của box sau khi predict

                # Đưa các giá trị tọa độ bounding box, conf và class vào mô hình StrongSORT
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # Vẽ box và hiển thị
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4] #trả thêm giá trị id của từng box
                        cls = output[5]
                        bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes
                        
                        if draw: # nếu sử dụng draw, ta sẽ vẽ đường quỹ đạo đối tượng
                            # object trajectory
                            center = ((int(bboxes[0]) + int(bboxes[2])) // 2,(int(bboxes[1]) + int(bboxes[3])) // 2)
                            if id not in trajectory:
                                trajectory[id] = []
                            trajectory[id].append(center)
                            for i1 in range(1,len(trajectory[id])):
                                if trajectory[id][i1-1] is None or trajectory[id][i1] is None:
                                    continue
                                # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
                                thickness = 2
                                try:
                                  cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                                except:
                                  pass


                        if save_txt: # lưu toàn bộ kêts quả về file .txt
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path+'.txt', 'a') as f:
                                f.write(('%g ' * 11 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                            if int(frame_idx + 1) % int(count_frame) == 0:
                                if os.path.exists(txt_path + '_' + str(count_frame) + '.txt'):
                                    os.remove(txt_path + '_' + str(count_frame) + '.txt')
                            if int(frame_idx + 1) % int(count_frame) != 0:
                                with open(txt_path + '_' + str(count_frame) + '.txt', 'a') as f:
                                    f.write(('%g ' * 11 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))


                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else: #nếu giá trị của box rỗng
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')


            if count: # sử dụng tính năng đếm số lượng vật thể
                itemDict={}
                ## NOTE: this works only if save-txt is true
                try: # sử dụng file .txt đã lưu, để lấy các thông tin về số lượng vật thể
                    df = pd.read_csv(txt_path +'.txt' , header=None, delim_whitespace=True)
                    df = df.iloc[:,0:3]
                    df.columns=["frameid" ,"class","trackid"]
                    df = df[['class','trackid']]
                    df = (df.groupby('trackid')['class']
                              .apply(list)
                              .apply(lambda x:sorted(x))
                             ).reset_index()

                    df.colums = ["trackid","class"]
                    df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0])) # kết quả số lượng vật thể
                except:
                    pass

                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1 = 10
                    y1 = 10
                    x2 = 10
                    y2 = 70
                    # vẽ kết quả lên trên ảnh
                    put_text = "Total: " + str(itemDict)
                    txt_size = cv2.getTextSize(put_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                    cv2.putText(im0, '{}'.format(put_text), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)

            if count:  # sử dụng tính năng đếm số lượng vật thể trong 60 frame
                itemDict = {}
                ## NOTE: this works only if save-txt is true
                try:  # sử dụng file .txt đã lưu, để lấy các thông tin về số lượng vật thể
                    df = pd.read_csv(txt_path + '_' + str(count_frame) + '.txt', header=None, delim_whitespace=True)
                    df = df.iloc[:, 0:3]
                    df.columns = ["frameid", "class", "trackid"]
                    df = df[['class', 'trackid']]
                    df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x: sorted(x))
                          ).reset_index()

                    df.colums = ["trackid", "class"]
                    df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))  # kết quả số lượng vật thể
                    # print(itemDict)
                except:
                    pass

                try:
                    if int(frame_idx) % int(count_frame) == 0:
                        line_excecl_new = line_excecl.copy()
                        print(line_excecl_new)
                        for j, item in enumerate(itemDict):
                            for k, name in enumerate(names):
                                if str(item) == str(name):
                                    line_excecl_new[k] = itemDict[item]
                        print(line_excecl_new)
                        name_excel = [str(frame_idx) + " " + "Frame"] + line_excecl_new
                        for col_num, data in enumerate(name_excel):
                            worksheet.write(int_line_excel, col_num, data)
                        int_line_excel = int_line_excel + 1
                except:
                    pass
                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1 = 10
                    y1 = 100
                    y2 = 160
                    # vẽ kết quả lên trên ảnh
                    put_text_60 = str(count_frame) + " " + "Frame: " + str(itemDict)
                    txt_size_60 = cv2.getTextSize(put_text_60, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(im0, (x1, y1 + 1), (txt_size_60[0] * 2, y2), (0, 0, 0), -1)
                    cv2.putText(im0, '{}'.format(put_text_60), (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (210, 210, 210), 2)
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)


            #current frame // tesing
            cv2.imwrite('testing.jpg',im0)


            if show_vid: # nếu sử dụng, sẽ hiển thị kết quả trực tiếp khi đang chạy
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid: # nếu sử dụng, ta sẽ save video kết quả về
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    workbook.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'strongsort/osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    parser.add_argument('--count_frame', type=int, default=100, help='display all MOT counts results on screen')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
