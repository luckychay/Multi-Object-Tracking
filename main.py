# limit the number of cpus used by high performance libraries
import os

from cv2 import WINDOW_NORMAL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import signal
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from detector.yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from detector.yolov5.utils.torch_utils import select_device, time_sync
from detector.yolov5.utils.plots import Annotator, colors

from tracker.build import build_tracker
from detector.build import build_detector

from tools.io import *


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def main(args):

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not args.evaluate:
        if os.path.exists(args.output):
            pass
            shutil.rmtree(args.output)  # delete output folder
        os.makedirs(args.output)  # make new output folder

    # Directories
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # check if desired device is avaliable
    args.device = select_device(args.device)
    device = args.device
    print("using device {} computing...".format(device))
    
    # initialize detector
    model = build_detector(args)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz  = args.imgsz
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    args.half &= pt and args.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if args.half else model.model.float()

    # initialize tracker
    tracker = build_tracker(args)

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Check if environment supports image displays
    if args.show_vid:
        show_vid = check_imshow()

    source = args.source
    rostopic = source == '1'
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    elif rostopic:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRosTopic(args.topic, img_size=imgsz, stride=stride, auto=pt and not jit, datatype=CompressedImage)
        bs = len(dataset)  # batch_size
    else:   
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    # Prepare for entrance counting
    if args.en_counting:
        in_id_list = list()
        out_id_list = list()
        in_flag = dict()
        out_flag = dict()
        prev_center = dict()
        count_str = ""

    # create publisher
    # pub = PublishRosTopic()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    signal.signal(signal.SIGINT, handler)

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        tram_status = 0
        # when stopped, start the detection process
        if tram_status==0:
            print("The tram is stopped, detection started.")
            # do image enhancement
            if args.process:
            # do equilized histgram for BCHW RGB images
                for i in range(img.shape[0]): 
                    pic = img[i][::-1,...].transpose((1,2,0)) # move the channel dim to the last and convert into BGR(Opencv needs)

                    lab= cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)
                    #-----Splitting the LAB image to different channels-------------------------
                    l, a, b = cv2.split(lab)
                    #-----Applying CLAHE to L-channel-------------------------------------------
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
                    limg = cv2.merge((cl,a,b))
                    #-----Converting image from LAB Color model to RGB model--------------------
                    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    # cv2.imshow('final', final)
                    # cv2.waitKey(1)
                    img[i] = final[...,::-1].transpose(2,0,1)
                    im0s[i] = final.copy()

            # Use roi to filter 
            if args.roi:
                origin = (50,80)
                img[...,0:origin[1],:] = 0
                img[...,origin[1]::,0:origin[0]] = 0

            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if args.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False
            pred = model(img, augment=args.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'webcam{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                # Determination of the lines may be tricky
                shape = im0.shape
                entrance1 =  tuple(map(int,[0, shape[0] / 2.0, shape[1], shape[0] / 2.0]))
                entrance2 =  tuple(map(int,[0, shape[0] / 1.8, shape[1], shape[0] / 1.8]))

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
    
                    # pass detections to deepsort
                    t4 = time_sync()
                    if args.tracker == 'deepsort':
                        outputs = tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    elif args.tracker == 'bytetracker':
                        outputs = tracker.update(det[:, 0:5].cpu(),[im0.shape[0],im0.shape[1]],[im0.shape[0],im0.shape[1]])
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        # xyxy boxes to be sent
                        data = boxes()
                        for j, (output, conf,cls) in enumerate(zip(outputs, confs, clss)):

                            bboxes = output[0:4]
                            track_id = output[4]
                            # cls = output[5]

                            #store box detected
                            b = box()
                            b.coordinates = bboxes
                            data.boxes.append(b)

                            c = int(cls)  # integer class
                            label = f'{track_id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                            # Use two line to do entrance counting
                            if args.en_counting and cls in args.classes:

                                entrance_y1 = entrance1[1] 
                                entrance_y2 = entrance2[1]

                                if track_id < 0: continue

                                x1, y1, x2,y2 = bboxes
                                center_x = (x1 + x2)/2.
                                center_y = (y1 + y2)/2.

                                if track_id in prev_center:

                                    # In number counting 
                                    if prev_center[track_id][1] <= entrance_y1 and \
                                    center_y > entrance_y1:
                                        in_flag[track_id] = 1
                                    elif prev_center[track_id][1] <= entrance_y2 and \
                                    center_y > entrance_y2 and in_flag[track_id] == 1:
                                        in_id_list.append(track_id)
                                        in_flag[track_id] = 0

                                    # Out number counting
                                    elif prev_center[track_id][1] >= entrance_y2 and \
                                    center_y < entrance_y2:
                                        out_flag[track_id] = 1
                                    elif prev_center[track_id][1] >= entrance_y1 and \
                                    center_y < entrance_y1 and out_flag[track_id] == 1:
                                        out_id_list.append(track_id)
                                        out_flag[track_id] = 0

                                    prev_center[track_id] = [center_x, center_y]
                                else:
                                    prev_center[track_id] = [center_x, center_y]
                                    in_flag[track_id] = 0
                                    out_flag[track_id] = 0
                                
                                count_str = f"In: {len(in_id_list)}, Out: {len(out_id_list)}"
                                print(count_str)

                            if args.save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                        # # send xyxy boxes
                        # pub.send(data)

                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                else:
                    if args.tracker == 'deepsort':
                        tracker.increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()

                if show_vid:
                    
                    lw = 3
                    tf = max(lw - 1, 1)
                    w, h = cv2.getTextSize(s, 0, fontScale=lw / 3, thickness=tf)[0] 
                    p1 = (0,0)
                    p2 = (p1[0] + int(w), p1[1]+int(h)+10)
                    cv2.rectangle(im0, p1, p2, (0,240,240), -1, cv2.LINE_AA)  # filled
                    cv2.putText(im0, s, (p1[0], p1[1]+h+3), 0, lw / 3, (255,255,255),
                                thickness=tf, lineType=cv2.LINE_AA)

                    if args.en_counting:
                        w, h = cv2.getTextSize(count_str, 0, fontScale=lw / 3, thickness=tf)[0]
                        p1 = (0,p2[1])
                        p2 = (p1[0] + int(w), p1[1]+int(h)+10)
                        cv2.rectangle(im0, p1, p2, (240,240,0), -1, cv2.LINE_AA)  # filled
                        cv2.putText(im0, count_str, (p1[0], p1[1]+h+3), 0, lw / 3, (255,255,255),
                                    thickness=tf, lineType=cv2.LINE_AA)
                        cv2.rectangle(im0,entrance1[0:2],entrance1[2:4],(0,255,255),1)
                        cv2.rectangle(im0,entrance2[0:2],entrance2[2:4],(0,255,255),1)

                    # cv2.namedWindow(str(p),WINDOW_NORMAL)  
                    # cv2.resizeWindow(str(p),640,480)  
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if args.save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    vid_writer.write(im0)
        else:
            print("The tram is driving, detection is stopped.")

            if args.en_counting:
                in_id_list.clear()
                out_id_list.clear()
                in_flag.clear()
                out_flag.clear()
                prev_center.clear()
                count_str = ""

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if args.save_txt or args.save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--tracker",type=str, default="deepsort",help="set tracker")
    parser.add_argument("--detector",type=str, default="yolov5", help="set detetor")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    ####byte track
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result",action="store_true",help="whether to save the inference result of image/video",)
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    ####deep sort
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="tracker/deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--roi', action='store_true', help='turn on roi filter')
    parser.add_argument('--en_counting', action='store_true', help='turn on entrance counting')
    parser.add_argument('--process', action='store_true', help='turn on image processing')
    parser.add_argument('--topic', default='/usb_cam/image_raw/compressed', help='rostopic to be subscribed')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()

    with torch.no_grad():
        main(args)



