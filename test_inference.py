"""
Run YOLOv5 detection inference on images
Code adapted from detect.py
Authors: Jonathan Walsh, Haixin Zhao, Christian Escobar. For any questions contact: jonwalsh@udel.edu, zhx@udel.edu, escobarc@udel.edu

"""
from models.common import DetectMultiBackend
from pathlib import Path
import sys
import os
import numpy as np
import torch
import bluetooth as bt
from picamera2 import Picamera2, Preview
import time
import cv2 as cv_t

from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath
import inputs
import pygame
from pygame.locals import *
from time import sleep
import math
import threading
import RPi.GPIO as GPIO
#import xbox
import os
#match the pins between Raspberry Pi and motor driver  
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)

save_dir ='outputs'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def pin_setup():
    # Pin setup
    in1, in2, ena = 24, 23, 25
    in3, in4, enb = 5, 6, 26

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([in1, in2, in3, in4], GPIO.OUT)
    GPIO.setup([ena, enb], GPIO.OUT)


    GPIO.output([in1, in2, in3, in4], GPIO.LOW)

    pR = GPIO.PWM(ena, 255)
    pL = GPIO.PWM(enb, 255)
    pR.start(0)
    pL.start(0)

#GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset

#GPIO.setup(in1,GPIO.OUT) #right motor
#GPIO.setup(in2,GPIO.OUT) #right motor

#GPIO.setup(in3,GPIO.OUT) #left motor
#GPIO.setup(in4,GPIO.OUT) #left motor

print("\n")
print(".....")
print('\n')



pygame.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    joystick.init()
def get_prediction(im0,weights,device,imgsz,name_to_save='temp',conf_thres=0.25,
                   iou_thres=0.45,save_txt=True,save_img=True):
        '''
        Inputs:
            im0: Image matrix 
            weights: model weights
            device: '' or 'cpu' or 'gpu'
            imgsz: image size to input the model 
            name_to_save: name of the file to save the annotated image 
                          and the labels in .txt format
            conf_thres:  confidence threshold
            iou_thres:   NMS IOU threshold
            save_txt: Boolean flag to save the labels to a .txt file
            save_img: Boolean flag to save the annotated image to a .jpg file
        '''
        # weights=ROOT / "yolov5s.pt"  # model path or triton URL
        data=ROOT / "data/coco128.yaml"  # dataset.yaml path
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # file_path=Path("bus.jpg")
        
        name_txt=name_to_save+'txt_out'
        file_im=name_to_save+'.jpg'

        im = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        bs = 1  # batch_size
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


        visualize = False
        augment = False
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        
        classes = None
        agnostic_nms = False
        max_det = 1000
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        s=''
        save_crop = False
        line_thickness=3
        hide_conf= False
        save_conf= False
        hide_labels= False
        PREDICTIONS=[]
        for i, det in enumerate(pred):  # per image
            seen += 1
            # p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            # p = Path(p)  # to Path
            save_path = save_dir+'/img/' # str(save_dir / 'img' / file_im)  # im.jpg
            txt_path = save_dir+'/labels/' #str(save_dir / 'labels' / name_txt) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            save_path+=file_im # str(save_dir / 'img' / file_im)  # im.jpg
            txt_path += name_txt #str(save_dir / 'labels' / name_txt) 

            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.numpy()), *xywh, conf) if save_conf else (int(cls.numpy()), *xywh)  # label format
                    PREDICTIONS.append(line)
                    if save_txt:  # Write to file
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
            # Stream results
            im0 = annotator.result()


            # Save results (image with detections)
            if save_img:
                    cv2.imwrite(save_path, im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        return PREDICTIONS

def x_position(PREDICTIONS):
   if not PREDICTIONS:
      print("No Detections")
   else:
      center_x_position = PREDICTIONS[0][1]
      return center_x_position

def move_center_x(center_x_position):
	# Pin setup
    in1, in2, ena = 24, 23, 25
    in3, in4, enb = 5, 6, 26
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([in1, in2, in3, in4], GPIO.OUT)
    GPIO.setup([ena, enb], GPIO.OUT)
    GPIO.output([in1, in2, in3, in4], GPIO.LOW)
    pR = GPIO.PWM(ena, 255)
    pL = GPIO.PWM(enb, 255)
    pR.start(0)
    pL.start(0)
    x_center=center_x_position
    if 0.55<x_center<0.65:
        pR.ChangeDutyCycle(30)
        pL.ChangeDutyCycle(0)
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        #sleep(1)
        print("move_left_1")
    if 0.65<x_center<0.75:
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(0)
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        #sleep(1)
        print("move_left_2")
    if 0.75<x_center<1.0:
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(0)
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        #sleep(1)
        print("move_left_3")

    if 0.35<x_center<0.45:
        pR.ChangeDutyCycle(0)
        pL.ChangeDutyCycle(30)
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        #sleep(1)
        print("move_right_1")
    if 0.25<x_center<0.35:
        pR.ChangeDutyCycle(0)
        pL.ChangeDutyCycle(60)
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        #sleep(1)
        print("move_right_2")
    if 0<x_center<0.25:
        pR.ChangeDutyCycle(0)
        pL.ChangeDutyCycle(100)
        GPIO.output(in1,GPIO.LOW)
    pR.stop()
    pL.stop()
    GPIO.cleanup()



def main():
    

    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
        
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
	
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration({'size': (1664,2048)})
    picam2.configure(camera_config)
    picam2.start()
    
    polling = True
    running = False
    
    print("Press A to enter autonomous mode, Press B to enter controller mode.")
    while polling:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    print(event)
                    polling = False
                    running = True
                    GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset
                    break
                if event.button == 1:
                    print(event)
                    polling = True
                    running = False
                    bt.controller()
                    GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset
                    break
    sleep(1)
    while running:
		# Pin setup
        in1, in2, ena = 24, 23, 25
        in3, in4, enb = 5, 6, 26
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([in1, in2, in3, in4], GPIO.OUT)
        GPIO.setup([ena, enb], GPIO.OUT)
        GPIO.output([in1, in2, in3, in4], GPIO.LOW)
        pR = GPIO.PWM(ena, 255)
        pL = GPIO.PWM(enb, 255)
        pR.start(0)
        pL.start(0)
        im0 = picam2.capture_array()
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        #im0=im0[:-1,:-1,:-1]
        #cv2.imwrite('pic1.jpg', im0)
        #print(im0.shape)
        #weights=ROOT / "best_trash.pt"  # model path or triton URL
        #weights=ROOT / "yolov5s.pt"  # model path or triton URL

        #file_im="pool1.jpeg"
        #im0 = cv2.imread(file_im) # 3D Matrix with the loaded image
        imgsz=[1664,2048]#[640,640]
        name_to_save='temp'
        all_weights=['best_trash_300ep.pt'] #'yolov5s.pt',

        PREDICTIONS=[] #[[(o,x,y,w,h),(),()...],[(o,x,y,w,h),(),()...]]
        device='cpu'
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 4:
                    running=False
                    polling = True
        for count, weight in enumerate(all_weights):

            weights=ROOT / weight
            PREDICTIONS.append(get_prediction(im0.copy(),weights,device,imgsz,name_to_save+str(count)))
            sorted_results = sorted(PREDICTIONS[0],key=lambda x:x[2]) # if adding more weights, change index to look at best_trash_300ep.pt
        print(PREDICTIONS[0])
        print(sorted_results)
        print(x_position(sorted_results))
        # ID=sorted_results[0][0]
        x_center=x_position(sorted_results)
        # width=sorted_results[0][3]
        # height=sorted_results[0][4]
        if not sorted_results:
            pR.ChangeDutyCycle(100)
            pL.ChangeDutyCycle(100)
            GPIO.output([in1,in4],GPIO.HIGH)
            GPIO.output([in2,in3],GPIO.LOW)
            print("Spin...Detecting....")
            time.sleep(4) #sleep to let spin, then make stop
            print("Done Detecting")
            pR.ChangeDutyCycle(0)
            pL.ChangeDutyCycle(0)
            GPIO.output([in1,in2,in3,in4],GPIO.LOW)
        elif (sorted_results[0][0]==0) and ((sorted_results[0][3])*(sorted_results[0][4]) <0.25):
            move_center_x(x_center)
            if 0.45<x_center<0.55:
                pR.ChangeDutyCycle(100)
                pL.ChangeDutyCycle(100)
                GPIO.output([in1,in3],GPIO.HIGH)
                GPIO.output([in2,in4],GPIO.LOW)
                print("move forward")
        elif (sorted_results[0][0] == 1) and ((sorted_results[0][3]) * (sorted_results[0][4]) < 0.25):
            move_center_x(x_center)
            if 0.45<x_center<0.55:
                pR.ChangeDutyCycle(100)
                pL.ChangeDutyCycle(100)
                GPIO.output([in1,in3],GPIO.HIGH)
                GPIO.output([in2,in4],GPIO.LOW)
                print("move forward")
        GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset
        pR.stop()
        pL.stop()
#        elif (((sorted_results[0][0]) == 0) and ((sorted_results[0][3])*(sorted_results[0][4]) < 0.25) and (0.45 < (x_position(sorted_results)) < 0.55)):
#            print("move foward")
#        elif (((sorted_results[0][0]) == 1) and ((sorted_results[0][3])*(sorted_results[0][4]) < 0.25) and (0.45 < (x_position(sorted_results)) < 0.55)):
#            print("move forward")
		


if __name__ == "__main__":
    main()
