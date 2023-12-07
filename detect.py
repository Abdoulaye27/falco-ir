import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from myopic_filter import bayes_filter
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet
import cv2
from julia.api import Julia
jl = Julia(compiled_modules=False, runtime="C:\\Users\\Abdoulaye\\AppData\\Local\\Programs\\Julia-1.8.5\\bin\\julia.exe")
jl.eval('include("falco_function.jl")')
import tqdm
import imageio

jl.eval("reset_belief()")

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/kaist_person.names'
    class_names = load_class_names(namesfile)
 
    use_cuda = True
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture("infraOrbit.mp4")
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    start = time.time()
    frames = []
    count_frame = 0
    while True:
        res, img = cap.read()
        if res:
            count_frame += 1
            if count_frame %5==0:
                sized = cv2.resize(img, (m.width, m.height))

                new_img = np.zeros_like(sized)
                img_mean = np.mean(sized,-1)
                new_img[:,:,0] = img_mean
                new_img[:,:,1] = img_mean
                new_img[:,:,2] = img_mean

                sized = new_img
                bboxes = do_detect(m, sized, 0.1, 0.4, use_cuda)
                frame = new_img
                frameResult, cs, target = plot_boxes_cv2(frame, bboxes, class_names=class_names)
                print('Confidence score is {}'.format(cs), flush=True)
                if target is not None:
                    print('Detected target is {}'.format(target))
                frames.append(frameResult)
                if cs is None:
                    cs = 0
                action, belief = jl.eval(f"generate_action({cs})")
                #action = bayes_filter(cs)
                if action == 1:
                    print('ALERT OPERATOR!')
                    cv2.imshow('Frame', frameResult)
                if action == 2:
                    print('GATHER INFORMATION!')
                if action == 3:
                    print('CONTINUE MISSION!')
                print('--------------------------------------------------------------')

                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        else:
            break

    finish = time.time()
    imageio.mimwrite('mission_recapVideo.mp4', frames, fps=20)
    cap.release()
    cv2.destroyAllWindows()
    
############################################
if __name__ == '__main__':
    cfgfile = 'cfg/yolov3_kaist.cfg'
    weightfile = 'weights/kaist_thermal_detector.weights'
    if len(sys.argv) >=1:
        if len(sys.argv) == 3:
            cfgfile = sys.argv[1]
            weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
    else:
        print('Usage:')
        print('    python demo.py [cfgfile] [weightfile]')
        print('    perform detection on camera')
