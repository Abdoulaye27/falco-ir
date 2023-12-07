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

namesfile=None

def detect_model(cfgfile, modelfile, dir):
    m = Darknet(cfgfile)

    check_model = modelfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(modelfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(modelfile)

    # m.print_network()
    use_cuda = True
    if use_cuda:
        m.cuda()

    m.eval()

    class_names = load_class_names(namesfile)
    newdir = dir.replace('/', '_') + 'predicted'
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    start = time.time()
    total_time = 0.0
    # count_img = 0
    for count_img, imgfile in enumerate(tqdm.tqdm(os.listdir(dir))):
        # count_img +=1
        imgfile = os.path.join(dir, imgfile)

        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        detect_time_start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

        detect_time_end = time.time() - detect_time_start
        total_time += detect_time_end

        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        plot_boxes_cv2(img, boxes, class_names=class_names, color=red)

        savename = (imgfile.split('/')[-1]).split('.')[0]
        savename = savename + '_predicted.jpg'
        savename = os.path.join(newdir, savename)
        # print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    finish = time.time() - start

    count_img += 1
    print('len dir = %d ' % (count_img))
    # print('Predicted in %d minutes %f seconds with average %f seconds / image.' % (finish//60, finish%60, finish/count_img))
    print('Predicted in %d minutes %f seconds with average %f seconds / image.' % (
    finish // 60, finish % 60, total_time / count_img))


def detect_cv2(cfgfile, weightfile, imgfile):

    m = Darknet(cfgfile)
    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    use_cuda = False
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()

    class_names = load_class_names(namesfile)
    print(len(boxes))
    plot_boxes_cv2(img, boxes, class_names=class_names)
    savename = imgfile.split('.')[0]
    savename = savename+'_predicted.jpg'
    print("save plot results to %s" % savename)
    cv2.imwrite(savename, img)

def readvideo_cv2(cfgfile, weightfile, videoname):
    m = Darknet(cfgfile)
    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    use_cuda = True
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(videoname)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result_' + videoname, fourcc, 28, (frame_width, frame_height))

    start = time.time()

    # List to save each frame as an array
    frames = []
    count_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count_frame += 1
            if count_frame %5==0:
                # Display the resulting frame
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sized = cv2.resize(frame, (m.width, m.height))

                # print('shape 1: ')
                # print(sized.shape)


                new_img = np.zeros_like(sized)
                img_mean = np.mean(sized,-1)
                new_img[:,:,0] = img_mean
                new_img[:,:,1] = img_mean
                new_img[:,:,2] = img_mean

                sized = new_img

                boxes = do_detect(m, sized, 0.1, 0.4, use_cuda)

                class_names = load_class_names(namesfile)

                ##add this
                frame = new_img

                frameResult, cs, target = plot_boxes_cv2(frame, boxes, class_names=class_names)
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
                #cv2.imwrite('./resultVideo/img%06d.jpg'%(count_frame),frameResult)
                out.write(frameResult)

                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        else:
            break
    finish = time.time()
    imageio.mimwrite('mission_recapVideo.mp4', frames, fps=20)
    print('Processed video %s with %d frames in %f seconds.' % (videoname, count_frame, (finish - start)))
    print("Saved video result to %s" % ('result_' + videoname))
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    globals()["namesfile"] = 'data/kaist_person.names'
    cfgfile = 'cfg/yolov3_kaist.cfg'
    weightfile = 'weights/kaist_thermal_detector.weights'
    if len(sys.argv) >= 1:
        if len(sys.argv) == 2:
            imgfile = sys.argv[1]
        elif len(sys.argv) == 3:
            imgfile = sys.argv[1]
            weightfile = sys.argv[2]

        if os.path.isdir(imgfile):
            detect_model(cfgfile, weightfile,imgfile)
        elif (imgfile.split('.')[1] == 'jpg') or (imgfile.split('.')[1] == 'png') or (imgfile.split('.')[1] == 'jpeg'):
            detect_cv2(cfgfile, weightfile, imgfile)
        else:
            readvideo_cv2(cfgfile, weightfile,imgfile)
    else:
        print('Usage: ')
        print('  python detect.py image/video/folder [weightfile]')
        print('  or using:  python detect.py thermal_kaist.png ')
