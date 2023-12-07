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
"""
jl.eval(""" """"
        mutable struct StatefulBelief
    belief::Any
end

        function reset_belief()
    global statefulbelief = StatefulBelief(initial_belief()) 
end

function initial_belief()
    # Transition Matrix
    T = zeros(2,3,2) # |S|x|A|x|S|, T[s', a, s] = p(s'|a,s)
    # Alert action
    T[1,1,1]=1
    T[2,1,2]=1
    # Gather info action
    T[1,2,1]=0.8
    T[2,2,1]=0.2
    T[1,2,2]=0.2
    T[2,2,2]=0.8
    # Continue
    T[1,3,1]=0.6
    T[1,3,2]=0.4
    T[2,3,1]=0.4
    T[2,3,2]=0.6

    # Observation Matrix - "Gather info: 2"
    O = zeros(2,3,2) # |O|x|A|x|S|, O[o, a, s] = p(o|a,s)
recall = 0.3395
precision = 0.4755
    # Confidence high
        O[1,2,1] = recall*precision + 0.4
        O[1,2,2] = 1 - (recall*precision + 0.4)
        # Confidence low
        O[2,2,1] = 1 - (recall*precision + 0.4)
        O[2,2,2] = recall*precision + 0.4

    for a in [1,3]
        for s in 1:2
            O[1, a, s] = 0.5 # some default value
            O[2, a, s] = 0.5 # ensure probabilities sum to 1
        end
    end

    # Reward Matrix
    R = zeros(2,3) # |S|x|A|, R[s, a]
    R[1,1]=-100
    R[1,2]=0
    #R[1,2]=80
    R[1,3]=80
    R[2,1]=50
    R[2,2]=-105
    R[2,3]=-75

    # Model
    discount = 0.95
    pomdp = TabularPOMDP(T, R, O, discount);
    updater = DiscreteUpdater(pomdp)
    belief = initialize_belief(updater, POMDPModels.DiscreteDistribution{Vector{Float64}}([0.5, 0.5]))
    return belief
end
global statefulbelief = StatefulBelief(initial_belief()) 
function generate_action(cs)

    # Transition Matrix
    T = zeros(2,3,2) # |S|x|A|x|S|, T[s', a, s] = p(s'|a,s)
    # Alert action
    T[1,1,1]=1
    T[2,1,2]=1
    # Gather info action
    T[1,2,1]=0.8
    T[2,2,1]=0.2
    T[1,2,2]=0.2
    T[2,2,2]=0.8
    # Continue
    T[1,3,1]=0.6
    T[1,3,2]=0.4
    T[2,3,1]=0.4
    T[2,3,2]=0.6

    # Observation Matrix - "Gather info: 2"
    O = zeros(2,3,2) # |O|x|A|x|S|, O[o, a, s] = p(o|a,s)
recall = 0.3395
precision = 0.4755
    # Confidence high
        O[1,2,1] = recall*precision + 0.4
        O[1,2,2] = 1 - (recall*precision + 0.4)
        # Confidence low
        O[2,2,1] = 1 - (recall*precision + 0.4)
        O[2,2,2] = recall*precision + 0.4

    for a in [1,3]
        for s in 1:2
            O[1, a, s] = 0.5 # some default value
            O[2, a, s] = 0.5 # ensure probabilities sum to 1
        end
    end


    # Reward Matrix
    R = zeros(2,3) # |S|x|A|, R[s, a]
    R[1,1]=-100
    R[1,2]=0
    #R[1,2]=80
    R[1,3]=80
    R[2,1]=50
    R[2,2]=-105
    R[2,3]=-75

    # Model
    discount = 0.95
    pomdp = TabularPOMDP(T, R, O, discount);

    #global statefulbelief

    belief = statefulbelief.belief
    sarsop_policy = SARSOP.load_policy(pomdp, "policy.out") 
    updater = DiscreteUpdater(pomdp)
    println(belief)
    action, _ = action_info(sarsop_policy, belief) 
    #println(action)
    if cs > 0.5
        observation = 2
    else 
        observation = 1
    end
    belief = update(updater, belief, action, observation) 
    #Main.statefulbelief.belief = belief
    statefulbelief.belief = belief
    return action, belief
end
        """#)
        

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
    yolo_alerts = []
    pomdp_alerts = []
    pomdp_gathers = []
    #belief = jl.eval("initial_belief()")
    count_frame = 0
    neoCount = 0
    neoCount_list = []
    cs_list = []
    action_list = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count_frame += 1
            if count_frame %5==0:
                neoCount += 1
                neoCount_list.append(neoCount)
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
                    yolo_alerts.append(1)
                frames.append(frameResult)
                #filenamex = f"frame_{count_frame}.jpg"
                #cv2.imwrite(rf"C:\Users\Abdoulaye\YOLOv3_PyTorch\yolo_infraStraight_frames\{filenamex}", frameResult)
                cv2.imshow('Frame', frameResult)
                #cs=0.8
                #if target == 'person' and cs is not None:
                #if cs is not None:
                    #print(belief)
                if cs is None:
                    cs = 0
                    yolo_alerts.append(0)
                cs_list.append(cs)
                action, belief = jl.eval(f"generate_action({cs})")
                #action = bayes_filter(cs)
                action_list.append(action)
                if action == 1:
                    print('ALERT OPERATOR!')
                    pomdp_alerts.append(1)
                    pomdp_gathers.append(0)
                if action == 2:
                    print('GATHER INFORMATION!')
                    pomdp_alerts.append(0)
                    pomdp_gathers.append(1)
                if action == 3:
                    print('CONTINUE MISSION!')
                    pomdp_alerts.append(0)
                    pomdp_gathers.append(0)
                print('--------------------------------------------------------------')
                #cv2.imwrite('./resultVideo/img%06d.jpg'%(count_frame),frameResult)
                out.write(frameResult)

                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        else:
            break
    finish = time.time()
    imageio.mimwrite('yolo_infraOrbit.mp4', frames, fps=20)
    print('Processed video %s with %d frames in %f seconds.' % (videoname, count_frame, (finish - start)))
    print("Saved video result to %s" % ('result_' + videoname))
    cs_array = np.array(cs_list)
    mean_cs = np.mean(cs_array)
    print('Mean confidence score: ', mean_cs)
    std_cs = np.std(cs_array)
    print('Standard Deviation confidence score: ', std_cs)

    # plot cs_list versus action_list
    df = pd.DataFrame({'Confidence scores over mission': cs_list, 
                    'Actions taken': action_list, 'Frames': neoCount_list})
    action_dict = {1: 'alert operator', 2: 'gather information', 3: 'continue mission'}
    df['Actions taken'] = df['Actions taken'].map(action_dict)
    df['Actions taken'] = pd.Categorical(df['Actions taken'], categories=action_dict.values())
    df = pd.concat([df, pd.DataFrame({'Confidence scores over mission': [np.nan], 'Actions taken': ['alert operator']})], ignore_index=True)
    plt.figure(1)
    '''
    plt.scatter(cs_list, action_list)
    '''
    sns.catplot(data=df, x='Confidence scores over mission', y='Actions taken', kind='swarm')
    plt.xlabel('Confidence scores over mission')
    plt.ylabel('Actions taken')
    # title of the graph
    plt.title('Decisions taken during Orbit maneuver - YOLO INFRA')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_infraOrbit_csac.png', bbox_inches='tight')
    plt.show()

    # plot frames versus cs_list
    plt.figure(2)
    plt.scatter(neoCount_list, cs_list)
    plt.xlabel('Frames')
    plt.ylabel('Confidence score')
    # title of the graph
    plt.title('Confidence scores evolution during Orbit maneuver - YOLO INFRA')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_infraOrbit_neocs.png')
    plt.show()

    # plot frames versus action_list
    plt.figure(3)
    '''
    plt.scatter(neoCount_list, action_list)
    '''
    sns.catplot(data=df, x='Frames', y='Actions taken', kind='swarm')
    plt.xlabel('Frames')
    plt.ylabel('Actions taken')
    # title of the graph
    plt.title('Decisions evolution during Orbit maneuver - YOLO INFRA')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_infraOrbit_neoac.png', bbox_inches='tight')
    plt.show()

    
    # plot yolo detections 
    detections_complete_yolo = [-1] * len(frames)
    detections_complete_pomdp = [-1] * len(frames)
    detections_complete_pomdp_gathers = [-1] * len(frames)
    for i in range(len(yolo_alerts)):
        detections_complete_yolo[i] = yolo_alerts[i]
    for i in range(len(pomdp_alerts)):
        detections_complete_pomdp[i] = pomdp_alerts[i]
    for i in range(len(pomdp_gathers)):
        detections_complete_pomdp_gathers[i] = pomdp_gathers[i]
    plt.figure(4)
    frame_num = list(range(len(frames)))
    ground_truth = np.zeros(len(frames))
    ground_truth[:] = 1
    plt.scatter(frame_num, ground_truth+3)
    detection_array_yolo = np.array(detections_complete_yolo)
    plt.scatter(frame_num, detection_array_yolo+2.75)
    detection_array_pomdp = np.array(detections_complete_pomdp)
    plt.scatter(frame_num, detection_array_pomdp+2.5)
    detection_array_pomdp_gathers = np.array(detections_complete_pomdp_gathers)
    plt.scatter(frame_num, detection_array_pomdp_gathers+2.25)
    plt.ylim(3.05,4.05)
    plt.xlabel('Frame Number')
    plt.ylabel('Algorithms & Ground truth alerts')
    plt.title('Alerts during Orbit maneuver - YOLO INFRA') 
    plt.yticks([4,3.75,3.5,3.25],['person present','yolo','filter alerts','filter gathers info'])
    plt.tight_layout()
    plt.savefig('yolo_infraOrbit_final.png')
    plt.show()
    
    # Performance metrics calculations
    yolo_tp=0
    yolo_fp=0
    yolo_tn=0
    yolo_fn=0
    yolo_true_positives = []
    yolo_false_positives = []
    yolo_true_negatives = []
    yolo_false_negatives = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1 and detection_array_yolo[i] == 1:
            yolo_tp += 1
            #yolo_tp.append(yolo_true_positives)
        if ground_truth[i] == 0 and detection_array_yolo[i] == 1:
            yolo_fp += 1
            #yolo_fp.append(yolo_false_positives)
        if ground_truth[i] == 0 and detection_array_yolo[i] == 0:
            yolo_tn += 1
            #yolo_tn.append(yolo_true_negatives)
        if ground_truth[i] == 1 and detection_array_yolo[i] == 0:
            yolo_fn += 1
            #yolo_fn.append(yolo_false_negatives)

    pomdp_tp=0
    pomdp_fp=0
    pomdp_tn=0
    pomdp_fn=0
    pomdp_true_positives = []
    pomdp_false_positives = []
    pomdp_true_negatives = []
    pomdp_false_negatives = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1 and detection_array_pomdp[i] == 1:
            pomdp_tp += 1
            #pomdp_tp.append(pomdp_true_positives)
        if ground_truth[i] == 0 and detection_array_pomdp[i] == 1:
            pomdp_fp += 1
            #pomdp_fp.append(pomdp_false_positives)
        if ground_truth[i] == 0 and detection_array_pomdp[i] == 0:
            pomdp_tn += 1
            #pomdp_tn.append(pomdp_true_negatives)
        if ground_truth[i] == 1 and detection_array_pomdp[i] == 0:
            pomdp_fn += 1
            #pomdp_fn.append(pomdp_false_negatives)

    pomdp_precision = pomdp_tp/(pomdp_tp+pomdp_fp)
    pomdp_recall = pomdp_tp/(pomdp_tp+pomdp_fn)
    pomdp_f1 = 2/((1/pomdp_precision)+(1/pomdp_recall))

    yolo_precision = yolo_tp/(yolo_tp+yolo_fp)
    yolo_recall = yolo_tp/(yolo_tp+yolo_fn)
    yolo_f1 = 2/((1/yolo_precision)+(1/yolo_recall))

    print('YOLO true positives: ', yolo_tp)
    print('YOLO false positives: ', yolo_fp)
    print('YOLO true negatives: ', yolo_tn)
    print('YOLO false negatives: ', yolo_fn)

    print('POMDP true positives: ', pomdp_tp)
    print('POMDP false positives: ', pomdp_fp)
    print('POMDP true negatives: ', pomdp_tn)
    print('POMDP false negatives: ', pomdp_fn)

    print('---------------------------------------')

    print('POMDP precision= ', pomdp_precision)
    print('POMDP recall= ', pomdp_recall)
    print('POMDP f1_score= ', pomdp_f1)

    print('YOLO precision= ', yolo_precision)
    print('YOLO recall= ', yolo_recall)
    print('YOLO f1_score= ', yolo_f1)
    
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
