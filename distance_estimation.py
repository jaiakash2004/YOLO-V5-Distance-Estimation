import torch
import cvzone
import math
import cv2
import numpy as np
import time



### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    #print(f"[INFO] Detecting. . . ")
    results = model(frame)
    #results.show()
    # print( results.xyxyn[0])
    #print(results.xyxyn[0][:, -1])
    #print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    #print(f"[INFO] Total {n} detections. . . ")
    #print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            #print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]


            if text_d == 'person':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2,1) ## BBox
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX, 0.9,(255,255,255), 2)
                bgr=(0,255,0)
                w=y2-y1
                W=70
                #d=15
                f=560 #focal length of my webcam
                #f=750
                #f=(w*d)/W
                d=(W*f)/(w)
                print(d)

                cvzone.putTextRect(frame,f'Distance:{int(d)}cm  ',(50,30),scale=2)
                #cv2.line(frame,(int(x1),int(y2)),(int((x1+x2)*0.5),int(y2)),(0,200,0),3)
                #cv2.line(frame,(int(x1),int(y1)),(int((x1+x2)*0.5),int((y1+y2)*0.5)),(0,200,0),3)



                #cv2.line(frame, (x1,y2),(x1,y1), (0, 200, 0), 3)
                #cv2.putText(frame, ang_label, (x2-29,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                #cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                

            elif text_d == 'no':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background

                
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))


    return frame


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):

    #print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    #model =  torch.hub.load('C:/Users/Desktop/yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format


    if img_path != None:
        #print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                #print(f"[INFO] Exiting. . . ")
                cv2.imwrite("final_output.jpg",frame) ## if you want to save he output result.

                break

    elif vid_path !=None:
        #print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) 
  
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file. 
            out = cv2.VideoWriter('C:/Users/lenovo/Desktop/distance estimation/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (width,height)) 

        
        
        # assert cap.isOpened()
        frame_no = 1
        # used to record the time when we processed last frame
        prev_frame_time = 0
  
        # used to record the time at which we processed current frame
        new_frame_time = 0
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret :
                #print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                fps = int(fps)

                framerate = "FPS :  {:.2f} ".format(fps)
                cv2.putText(frame, framerate, (50,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("vid_out", frame)
                if vid_out:
                    #print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                frame_no += 1
        #print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


main(vid_path="result.mp4",vid_out="result.avi") ### for custom video
#main(vid_path=0,vid_out="result.avi") #### for webcam
#main(img_path="test2.jpg") ## for image
