#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str
import facedetect as fd

import Train_Common as tc

help_message = '''
USAGE: facedRec_LBP.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>]  [--model-name <ModelName>] [--label-name <LabelName>]
'''



out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

if __name__ == '__main__':
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade=', 'model-name=', 'label-name='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")
    ModelName = args.get('--model-name', "Model_default.yml")
    LabelName = args.get('--label-name', "Label_default.txt")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    
    reconizer = cv2.createLBPHFaceRecognizer()
    reconizer.load(ModelName)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
    Name, Labels = tc.Read_List_Label(LabelName)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = fd.detect(gray, cascade)
        vis = img.copy()
        fd.draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = fd.detect(roi.copy(), nested)
            fd.draw_rects(vis_roi, subrects, (255, 0, 0))
            
            label, distance = reconizer.predict(roi)
            
            draw_str(vis, (x1, y2+15), '%.1f' % (distance))
            if(distance < 300):
                draw_str(vis, (x1, y2), '%s' % (Name[label]))
            else:
                draw_str(vis, (x1, y2), '%s' % ("Known H"))
            
        dt = clock() - t

        
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)
        
        #cv2.imwrite("x121_0.png", vis)
        
        # write the flipped frame
        out.write(vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
            
    cv2.destroyAllWindows()
    cam.release()
    out.release()
