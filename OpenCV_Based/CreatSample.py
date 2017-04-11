#!/usr/bin/env python

import os
import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str

import facedetect as fd

help_message = '''
USAGE: CreatSample.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [--outdir <OutDir>] [--sample-num <SampleNum>]
'''

CntLimit = 10
CntSample = 0

if __name__ == '__main__':
    import sys, getopt
    print (help_message)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade=','outdir=','sample-num='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")
    out_dir = args.get('--outdir', "./sample")
    CntLimit = args.get('--sample-num', 10)
    CntLimit = int(CntLimit) # args.get only return string, so we need to covert it to digit
    
    print ("SampleNum: " + str(CntLimit))
    print ("out dir: " + out_dir)
    
    if not os.path.exists(out_dir):
        print ("create out_dir")
        os.makedirs(out_dir)

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

    while True:    
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = fd.detect(gray, cascade)
        vis = img.copy()
        fd.draw_rects(vis, rects, (0, 255, 0))
        #fd.draw_rects_1(vis, rects, (255, 255, 0))
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = fd.detect(roi.copy(), nested)
            fd.draw_rects(vis_roi, subrects, (255, 0, 0))
            #print len(subrects)
            if len(subrects) == 2:
                print ("imwrite Sample " + str(CntSample))
                dir = out_dir + "/sample_" + str(CntSample) + ".png"
                cv2.imwrite(dir, roi)
                CntSample = CntSample+1
            
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)
        
        if CntSample >= CntLimit:
            print ("finish")
            break
        
        
        
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
