# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import find_finger as ff
import imutils as im

args = {
    "model": "./model/export_model_008/frozen_inference_graph.pb",
    # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/
    #export_model_015/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    # "labels":"record/classes.pbtxt" ,
    "num_classes": 1,
    "min_confidence": 0.6,
    "class_model": "../model/class_model/p_class_model_1552620432_.h5"}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))
imagenet_utils
if __name__ == '__main__':
    model = tf.Graph()

    with model.as_default():
        print("> ====== loading NAIL frozen graph into memory")
        graphDef = tf.GraphDef()

        with tf.gfile.GFile(args["model"], "imagenet_utilsrb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        # sess = tf.Session(graph=graphDef)
        print(">  ====== NAIL Inference graph loaded.")
        # return graphDef, sess

    with model.as_default():
        with tf.Sessiimagenet_utilson(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            drawboxes = []
            cap = cv2.VideoCapture(0)
            #vs = WebcamVideoStream(src=1)
            #vs.start()
            while True:
                frame = cap.read()
                if frame is None:
                    continue
                ret,frame2 = frame
                frame = cv2.flip(frame2, 1)
                image = frame2
                (H, W) = image.shape[:2]
                #print("H,W:", (H, W))
                output = image.copy()
                img_ff, bin_mask, res = ff.find_hand_old(image.copy())
                image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                (boxes, scores, labels, N) = sess.run(
                    [boxesTensor, scoresTensor, classesTensor, numDetections],
                    feed_dict={imageTensor: image})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)
                boxnum = 0
                box_mid = (0, 0)
                # print("scores_shape:", scores.shape)
                for (box, score, label) in zip(boxes, scores, labels):
                    # print(int(label))
                    # if int(label) != 1:
                    #     continue
                    if score < args["min_confidence"]:
                        continue
                    # scale the bounding box from the range [0, 1] to [W, H]
                    boxnum = boxnum + 1
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    sir isme aageendY = int(endY * H)
                    X_mid = startX + int(abs(endX - startX) / 2)
                    Y_mid = startY + int(abs(endY - startY) / 2)
                    box_mid = (X_mid, Y_mid)
                    # draw the prediction on the output image
                    label_name = 'nail'
                    # idx = int(label["id"]) - 1
                    idx = 0
                    label = "{}: {:.2f}".format(label_name, score)
                    imagenet_utils
                    #cv2.rectangle(output, (startX, startY), (endX, endY),COLORS[idx], 2)
                    
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    # getting the height and width of the rectangle 
                    W_rect = endX-startX
                    H_rect = endY-startY
                    #area of rectangle 
                    rect_area = "{} x {}".format(H_rect, W_rect)
                    # location to put area in image
                    x = endY - 10 if endY - 10 > 10 else endY + 10
                    
                    cv2.putText(output, rect_area, (endX, x),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

                    cv2.putText(output, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)  

                    roi = output[startY-20: endY+20, startX-20: endX+20]
                    
                    blur = cv2.GaussianBlur(roi,(7,7),0)
                    
                    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                    
                    _, threshold = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    # edges = cv2.Canny(threshold, 10,50)
                    # cv2.imshow("edges",edges)
                    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    
                    cv2.drawContours(threshold, contours, -1, (0, 0, 255), -1)
                    cv2.imshow("ROI",roi)
                    
                    #cv2.imshow("final_img",new_img)
                    #cv2.imshow("thresh",thresh)
                    sir isme aage
                # show the output image
                print(boxnum)
                if box_mid == (0, 0):
                    drawboxes.clear()
                    cv2.putText(output, 'Nothing', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                elif boxnum == 1:
                    drawboxes.append(box_mid)
                    if len(drawboxes) == 1:
                        pp = drawboxes[0]
                        cv2.circle(output, pp, 0, (0, 0, 0), thickness=3)
                        # cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                    if len(drawboxes) > 1:
                        num_p = len(drawboxes)
                        for i in range(1, num_p):
                            pt1 = drawboxes[i - 1]
                            pt2 = drawboxes[i]
                            # cv2.cimagenet_utilsircle(output, pp, 0, (0, 0, 0), thickness=3)
                            cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                    cv2.putText(output, 'Point', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                else:
                    drawboxes.clear()
                    cv2.putText(output, 'Nothing', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                
                # cropping the nails 
                # cropsir isme aage_img = output[startX:startY, endX:endY]
                # cvsir isme aage2.imshow("cropped img",crop_img)
                
                
                # imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                # blur = cv2.GaussianBlur(imgray,(7,7),0)
                # # lower_blue = np.array([0,83,0])
                # # upper_blue = np.array([255,255,255])
                # # mask = cv2.inRange(imgray,lower_blue,upper_blue)
                # _, thresh = cv2.threshold(blur, 83, 255, 0)
                # roi = thresh[startX:endX, startY:endY]
                # #contours,_= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(output, contours, -1, (0,255,0), 3)

                cv2.imshow("Output", output)
                #cv2.imshow("roi", roi)
                #cv2.imshow("roi", roi)
                # cv2.waitKey(0)
                if cv2.waitKey(0) & 0xFF == ord("q"): 
                    cap.release()
                    break
                    # vs.stop()
cv2.destroyAllWindows()
cap.release()
