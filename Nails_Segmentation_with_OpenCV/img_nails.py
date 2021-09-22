import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
#from imutils.video import WebcamVideoStream
import find_finger as ff
import cv2

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

if __name__ == '__main__':
    model = tf.Graph()

    with model.as_default():
        print("> ====== loading NAIL frozen graph into memory")
        graphDef = tf.GraphDef()

        with tf.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        # sess = tf.Session(graph=graphDef)
        print(">  ====== NAIL Inference graph loaded.")
        # return graphDef, sess

    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            drawboxes = []
            frame = cv2.imread("nails/b.png")
            if frame.shape[0] <=800 or frame.shape[1] <=800:
                print("shape is low to resize the image")
            else:
                scale_percent = 20 # percent of original size
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)
            image = frame
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
                endY = int(endY * H)
                X_mid = startX + int(abs(endX - startX) / 2)
                Y_mid = startY + int(abs(endY - startY) / 2)
                box_mid = (X_mid, Y_mid)
                # draw the prediction on the output image
                label_name = 'nail'
                # idx = int(label["id"]) - 1
                idx = 0
                label = "{}: {:.2f}".format(label_name, score)
                
                #cv2.rectangle(output, (startX, startY), (endX, endY),COLORS[idx], 2)
                
                y = startY - 10 if startY - 10 > 10 else startY + 10

                # getting the height and width of the rectangle 
                W_rect = endX-startX
                H_rect = endY-startY
                #area of rectangle 
                rect_area = "{} x {}".format(H_rect, W_rect)
                # location to put area in image
                x = endY - 10 if endY - 10 > 10 else endY + 10                
            #cv2.putText(output, rect_area, (endX, x),cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
            #cv2.putText(output, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)  
                
                
                
                '''
                def auto_canny(roi, sigma=0.33):                 	
                 	v = np.median(roi)                 	
                 	lower = int(max(0, (1.0 - sigma) * v))
                 	upper = int(min(220, (1.0 + sigma) * v))
                 	edged = cv2.Canny(roi, lower, upper)
                 	return edged
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
               	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
               	auto = auto_canny(blurred)
                ht = cv2.Canny(blurred, 200, 255)
                
                mask = np.zeros(roi.shape,dtype='int8')
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
                threshed = cv2.morphologyEx(blurred,cv2.MORPH_CLOSE, rect_kernel)
                wide = cv2.Canny(threshed,50, 150)
                contours, h = cv2.findContours(wide,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # get convex hull
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(roi, [hull], -1, (0, 0, 255), -1) 
                '''
                
                roi = output[startY: endY, startX: endX]
                
                def find_if_close(cnt1,cnt2):
                    row1,row2 = cnt1.shape[0],cnt2.shape[0]
                    for i in range(row1):
                        for j in range(row2):
                            dist = np.linalg.norm(cnt1[i]-cnt2[j])
                            if abs(dist) < 25 :
                                return True
                            elif i==row1-1 and j==row2-1:
                                return False                
                
                gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                
                blurred =  cv2.bilateralFilter(gray, 3, 40, 40) 
                thresh = cv2.Canny(blurred,80, 150)
                contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)                
                
                LENGTH = len(contours)
                status = np.zeros((LENGTH,1))
                
                for i,cnt1 in enumerate(contours):
                    x = i
                    if i != LENGTH-1:
                        for j,cnt2 in enumerate(contours[i+1:]):
                            x = x+1
                            dist = find_if_close(cnt1,cnt2)
                            if dist == True:
                                val = min(status[i],status[x])
                                status[x] = status[i] = val
                            else:
                                if status[x]==status[i]:
                                    status[x] = i+1
                
                unified = []
                maximum = int(status.max())+1
                for i in range(maximum):
                    pos = np.where(status==i)[0]
                    if pos.size != 0:
                        cont = np.vstack(contours[i] for i in pos)
                        hull = cv2.convexHull(cont)
                        unified.append(hull)
                
                cv2.drawContours(roi,unified,-1,(0,255,0),1)
                cv2.drawContours(thresh,unified,-1,255,2)
                
                # cv2.imshow("wide", wide)
                cv2.imshow("original", roi)
                cv2.imshow("thresh", thresh)
                
            print(boxnum)
            
            
            
            '''if box_mid == (0, 0):
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
                        # cv2.circle(output, pp, 0, (0, 0, 0), thickness=3)
                        cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                cv2.putText(output, 'Point', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            else:
                drawboxes.clear()
                cv2.putText(output, 'Nothing', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            '''
            cv2.imshow("Output", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        