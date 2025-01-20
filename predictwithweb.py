# USAGE python predictwithweb.py --model models/output_model_50.h5 --labels retinanet_classes.csv --confidence 0.5

from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to class labels")
#ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split("\n")
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}
print("labels ",LABELS)

model = models.load_model(args["model"], backbone_name="resnet50")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cap = cv2.VideoCapture("video5.mp4")#0 if not then 1 for usb
while True:
    _,frame=cap.read()
    '''
    (h,w) = frame.shape[:2]
    (CX,CY)=(w/2,h/2)
    M=cv2.getRotationMatrix2D((CX,CY),180,1.0)
    frame=cv2.warpAffine(frame,M,(w,h))
    '''
    frame=imutils.resize(frame,width=800)
    image=frame
    output = frame.copy()
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        if score < args["confidence"]:
            continue

        box = box.astype("int")
        
        label = "{}: {:.2f}".format(LABELS[label], score)
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                (0, 255, 0), 2)
        cv2.putText(output, label, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break