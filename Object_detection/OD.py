import cv2
import numpy as np
from MVN_model import MVN






# OD model information
yolo = cv2.dnn.readNetFromONNX(r"Object_detection\best.onnx")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
INPUT_IMG_SIZE = 224

class_name = ["Non-Meatal", "Steel", "Copper", "Rusted", "Iron", "Aluminium"]
class_color = [(0,0,139), (255,0,0), (0,255,0), (255,165,0), (162,59,236), (150,75,0)]





#5
def get_value(val, image_width):
    return int((image_width * val) / 4000)





#4
def get_OD_array(image_array, image_width, indexes, boxes):
    helper = image_array.copy()
    for index in indexes:
        x, y, w, h = [(x*-1) if x < 0 else x for x in boxes[index]]
        MVN_output = MVN.predict(helper[y:y+h, x:x+w])
        prob_index = np.argmax(MVN_output[0])
        
        value = MVN_output[0][prob_index]
        text = class_name[prob_index]
        color = class_color[prob_index]
        bb_conf = int(value*100)
        
        text=f'{text}: {bb_conf}%'
        cv2.rectangle(image_array, (x,y), (x+w,y+h), color, get_value(15, image_width))
        cv2.rectangle(image_array, (x-get_value(10, image_width), y-get_value(100, image_width)),
                      (x+w+get_value(10, image_width), y), color, -1)
        cv2.putText(image_array, text, (x+get_value(10, image_width), y-get_value(20, image_width)),
                    cv2.FONT_HERSHEY_PLAIN, get_value(5, image_width),(255,255,255), get_value(6, image_width))
    return image_array





#3
def get_optimize_prediction(output, image_width, image_height):
    x_factor = (image_width / INPUT_IMG_SIZE)
    y_factor = (image_height / INPUT_IMG_SIZE)
    
    predictions = output[0]
    boxes = []
    confidences = []

    for i in range (len(predictions)):
        confidence = predictions[i][4]
        if confidence > 0.5:
            cx, cy, w, h = predictions[i][:4]
            
            left = int ((cx-0.5*w)*x_factor)
            top = int ((cy-0.5*h)*y_factor)
            width = int (w*x_factor)
            height = int (h*y_factor)
            
            box = np.array([left,top,width,height])
            confidences.append(confidence)
            boxes.append(box)
            
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    if len(boxes_np) != 0:
        indexes = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.5).flatten()
        return indexes, boxes_np
    return [] , []





#2
def get_prediction(img_array):
    blob = cv2.dnn.blobFromImage(img_array, 1/255, (INPUT_IMG_SIZE,INPUT_IMG_SIZE), crop=False)
    yolo.setInput(blob)
    output = yolo.forward()
    output = np.transpose(output, (0, 2, 1))
    return output





#1
def detect_object(rgb_array):
    image_height, image_width, channel = rgb_array.shape
    output = get_prediction(rgb_array)
    indexes, boxes = get_optimize_prediction(output, image_width, image_height)
    if len(indexes) == 0:
        return rgb_array
    OD_detected_array = get_OD_array(rgb_array, image_width, indexes, boxes)
    return OD_detected_array