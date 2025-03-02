import cv2
import numpy as np





# Preprocessing for MVN model
mean_image = cv2.imread(r"MVN_model\mean_image.png")
mean_image = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)
mean_image = np.array([mean_image]).astype("float32")
def crop_center(img, crop_width, crop_height):
    height, width = img.shape[:2]
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    cropped_img = img[start_y:end_y, start_x:end_x]
    return cropped_img
def preprocess(rgb_array):
    global mean_image
    img = cv2.resize(rgb_array, (256,256))
    img = crop_center(img, 227, 227)
    img = np.array([img]).astype("float32")
    img = img - mean_image
    return img





# MVN model information
model = cv2.dnn.readNetFromONNX(r"MVN_model/MVN-35-0.7969.onnx")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)





def predict(rgb_array):
    rgb_array = preprocess(rgb_array)
    model.setInput(rgb_array)
    return model.forward()