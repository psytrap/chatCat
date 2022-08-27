import cv2
import time
import matplotlib.pyplot as pyplot
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_hub as hub
#import keras_efficientnet_v2
import datetime
import queue, threading
import cv2.dnn
#import pygame
import requests #dependency
import argparse
import base64

IMAGE_FILE="./image001.jpg"
IMAGE_FILE="cat.jpg"


parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True, help='Discord webhook URL')

args = parser.parse_args()
URL = args.url


print(URL)
#options: 

# https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        ret, frame = self.cap.retrieve()
        return frame


def load_image_with_padding(image, dim):
    print(image.shape)
    max_dim = image.shape[1]
    min_dim = image.shape[0]
    delta = max_dim - min_dim
    print(delta)
    padding_bottom = round(delta/2)
    padding_top = delta - padding_bottom
    print(padding_top, padding_bottom)
    print(max_dim)

    # background = np.zeros((max_dim, max_dim, 3), np.uint8)
    padding_top    = np.zeros((padding_top,    max_dim, 3), np.uint8)
    padding_bottom = np.zeros((padding_bottom, max_dim, 3), np.uint8)

    image = np.append(padding_top, image, axis=0)
    image = np.append(image, padding_bottom, axis=0)
    # vis = np.append((image, background), axis=1)
    print(image.shape)
    width = dim
    height = dim
    
    #Load image by Opencv2
    #Resize to respect the input_shape
    image = cv2.resize(image, (width , height ), interpolation = cv2.INTER_CUBIC)
    
    #cv2.imshow(IMAGE_FILE, image)
    #cv2.waitKey(1000)
 
    #Convert img to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(rgb_image.shape)
    # COnverting to uint8
    rgb_tensor_image = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
    #rgb_tensor_image = tf.convert_to_tensor(rgb_image/256., dtype=tf.float32)
    print(rgb_tensor_image.shape)
    #Add dims to rgb_tensor
    rgb_tensor_image = tf.expand_dims(rgb_tensor_image , 0)
    print("dims", rgb_tensor_image.shape)
    return rgb_tensor_image

#detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d4/1")
#detector_output = detector(rgb_tensor_image)
#class_ids = detector_output["detection_classes"]
#print(class_ids)

#quit()


def load_keras_model():
    return EfficientNetV2B3()

def predict_keras_efficientnet_v2(model, rgb_tensor_image):
    predictions = model(rgb_tensor_image).numpy()
    raw_decoded = keras.applications.imagenet_utils.decode_predictions(predictions, 10)[0]
    # vprint(decoded)
    print("max", max(predictions[0]))
    decoded = dict()
    for i in raw_decoded:
       print(i[1:])
       decoded[i[1]] = i[2]
    prob = 0.
    for i in decoded:
       if "cat" in i:
          prob += decoded[i]
       elif "tabby" in i:
          prob += decoded[i]
       elif "tiger" in i:
          prob += decoded[i]
    return prob, decoded
    


# https://gist.github.com/Bilka2/5dd2ca2b6e9f3573e0c2defe5d3031b2
def discord_katze_anwesend(probabilities, image):
    url = URL
    message = "Katze anwesend" + "\n" + str(probabilities)
    image = cv2.resize(image, (int(image.shape[1]/12), int(image.shape[0]/12)))
    image_string = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1]).decode()
    message += "\n" + image_string
    print("len(message)", len(message))

    #for all params, see https://discordapp.com/developers/docs/resources/webhook#execute-webhook
    data = {
        "content" : message,
        "username" : "cURL Bot" # "pyWebhook Bot"
    }
    #leave this out if you dont want an embed
    #for all params, see https://discordapp.com/developers/docs/resources/channel#embed-object
    
    data["embeds"] = []
    
    result = requests.post(url, json = data)
    
    print(result.text)
    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print("Payload delivered successfully, code {}.".format(result.status_code))



print("keras_model")
print()
kerasModel = load_keras_model()

print("open camera")
video_capture_device_index = 0
webcam = VideoCapture(video_capture_device_index)

print("Prepare for prediction")

last_pic = datetime.datetime.now()
cat = False
first_iteration = True
cat_last_time = datetime.datetime.now()
frame = webcam.read()
time.sleep(3)

while True:
    frame = webcam.read()
    # image = cv2.imread(filename, cv2.IMREAD_COLOR)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    today = datetime.datetime.now().isoformat(timespec='seconds')
    print(today)
    rgb_tensor_image = load_image_with_padding(frame, 300)
    # archive image
    start = datetime.datetime.now()
    [probability, decoded] = predict_keras_efficientnet_v2(kerasModel, rgb_tensor_image)
    delta = datetime.datetime.now() - start
    print("Processing in seconds:", delta.total_seconds())

    print("Decoded:", decoded)
    if last_pic + datetime.timedelta(seconds=20) < datetime.datetime.now():
        isWritten = cv2.imwrite('archive/image-'+today+'.jpg', frame)
        last_pic = datetime.datetime.now()
    print("Probably a cat:", 100. * probability, "%")
    if probability < 0.1 and first_iteration is False:
        cat_delta = datetime.datetime.now() - cat_last_time
        if cat_delta.total_seconds() > 300:
            cat = False
    else:
        cat_last_time = datetime.datetime.now()
        if cat is False or first_itertion is True:
            cat = True
            first_iteration = False
            print("discord")
            discord_katze_anwesend(decoded, frame)

    #if result is True:
    #    pygame.mixer.music.play()
    time.sleep(10)



