from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
import argparse
import numpy as np

file = open("food/food.txt")
food_keywords = file.read()


#argument to pass image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")

args = vars(ap.parse_args())

inputShape = (299, 299)
preprocess = preprocess_input

# load our the network weights from disk
print("[INFO] loading InceptionV3 model...")

Network = InceptionV3
model = Network(weights="imagenet")


# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through thenetwork
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with InceptionV3...")
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

#store the labels
labels = []

'''
# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
	labels.append("{}".format(label.replace("_", " ")))
'''

for label in labels:
	if (label in food_keywords):
		print("Food found!")
		break
