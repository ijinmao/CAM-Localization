import scipy.ndimage as ndimage
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img

def preprocess_image(path, size, expand_dims=False):
	img = load_img(path, target_size=(size, size))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	if not expand_dims:
		img = img.reshape(img.shape[1:])
	return img

def model_cam(model_input,
			  gap_input,
			  gap_spacial_size,
			  num_classes,
			  cam_conv_layer_name):
	"""Build CAM model architecture
	
	# Arguments
		model_input: input tensor of CAM model
		gap_input: input tensor to cam gap layers
		gap_spacial_size: average pooling size
		cam_conv_layer_name: the name of new added conv layer

	"""
	x = Convolution2D(1024, 3, 3, 
					  activation='relu', 
					  border_mode='same',
					  name=cam_conv_layer_name)(gap_input)
	# Add GAP layer
	x = AveragePooling2D((gap_spacial_size, gap_spacial_size))(x)
	x = Flatten()(x)
	predictions = Dense(num_classes, activation='softmax')(x)
	model = Model(input=model_input, output=predictions)
	return model

def create_cam_model(pretrained_model,
					 gap_spacial_size,
					 num_classes,
					 in_layer_name,
					 cam_conv_layer_name):
	"""Create CAM model

	# Arguments
		pretrained_model: your pretrained model
		gap_spacial_size: average pooling size
		num_classes: the number of labels class
		in_layer_name: the layer name before new added layer of CAM model
		cam_conv_layer_name: the name of new added conv layer

	"""
	# The last convnet(vgg) or mergenet(inception, other architectures) layer output
	gap_input = layer_output(pretrained_model, in_layer_name)
	model = model_cam(pretrained_model.input, 
					  gap_input,
					  gap_spacial_size, 
					  num_classes,
					  cam_conv_layer_name)

	# Fix pretrained model layers, only train new added layers
	for l in pretrained_model.layers:
		l.trainable = False
	return model

def layer_output(model, layer_name=None):
	"""Output tensor of a specific layer in a model.

	"""
	conv_index = -1

	for i in range(len(model.layers) - 1, -1, -1):
		layer = model.layers[i]
		if layer_name in layer.name:
			conv_index = i
			break

	if conv_index < 0:
		print('Error: could not find the interested layer.')

	return model.layers[conv_index].output

def index_layer(model, name):
	"""Index of layer in one model.

	"""
	for i in range(len(model.layers) - 1, -1, -1):
		layer = model.layers[i]
		if name in layer.name:
			return i

def get_cam_img(model, X, label, 
				cam_conv_layer_name, 
				ratio=1):
	"""Get class map image.

	# Arguments:
		model: Trained CAM model
		X: test image array
		label: which label you want to visualize
		cam_conv_layer_name: the name of new added conv layer
		ratio: upsampling ratio

	"""
	inc = model.input

	# The activation of conv before gap layer
	# f_k(x, y), for VGG16, the shape is (1, 14, 14, 1024)
	conv_index = index_layer(model, cam_conv_layer_name)
	conv_output = model.layers[conv_index].output

	# Project the conv output to image space
	resized_output = K.resize_images(conv_output, ratio, ratio, 'tf')

	# The weights of GAP layer to softmax layer(ignore bias), the shape is (1024, num_classes)
	weights = model.layers[-1].weights[0]

	# Get the weighted conv activations
	classmap = K.dot(resized_output, weights)

	# Define the function
	get_cmap = K.function([K.learning_phase(), inc], [classmap])

	# Get class map in image space
	im_cam = get_cmap([0, X])[0]

	# Only show the positive activations
	im_cam = np.clip(im_cam, 0, im_cam.max())

	# Use the label that you want to visualize
	im_cam = im_cam[0, :, :, label]
	print('Info: Class map shape:', im_cam.shape)

	# Blur the map
	im_cam = ndimage.gaussian_filter(im_cam, sigma=(5), order=0)

	return im_cam
