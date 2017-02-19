
import numpy as np
import cv2
import matplotlib.pylab as plt
from keras.preprocessing.image import load_img
from keras.models import model_from_json
from models import (
	create_cam_model, preprocess_image, 
	get_cam_img
)

# Define CAM conv layer name
CAM_CONV_LAYER = 'cam_conv_layer'


def read_model(model_path, weigths_path):
	"""Load your pretrained model
	"""
	model = model_from_json(open(model_path).read())
	model.load_weights(weigths_path)
	return model

def train_cam_model(model, X_train, Y_train, X_test, Y_test, 
					batch_size, nb_epoch):
	"""Train CAM model based on your pretrained model

	# Arguments
		model: your pretrained model, CAM model is trained based on this model.

	"""

	# Use your allready trained model
	pretrained_model_path = ''
	pretrained_weights_path = ''

	# Your pretrained model name
	pretrained_model_name = 'VGG16'

	# Label class num
	num_classes = 10

	# CAM input spacial size
	gap_spacial_size = 14

	# The layer before CAM(GAP) layers.
	# CAM paper suggests to use the last convnet(VGG) or mergenet(Inception, or other architectures)
	# Change this name based on your model.
	if pretrained_model_name == 'VGG16':
		in_layer_name = 'block5_conv3'
	elif pretrained_model_name == 'InceptionV3':
		in_layer_name = 'batchnormalization_921'
	elif pretrained_model_name == 'ResNet50':
		in_layer_name = 'merge_13'
	else:
		in_layer_name = ''

	# Load your allready trained model, transfer it to CAM model
	pretrained_model = read_model(pretrained_model_path, 
								  pretrained_weights_path)

	# Create CAM model based on trained model
	model = create_cam_model(pretrained_model,
							 gap_spacial_size,
							 num_classes,
							 in_layer_name,
							 CAM_CONV_LAYER)

	# Train your CAM model
	model.compile(loss='categorical_crossentropy',
			  	  optimizer='adadelta',
			  	  metrics=['accuracy'])
	model.fit(X_train, Y_train, 
			  batch_size=batch_size, 
			  nb_epoch=nb_epoch,
			  shuffle=True, verbose=1, 
			  validation_data=(X_test, Y_test))

	# Save model
	model.save_weights('')
	return model

def cam_model():
	"""
	Return your trained CAM model
	"""
	return

def plot_cam_map(img_path, img_size, batch_size, label_plot):
	"""Plot class activation map.

	"""
	
	# CAM input spacial size
	gap_spacial_size = 14

	# Use your trained CAM model
	model = cam_model()

	# Load and format data
	im_ori = np.asarray(load_img(img_path, target_size=(img_size, img_size)))
	test_data = preprocess_image(img_path, img_size, expand_dims=True)

	# Get class map image
	im_cam = get_cam_img(model,
						 test_data,
						 label_plot,
						 ratio=img_size / gap_spacial_size)

	# Resize if the shape of class map is not equal to original image
	if im_cam.shape != im_ori[:, :, 0].shape:
		im_cam = cv2.resize(im_cam, (img_size, img_size), cv2.INTER_LINEAR)
	
	# Show the predictions. You can analyze the class map with the predictions.
	prediction_labels = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=1)
	print('Info: Predictions:\n{}'.format(prediction_labels))

	# Plot original image and the class map
	plt.imshow(im_ori)
	plt.imshow(im_cam,
			   cmap='jet',
			   alpha=0.5,
			   interpolation='bilinear')
	plt.show()
