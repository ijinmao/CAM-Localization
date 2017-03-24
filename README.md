# CAM-Localization

## References

Implementation of the paper [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf).

Learn more about this paper and the original matlab implementation [here](http://cnnlocalization.csail.mit.edu/).

Part of codes is based on [VGG16CAM-keras](https://github.com/tdeboissiere/VGG16CAM-keras). This repo implements VGG16-CAM model with keras in Theano backend.

## Requirements

- keras with tensorflow
- numpy
- matplotlib
- opencv-python
- scipy

## Usage

View `demo.py`. 

First you need to convert your pretrained model to CAM-model. And then train the new model on your data. See the sample function `train_cam_model` in `demo.py`.

After trianning the CAM-model, you can use the input features of GAP layer and the weights of last classifier to generate final output. See the sample function `plot_cam_map` in `demo.py`.



## Examples

Below image is from [kaggle competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring).



