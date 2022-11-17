# README

**Object Detection using PyTorch Faster RCNN ResNet50 FPN V2 trained on PPE datasets**

PyTorch recently released an improved version of the Faster RCNN object detection model. They call it the Faster RCNN ResNet50 FPN V2. This model is miles ahead in terms of detection quality compared to its predecessor, the original Faster RCNN ResNet50 FPN. In this repo, we will discover what makes the new Faster RCNN model better, why it is better, and what kind of detection results we can expect from it.
To improve the Faster RCNN ResNet50 (to get the V2 version) model, changes were made to both:

* The ResNet50 backbone recipe
* The object detection modules of Faster RCNN

### Let’s check out all the points that we will cover in this post:

* We will fine-tune the Faster RCNN ResNet50 FPN V2 model in this post.
* For training, we will use a PPE detection dataset.
* After training, we will analyze the mAP and loss plots.
* We will also run inference on videos to check how the model performs in real-world scenarios.

### The PPE Detection Dataset
To train the object detection model in this post, we will use the COVID-19 PPE Dataset for Object Detection from Kaggle (https://www.kaggle.com/datasets/ialimustufa/object-detection-for-ppe-covid19-dataset).

The dataset contains images of medical personnel wearing PPE kits for the COVID-19 pandemic. It consists of 366 training images and 50 test images across 5 classes. Also, it is worthwhile to note that the dataset is structured in the Pascal VOC XML format.

The class names given on the Kaggle page and those in the XML files slightly mismatch. As we will be using the class names from the XML files, let’s take a look at them:

Mask, Face_Shield, Coverall, Gloves, Goggles

### Pretraining ResNet50 Backbone

Pretraining the ResNet50 backbone is an essential task in improving the performance of the entire object detection model. The ResNet50 (as well as many other classification models) model was trained with a new training recipe. These include, but are not limited to:

* Learning rate optimizations.
* Longer training.
* Augmentation such as TrivialAugment, Random Erasing,  MixUp, and CutMix.
* Repeated Augmentation
* EMA
* Weight Decay Tuning

With these new techniques, the ResNet50 Accuracy@1 jumps to 80.858% from the previous 76.130%.

Training the Faster RCNN ResNet50 FPN V2 Model
As mentioned earlier, most of the improvements to train the entire object detection model were taken from the aforementioned paper.

The contributors to these improvements call these improvements as per post-paper optimization. These include:

* FPN with batch normalization.
* Using two convolutional layers in the Region Proposal * Network (RPN) instead of one. In other words, using a heavier FPN module.
* Using a heavier box regression head. To be specific, using four convolutional layers with Batch Normalization followed by linear layer. Previously, a two layer MLP head without Batch Normalization was used.
* No Frozen Batch Normalizations were used.

Using the above recipe improves the mAP from the previous 37.0% to 46.7%, a whopping 9.7% increase in mAP.

## Directory Structure

* The data folder contains the dataset for this project in the data/ppe folder. The train and test folders contain the images along with the XML files.
* The data_configs directory contains the dataset information and configuration. We will look at it at a later stage in the post.

We have the script for two models in the models directory. One is for the older Faster RCNN ResNet50 FPN and the other is for the FPN V2 one. We can use either one of them during training just by changing one command line flag value.

* The outputs directory will hold all the training outputs.

We have a torch_utils and a utils directory which holds a lot of helper code and training utilities. We will not be diving into the details of these here. But you are free to check them out.

* There are two inference scripts as well, one for image inference and one for video inference. Along with that, we have the datasets.py for preparing the dataset and data loaders. The train.py is the executable script to start the training.

#### Train the Faster RCNN Model

`python train.py --model fasterrcnn_resnet50_fpn_v2 --config data_configs/ppe.yaml --epochs 50 --project-name fasterrcnn_resnet50_fpn_v2_ppe --use-train-aug --no-mosaic`

The following are the command line arguments that we use:

* --model: Here, we use fasterrcnn_resnet50_fpn_v2 to indicate that we want to train the new Faster RCNN model. To train the older model, you can simply provide the value as fasterrcnn_resnet50_fpn.
* --config: It takes the path to the dataset configuration file which is data/ppe.yaml file in this case.
* --epochs: We are training the model for 50 epochs.
* --project-name: Providing a string value to this argument will save the results with that folder name inside outputs/training. In this case, it is going to be outputs/training/fasterrcnn_resnet50_fpn_v2_ppe.
* --use-train-aug: The data loader supports various image augmentations. This argument is a boolean value that ensures that those augmentations are applied.
* --no-mosaic: Additionally, the data loader also supports mosaic augmentation. But providing this boolean argument will turn it off.

You can also get the trained model here: https://drive.google.com/drive/folders/15j7tPtCfyhb2yhN4ovS6pHag8ARFt2gL?usp=sharing which will be then placed in a folder training inside the outputs folder.

#### Executing detect_video.py for Video Inference

`python inference_video.py --weights outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/best_model.pth --input data/inference_data/video_1.mp4 --show-image --threshold 0.9 /`

The following are the command line arguments:

* --weights: Path to the weights file. We are using best-trained weights here.
* --input: Path to the source video. You may give the path to your own videos as well.
* --show-image: This tells the script that we want to visualize the results on the screen.
* --threshold: We are using a confidence threshold of 90% for the visualizations.

## Important Links

* https://github.com/pytorch/vision/pull/5763
* https://github.com/pytorch/vision/pull/5444
* https://github.com/pytorch/vision/issues/5307
* https://github.com/pytorch/vision/issues/3995
* Improving the Backbones - How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives => https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
* Direct Link to new ResNet50 Training recipe.

## Summary and Conclusion

In this repo post, we set up an entire pipeline for training the PyTorch Faster RCNN ResNet50 FPN V2 object detection model. Although we were not able to achieve the best fine tuning results, we will surely do so in the future. I hope that you learned something new from this tutorial.