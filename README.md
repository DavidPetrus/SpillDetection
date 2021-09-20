# Spill Detection


## Setup

Requires the following to be installed:
```
Pytorch and torchvision
CLIP (https://github.com/openai/CLIP)
absl
OpenCV
```

## Usage
A trained model can be run on a video using the following command:
```
python3 run_spill_det.py --model_weights=weight_file.pt  --video=video.avi
```

The weight_file.pt should be put in the weights folder and the video should be put in the input_videos folder. An output video will be generated and saved in the output_videos folder.

## Hyperparameter settings
The following hyperparameters must be specfied: Crop dimensions and the threshold values.

Crop dimensions is specified on line 50 in run_spill_det.py. Example settings are given below:
```
crop_dims = [(3,5),(4,7)] # When running on a typical security camera.
crop_dims = [(2,1),(3,1)] # When running on a mobile camera showing close-up spills
```

The optimal threshold value is affected by the following factors:
(1) How large/small are the spills compared to the crop_size.
(2) What does the background look like and what objects does it contain.

The following thresholds should work well for weight_file=17Sep7.pt
```
# Security camera in store:
potential_thresh = 0.17
spill_thresh = 0.19

# Close-up on mobile in pool:
potential_thresh = 0.2
spill_thresh = 0.255
```

Lines 119,123 and 127 can be uncommented in order to view the spill similarity values when running the spill_detector.

TODO:
An easy way to automatically determine a good threshold for each scene is to have a short (5-10 sec) calibration of the scene containing no spills.