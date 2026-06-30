# Implementation of a Model for Detecting 9 Vehicle Classes

## Sources

* data/processed_300/data_300.csv: CSV containing the images and bounding boxes
* data/processed_300/imgs/: images

## Objective

The main objective is to develop an artificial intelligence–based solution capable of detecting and classifying vehicles at urban intersections from images.

## CSV Structure

The train.csv file contains the training annotations.

### It has two columns

Id,Target
Where:

* Id is the filename without extension.
* Target contains the annotations for the corresponding frame.

For example, if the image v_ab12cd34ef_0000.jpg exists in train.zip, then the corresponding identifier in train.csv will be v_ab12cd34ef_0000.

### Target Column Format in train.csv

Each annotated object is represented as an oriented bounding box using the following format:

category_id cx cy width height angle_deg

Where:

* category_id is the class identifier.
* cx is the x-coordinate of the box center.
* cy is the y-coordinate of the box center.
* width is the width of the box.
* height is the height of the box.
* angle_deg is the rotation angle of the box, in degrees.

All coordinates are expressed in pixels. If a frame contains multiple objects, they are separated by a semicolon (;).
(There can be more than one annotation in a frame, or none, represented by 'none' in the Target field.)

Example:

Id,Target
v_ab12cd34ef_0000,"1 987.86 598.84 48.84 94.88 339.94;9 1236.10 506.05 39.07 29.30 0.00"

If the model does not predict any object for a frame, the Target field must contain **none**. Empty cells in the Target column are not allowed.

## Official Classes

The official categories are:

* ID 1: Car
* ID 2: Van
* ID 3: Microbus
* ID 4: Minibus
* ID 5: Bus
* ID 6: Articulated Bus
* ID 7: Truck
* ID 8: Mototaxi
* ID 9: Motorcycle

### Prediction Format

For each frame in the test set, the participant must submit a set of detections in the Target column.

Each prediction must follow this format:

score category_id cx cy width height angle_deg

Where:

* score: detection confidence, between 0 and 1.
* category_id: class identifier, between 1 and 9.
* cx: x-coordinate of the predicted box center.
* cy: y-coordinate of the predicted box center.
* width: width of the predicted box.
* height: height of the predicted box.
* angle_deg: rotation angle of the predicted box, in degrees.

Multiple detections for the same frame must be separated using a semicolon (;).

Example:

0.93 1 987.86 598.84 48.84 94.88 339.94;0.81 9 1236.10 506.05 39.07 29.30 0.00

If the model does not predict any object for a frame, the Target field must contain exactly **none**. Empty cells in the Target column are not allowed.
