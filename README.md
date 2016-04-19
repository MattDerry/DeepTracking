DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks
==================================================================
This is an *un-official* Keras 1.0 implementation of the method for the end-to-end object tracking from occluded sensor measurements using neural network presented in the academic paper:

[P. Ondruska and I. Posner, *"Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks"*, in The Thirtieth AAAI Conference on Artificial Intelligence (AAAI), Phoenix, Arizona USA, 2016.](http://www.robots.ox.ac.uk/~mobile/Papers/2016AAAI_ondruska.pdf)

* **author**: Matt Derry
* **email**: mderry(at)gmail.com

For any questions about the code or the method please contact the author.

Installation
------------
Install [Keras](http://keras.io/) and the following dependencies:
* tensorflow or theano
* numpy
* scipy
* cudnn (optional for training on a GPU)

Data
----
Download and unzip the training data for the simulated moving balls scenario:
```
http://mrg.robots.ox.ac.uk:8080/MRGData/deeptracking/DeepTracking_1_1.t7.zip
```
This is a native Torch 7 file format, but it will be converted by the software

Training
--------
To train the model run:
```
python train.py
```

Training of the neural network using provided data takes about 12 hours on Nvidia Titan X. Every 1000 iterations the training error is logged to *log_model.txt*, network weights are saved to *weights_model* and the visualisation of its performance is stored to *video_model*.

#### Optional parameters
Flag                                      | Description
------------------------------------------|----------------------------------
-gpu [id]                                 | use GPU [id] (0 to use CPU)
-model [file]                             | neural network model
-data [file]                              | data for training
-iter [number]                            | the number of training iterations
-N [number]                               | the length of training sequences
-learningRate [number]                    | learning rate
-initweights [file]                       | initial weights
-grid_[minX/maxX/minY/maxY/step] [number] | 2D occupancy grid parameters
-sensor_[start/step]                      | 1D depth sensor parameters

License
-------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Release Notes
-------------
#### Version 1.1 (Keras)
* Native decoding of raw 1D depth data into 2D input.
* Larger NN network.
