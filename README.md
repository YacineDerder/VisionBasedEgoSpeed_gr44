<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- 
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] 
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">DLAV project 2023</h3>

  <p align="center">
    20. Vision-based Vehicle Speed Estimation.
    <br />
    Group 44 : Yacine Derder & David Junqueira.
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-objectives-and-contribution">Project Objectives</a>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
    </li>
    <li>
      <a href="#experimental-setup">Experimental Setup</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
    <li>
      <a href="#conclusion">Conclusion</a>
    </li>
  </ol>
</details>



<!-- PROJECT OBJECTIVES AND CONTRIBUTION -->
## Project Objectives and Contribution

The aim of this project is to replicate the results from the reference paper, by [Mathew & Khalid](https://arxiv.org/pdf/2212.05432.pdf). That is, we want to extract the ego speed of a vehicle given a certain number of frames before and after the frame of interest : 

<img src="Figures\Objective.png" alt="Objective">

More specifically, we aim to reproduce the following network architecture :

<img src="Figures\NetworkOriginal.png" alt="NetworkOriginal">

As we can see in the above figure, in order to obtain the vehicles ego-velocity from a video stream with this method, we first need to :
* Extract the relevent frames from the dataset in the correct order. In our case, we need the front camera frames.
* Obtain the road lanes attention maps by passing the frames through the lane line segmentation network ([yoloP](https://github.com/hustvl/YOLOP)).
* Reshape, convert to grayscale, and concatenate the frames with their respective masks in a new preprocessed dataset.
* Design a 3D CNN network following the architecture described in the reference paper and train it using the inputs and labels of the preprocessed dataset.

The following figure shows more details about the architecture to reproduce :

<img src="Figures\NetworkDetail.png" alt="NetworkDetail">

As we can see, the network consists of 6 convolutional layers, 2 maxPool layers and 3 dense layers. The kernel used for the convolutional layers is always (3,3,3), however, the first maxPool kernel does not span in depth in order to preserve temporal information. As a padding of 1 and a stride of 1 are used for the convolutions, the dimensions are preserved between convolution layers.

With the final network trained and the preprocessed dataset completed, we obtain the following structure, as opposed to the first one :

<img src="Figures\NetworkContribution.png" alt="NetworkContribution">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASET -->
## Dataset

We have worked with 2 datasets : [NuImage](https://www.nuscenes.org/nuimages) and [KITTI](https://www.cvlibs.net/datasets/kitti/).

Both of them contain more inputs than just the front camera and more outputs than just the ego speed, so the first set was to create a PyTorch _Dataset_ object able of sorting through the dataset and extracting only the features of interest in the correct order and by clips of the correct length. In the case of NuImage, this object is defined in the `Datasets.py` script, while with KITTI we can manually simply download the clips of interest directly. We then use the `clip_to_laneline()` function to take the formatted clip as input and output the attention masks. Finally, we concatenate the resized results in 64x64 grayscale format before adding it to the new preprocessed dataset. The `MakeDataset.py` generates the preprocessed datasets from the original ones. The following figure summarizes the process.

<img src="Figures\DatasetPreprocessing2.png" alt="DatasetPreprocessing2">

Another PyTorch _Dataset_ object, also defined in `Datasets.py`, can be used to make a dataloader with the preprocessed data. The main advantage of creating a new preprocessed dataset is that we only have to apply some of the time costly functions to each frame once. If we then decide the do 100 epochs during training, we do not have to select relevent clips 100 times, or pass the same clips 100 times in the lane line network, which is what we should do if we started from the original dataset everytime. Furthermore, since the result of the preprocessing is much smaller memory-wise. For NuImage, the original RGB, 1600x900, all cameras US dataset is 91GB, while the 64x64 grayscale only front camera one is 1.6GB (for ~13300 clips of 13 frames).

Both preprocessed datasets can be directly downloaded [here](https://drive.google.com/drive/folders/1U_DB388k41wZYvP8C2IXk6FzKInPLSbG?usp=sharing) as numpy arrays, their naming convention is `<NuImage/KITTI>_<input/output>_<train/test>.npy`. As an example, the NuImage training features dataset is named `NuImage_input_train.npy`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXPERIMENTAL SETUP -->
## Experimental Setup

The training of the networks happens in the `Train_NN.py` script. As they were specified in the reference paper, we use the following parameters : 

* Learning rate = 0.001
* ADAM optimizer
* MSE loss

To use the file, the user must replace the `dataset_loc` variable defined at the begining of the file by their own path to the preprocessed dataset (downloadable on google drive). The user must then specify, when running the script, with 2 parameters whether to do the training and whether to save the result. As an example, in order to train the model but not save the results, one would run the script as such (on windows): `python Train_NN.py 1 0`. If we do decide to save the results, we will obtain 4 files in the same directory from which the script is ran :

* `iteration.npy` contains an array indicated after which iterations the test loss was evaluated.
* `train_loss.npy` contrains the train loss at each iteration in `iteration.npy`.
* `test_loss.npy` contrains the test loss at each iteration in `iteration.npy`.
* `model.pth` contains the model weights.

In order to evaluate our network after training, we have used the RMSE error on the whole test datasets, as in the reference paper : 
$$RMSE = \sqrt{\sum_{i=1}^n \frac{(\hat{y}_i - y_i)^2}{n}}$$

For both datasets, we use the predefined provided partition between train and test data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

The results of training are summarized in the following figure :

<img src="Figures\TrainResults.png" alt="TrainResults">

As we can see we have much more satisfying results on the NuImage dataset, with the test loss on the KITTI dataset remaining very high. The training for the NuImage network happened using the preprocessed dataset and took about 10 hours using cuda on a NVIDIA GeForce RTX 3050 Ti Laptop GPU. 

The RMSE loss computed on the whole test dataset is 1.02, which is actually slightly less than the one given in the paper. The `Evaluate_NN.py` script implements this computation if it is run with its `animate` flag set to `False`. Furthermore, this script provides a `evaluate()` function which takes an RGB clip of any size, as well as both networks required for a forward pass, and outputs the estimated velocity. For clips extracted from NuImage, the evaluation function takes ~0.4 seconds to run. Despite this, the network would not be able to perform in true real-time, as it requires frames from a time window in the past as well as in the futur to make a prediction.

The following figure shows an example of results obtained running `Evaluate_NN.py`, where for each sample the evaluation time, estimated ego-speed, actual ego-speed and absolute error is computed and printed.

<img src="Figures\Evaluation.png" alt="Evaluation">

`EvaluateTestDrives.py` builds clips of the right format from the provided [test datasets](https://drive.google.com/drive/folders/16xf0AF9zgWAuK6Xyr5xK85t77hM3BwAv) and uses the `evaluate()` function on all of them before storing the results in the required `.json` format. The results can be found at the dedicated place on the [same drive](https://drive.google.com/drive/folders/1OgSYpEttJQSREDNMQVNPdaOmjBM9xMXC).

NOTE : As the frequency was not provided for the test clips, it was assumed to be 30Hz.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONCLUSION -->
## Conclusion

To conclude, this project has tought us many aspects of scientific research. From the initial ideas to the final implementation, most of the information we need must be found by ourselves in scientific litterature, as we are working on a very specific problem that not any data scientist would have intuitive solutions to give to us. This project has also highlighted the huge part that datasets play in depp learning, and the time and effort that comes with simply handling them in a clear and efficient way : in our case, dataset extraction and preprocessing actually took more time than network design and training. Overall, it was a very enriching learning experience as it has also made us more confortable with a deep learning library as popular as PyTorch.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



