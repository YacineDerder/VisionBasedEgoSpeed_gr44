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
      <a href="#experimental-setup">Experimental Setup</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
    </li>
    <li>
      <a href="#conclusion">Conclusion</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- PROJECT OBJECTIVES AND CONTRIBUTION -->
## Project Objectives and Contribution

The aim of this project is to replicate the results from the reference paper, by [Mathew & Khalid](https://arxiv.org/pdf/2212.05432.pdf).

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
The network is trained with a learning rate of 0.001, the ADAM optimizer and the MSE loss.

With the final network trained and the preprocessed dataset completed, we obtain the following structure, as opposed to the first one :

<img src="Figures\NetworkContribution.png" alt="NetworkContribution">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASET -->
## Dataset

We have worked with 2 datasets : [NuImage](https://www.nuscenes.org/nuimages) and [KITTI](https://www.cvlibs.net/datasets/kitti/).

Both of them contain more inputs than just the front camera and more outputs than just the ego speed, so the first set was to create a PyTorch _Dataset_ object able of sorting through the dataset and extracting only the features of interest in the correct order and by clips of the correct length. Both of those objects are defined in the `Datasets.py` script. We then use the `clip_to_laneline()` function to take the formatted clip as input and output the attention masks. Finally, we concatenate the resized results in 64x64 grayscale format before adding it to the new preprocessed dataset. The following figure summarizes the process.

<img src="Figures\DatasetPreprocessing.png" alt="DatasetPreprocessing">

Another PyTorch _Dataset_ object, also defined in `Datasets.py`, can be used to make a dataloader with the preprocessed data. The main advantage of creating a new preprocessed dataset is that we only have to apply some of the time costly functions to each frame once. If we then decide the do 100 epochs during training, we do not have to select relevent clips 100 times, or pass the same clips 100 times in the lane line network, which is what we should do if we started from the original dataset everytime. Furthermore, since the result of the preprocessing is much smaller memory-wise. For NuImage, the original RGB, 1600x900, all cameras US dataset is 91GB, while the 64x64 grayscale only front camera one is 1.6GB.

Both preprocessed datasets can be directly downloaded [here](https://drive.google.com/drive/folders/1U_DB388k41wZYvP8C2IXk6FzKInPLSbG?usp=sharing) as numpy arrays, their naming convention is `<NuImage/KITTI>_<input/output>_<train/test>.npy`. As an example, the NuImage training features dataset is named `NuImage_input_train.npy`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXPERIMENTAL SETUP -->
## Experimental Setup

In order to evaluate our network, we have used the RMSE error on the whole test datasets, as in the reference paper : $RMSE = \sum_{i=1}^{n} \frac{(\hat{y}_i - y_i)^2}{n}$

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

* Bullet point
* Other bullet point

<img src="Figures\Network.png" alt="Network">

`hello_world.py`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONCLUSION -->
## Conclusion

* Bullet point
* Other bullet point

<img src="Figures\Network.png" alt="Network">

`hello_world.py`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



