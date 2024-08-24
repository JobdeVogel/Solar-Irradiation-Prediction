## IrradianceNet: *Towards Solar Irradiation Prediction on 3D Urban Geometry using Deep Neural Networks*
This code aims to predict solar irradiation on 3D urban geometry based on 3D BAG data. This code consists of three steps:
* Download: a webscraper which downloads and extracts online 3D BAG files;
* Dataset: a Python package which preprocesses 3D BAG geometry to a format which can be used for irradiation prediction;
* Prediction: Python code based on PointNeXt which is used to predict irradiation on point clouds from the preprocessed geometry.

### Installation
Gneral prerequisites:
* Dedicated Nividia GPU;
* Windows Home/Pro operating system.

Prerequisites for dataset generation:
* Valid installation of McNeel Rhino 7;
* Radiance or AcceleRad (https://nljones.github.io/Accelerad/)

Prerequisites for irradiation prediction:
* Visual Studio 19 Enterprise (https://visualstudio.microsoft.com/vs/older-downloads/)
* Nvidia CUDA Driver 11.3

It is highly recommended to use (Mini)conda for the installation of this Python package (https://docs.anaconda.com/miniconda/)

Install IrradianceNet on Windows by running:
`install.bat`

### How to run?
#### Download a 3DBAG dataset:
`cd download`
`python bag.py`

If the scraper is unable to fetch data, this may be caused by a newer version of the 3D BAG. In that case, please overwrite the VERSION global parameter accordingly

#### Run the dataset generation sequentially:
`cd dataset`
`python main.py -std`

The `-std` paramameter indicates that the logs should be printed to stdout. When using parallel mode, this should be avoided, because multiple processes will send logs to stdout simultaneously

#### Run the dataset generation in parallel:
`cd dataset`
`python run.py`

General setting for the dataset generation can be changed in `dataset/parameters/params`

#### Train a neural network for solar irradiation prediction.
`cd pointnext`
`python examples/segmentation/main.py --cfg cfgs/irradiance/irradiancenet-l.yaml --dataset.common.data_root path_to_dataset`

Parameters for training, validation and testing can be changed in `cfgs/default`, `cfgs/irradiance/default` and `cfgs/irradiance/ ... .yaml`

## Sources
Peters, R., Dukai, B., Vitalis, S., van Liempt, J., & Stoter, J. (2022). Automated 3D Reconstruction of
LoD2 and LoD1 Models for All 10 Million Buildings of the Netherlands. Photogrammetric
Engineering & Remote Sensing, 88(3), 165–170. https://doi.org/10.14358/PERS.21-
00032R2

Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H. A. A. K., Elhoseiny, M., & Ghanem, B. (2022).
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies. 36th
Conference on Neural Information Processing Systems (NeurIPS 2022). https://doi.org/
https://doi.org/10.48550/arXiv.2206.04670
