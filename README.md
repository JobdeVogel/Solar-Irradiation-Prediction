## IrradianceNet: *Towards Solar Irradiation Prediction on 3D Urban Geometry using Deep Neural Networks*
This code aims to predict solar irradiation on 3D urban geometry based on 3D BAG data. This code consists of four steps:
* Download: a webscraper which downloads and extracts online 3D BAG files;
* Dataset: a Python package which preprocesses 3D BAG geometry to a format which can be used for irradiation prediction;
* Prediction: Python code based on PointNeXt which is used to predict irradiation on point clouds from the preprocessed geometry.
* Interaction: A Grasshopper Server-Client system to predict irradiation in McNeel Rhino based on 3D BAG data.

### Prerequisites
General prerequisites:
* Dedicated Nividia GPU;
* Windows Home/Pro operating system.
* Anaconda/Miniconda (https://docs.anaconda.com/miniconda/)

Prerequisites for dataset generation:
* Valid installation of McNeel Rhino 7;
* Radiance 5.4a(2021-3-28) optionally with AcceleRad (https://nljones.github.io/Accelerad/)

WARNING: installing multiple versions of Radiance (e.g. due to different lbt libaries in Grasshopper/Python) can lead to problems.

Prerequisites for irradiation prediction:
* Visual Studio 19 Enterprise (https://visualstudio.microsoft.com/vs/older-downloads/)
* Nvidia CUDA Driver 11.7 or 11.8

Having multiple version of Visual Studio may lead to issues in building the required modules.

### Installation
* Download the zip file or pull this repo with git.
* Install IrradianceNet on Windows by running: `install.bat`

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

#### Interaction model from Grasshopper
The server code for the Grasshopper interaction model can be found in `examples/segmentation/server.py`.

## Sources
Peters, R., Dukai, B., Vitalis, S., van Liempt, J., & Stoter, J. (2022). Automated 3D Reconstruction of
LoD2 and LoD1 Models for All 10 Million Buildings of the Netherlands. Photogrammetric
Engineering & Remote Sensing, 88(3), 165–170. https://doi.org/10.14358/PERS.21-
00032R2

Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H. A. A. K., Elhoseiny, M., & Ghanem, B. (2022).
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies. 36th
Conference on Neural Information Processing Systems (NeurIPS 2022). https://doi.org/
https://doi.org/10.48550/arXiv.2206.04670
