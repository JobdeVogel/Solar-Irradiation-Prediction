## IrradianceNet: *Solar irradiance prediction using Deep Learning*
This code was developed by Job de Vogel.

### Installation
Gneral prerequisites:
* Nividia GPU

Prerequisites for dataset generation:
* Valid installation of Rhino 7;
* Radiance or AcceleRad (https://nljones.github.io/Accelerad/)

Prerequisites for Neural Network training:
* Visual Studio 19 Enterprise (https://visualstudio.microsoft.com/vs/older-downloads/)
* Cuda Driver 11.3

Install IrradianceNet on Windows by running:
`install.bat`

### How to run?
Download a 3DBAG dataset:
`cd download`
`python bag.py`

Run the dataset generation sequentially:
`cd dataset`
`python main.py -std` (use -std for printing the logs to stdout)

Run the dataset generation in parallel:
`cd dataset`
`python run.py`

General setting for the dataset generation can be changed in `parameters\params`

Train a neural network for solar irradiance prediction. Make sure the correct paths are set in the IrradianceNet cfg:
`cd pointnext`
`python examples/segmentation/main.py --cfg cfgs/irradiance/irradiancenet-l.yaml`