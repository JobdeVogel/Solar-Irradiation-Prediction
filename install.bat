@echo off
echo Please make sure you have cloned the IrradianceNet package, Rhino 7 installed and Visual Studio 2019 Enterprise
echo Please make sure you have installed AcceleRad from https://nljones.github.io/Accelerad/ and added to path, for dataset generation
echo ------------

REM Define variables
set ENV_NAME=IrradianceNet310
set PYTHON_VERSION=3.10.10

REM Create conda environment with specific Python version
echo Creating Anaconda environment...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
echo Anaconda environment created.

REM Activate conda environment
echo Activating Anaconda environment...
call conda activate %ENV_NAME%

@REM REM Git clone irradiancenet
@REM echo installing git
@REM call conda install git -y

@REM echo init git
@REM call git init -y

@REM echo clone
@REM call git clone https://github.com/JobdeVogel/Solar-Irradiation-Prediction -y -y
@REM cd Solar-Irradiation-Prediction

REM Install packages with conda
echo Installing packages with conda...
call conda install numpy numba -y
call conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

REM Go back to package in case compiler install fails
call cd %CD%
call conda activate %ENV_NAME%

REM Install packages with pip
echo Installing packages with pip...
@REM call python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

REM To pointnext folder
cd pointnext

REM Install requiremtents.txt
call python -m pip install -r requirements.txt
call python -m pip install windows-curses
call python -m pip install urllib3==1.26.15

REM install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
call python setup.py install
cd ../

REM grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
call python setup.py build_ext --inplace
cd ..

REM point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
call python setup.py install
cd ..

REM Below are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
call python setup.py install --user
cd ../emd
call python setup.py install --user
cd ../../../../

REM Install package for dataset generation
echo Installing dataset generation packages
cd dataset
call python -m pip install -r requirements.txt
call conda install setuptools=59.5.0

echo Ready to run IrradianceNet!!
echo Download data using ...
echo Generate dataset using ...
cd ../

@REM cd pointnext
@REM echo Train model using python examples/segmentation/main.py --cfg cfgs/irradiance/irradiancenet-l.yaml