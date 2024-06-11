@echo off
echo Please make sure you have cloned the IrradianceNet package, Rhino 7 installed and Visual Studio 2019 Enterprise
echo Please make sure you have installed AcceleRad from https://nljones.github.io/Accelerad/ and added to path, for dataset generation
echo ------------

REM Define variables
set ENV_NAME=IrradianceNet310_clone
set PYTHON_VERSION=3.10

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
@REM call git clone https://github.com/JobdeVogel/Graduation-Building-Technology -y -y
@REM cd Graduation-Building-Technology

REM Install packages with conda
echo Installing packages with conda...
call conda install numpy numba -y
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
@REM call conda install conda-forge::cxx-compiler -y

REM Go back to package in case compiler install fails
call cd %CD%
call conda activate %ENV_NAMEREM Install packages with pip
echo Installing packages with pip...
@REM call python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

REM To pointnext folder
cd pointnext

REM Install requiremtents.txt
call python -m pip install -r requirements_clone.txt
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

REM Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
call python setup.py install --user
cd ../emd
call python setup.py install --user
cd ../../../../

@REM REM Install package for dataset generation
@REM echo Installing dataset generation packages
@REM cd dataset
@REM call python -m pip install -r requirements.txt
@REM call conda install setuptools=59.5.0

@REM echo Ready to run IrradianceNet!!
@REM echo Download data using ...
@REM echo Generate dataset using ...
@REM cd ../

@REM cd pointnext
@REM echo Train model using python examples/segmentation/main.py --cfg cfgs/irradiance/irradiancenet-l.yaml