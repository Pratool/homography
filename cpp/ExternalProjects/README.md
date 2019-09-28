# External Dependencies
CMake is used to manage dependencies in this project. A subset of the
dependencies necessary to build and run the applications of this repository can
be installed using this incomplete guide. No attempt was made to address
cross-platform issues and has only been tested on a personal machines running
Arch Linux and Ubuntu 18.04 LTS. For the sake of brevity, I have omitted various
configuration parameters and provide no warning against using danagerous
commands. You will never need super user privileges to build and install the
external dependencies. All commands provided are examples only.

## Installation
1. Set the current working directory to the directory containing the
`CMakeLists.txt` in this directory.  
`cd homography/ExternalProjects/`

2. Make build and installation directories. Switch to the build directory.  
`mkdir build`  
`mkdir ../install`
`cd build`

3. Run the CMake configuration step.  
`cmake -DCMAKE_INSTALL_PREFIX=../../install ..`

4. Make each of the dependency targets necessary specifying the number of
maximum parallel jobs to be the number of CPU cores. Specifying all as the build
target will attempt to build and install all of the external dependencies.  
`make -j $(nproc) all`
