# Deepsream_ModelParallel_ProcessFramework

## 1. Prerequisites

Follow these procedures to use the deepstream-app application for native
compilation.

You must have the following development packages installed

    GStreamer-1.0
    GStreamer-1.0 Base Plugins
    GStreamer-1.0 gstrtspserver
    X11 client-side library
    Glib json library - json-glib-1.0
    yaml-cpp

1. To install these packages, execute the following command:
   sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev libyaml-cpp-dev


## 2. Main directory structure explain
```
|── data                        // engine file
|── dot                         // visual pipeline file
|── gst-nvdsmetamux             // additional important plug-in                     
|── deepstream_parallel_app.cpp // main code 
|—— dstest3_config              // pipeline config
|—— dstest3_pgie_config_0       // model_1 config
|—— dstest3_pgie_config_1       // model_2 config
```

## 3. CUDA_VER require

  $ Set CUDA_VER in the MakeFile as per platform.
      For Jetson, CUDA_VER=11.4
      For x86, CUDA_VER=12.1


## 4. Usage

  firstly: prepare your engine giles and put them into the ./data
  
  sencondly: according to your demand,change the relative files, includethe dstest3_config、dstest3_pgie_config_0 and dstest3_pgie_config_1

  then, run the application by executing the command:
   bash run_parallel.sh

NOTE:
1. Prerequisites to use nvdrmvideosink (Jetson only)
   a. Ensure that X server is not running.
      Command to stop X server:
          $sudo service gdm stop
          $sudo pkill -9 Xorg
   b. If "Could not get EGL display connection" error is encountered,
      use command $unset DISPLAY
   c. Ensure that a display device is connected to the Jetson board.
