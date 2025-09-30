1. Flash the NVIDIA Jetson with Jetpack 6.2.1 using NVIDIA SDK manager. Follow the steps [here](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/two_ways_to_set_up_software.html).
2. Connect the external hardware and setup:
   1. Microphone:
         1. Connect the microphone to the Jetson via USB. Make sure the light on the microphone lights up.
   2. Camera:
        1.  Connect the camera via Ethernet cable to the Jetson. Once connected go to wired connection in settings find the device that is 100/mb speed and go into the setting. Click on IPv4 and choose
        2.  To find IP of the Jetson use `arp -a` inside the terminal once you connected the camera. This should find the IP of the camera, look at the terminal to find the IP.
        3.  Copy and paste the IP of the camera to a search browser (such as firefox) and if an Amcrest page pops up the camera is connected!
   3. BME Sensor:
        1.  Please follow the BME sensor setup guide on [sensor setup github](https://github.com/uic-evl/SageEdge/blob/main/EDU_SetUp/configuring_env_sensor.md).
3. Set up the AI program by following the steps from the [setup github](https://github.com/uic-evl/SageEdge/tree/main/AI_Programs/Movement_Tracking).
