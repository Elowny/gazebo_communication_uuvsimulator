# gazebo_communication_uuvsimulator
This is a communication file connecting the UUVsimulator and the user's program. The user can use this file to carry out development experiments on the UUVsimulator, such as path planning, or trajectory control, etc.

## uuvsimulator
uuv_simulator is a public project that includes plugins and ROS applications that allow simulation of underwater vehicles in Gazebo.

Before using, please make sure you have read and studied the official website of uuv_simulator： https://uuvsimulator.github.io/

## Deep Reinforcement Learning
We provide a path planning simulation program based on deep reinforcement learning, and provide some classic deep reinforcement learning algorithm programs, users can directly choose an algorithm and run it in gazebo together with the gazebo_env program

## notice：
If you want to use the sonar function (used by default), please activate sonar with the following command after activating the uuvsimulator environment:

roslaunch uuv_gazebo rexrov_sonar.launch
