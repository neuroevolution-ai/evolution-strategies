**Status:** Archive (code is provided as-is, no updates expected)

# Distributed evolution with redis and MuJoCo

## Installation of MuJoCo an mujoco-py
(Copied from https://github.com/openai/mujoco-py)

Install MuJoCo

1. Obtain a 30-day free trial on the MuJoCo website or free license if you are a student. The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 2.0 binaries for Linux or OSX.
3. Unzip the downloaded mujoco200 directory into ~/.mujoco/mujoco200, and place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt.

### Requirements and mujoco-py

Ubuntu 16.04 based. Changed packages according to distribution

Base requirements
`sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev xvfb ffmpeg python3-pip`

When an error occurs that -lGl cannot be found execute this line
`sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so`

Requirements
`pip3 install --user glfw>=1.4.0 numpy>=1.11 Cython>=0.27.2 imageio>=2.1.2 cffi>=1.10 lockfile>=0.12.2 click redis tensorflow imageio imageio-ffmpeg`

`pip3 install --user -U 'mujoco-py<2.1,>=2.0'`

## Visualization

Run the following to create a fake display and render the Humanoid-v2 with the given policy file created in training. When not running on a headless machine
remove the fake display and uncomment the imageio command in viz.py and change render mode to "human" in policy.py in method rollout().

`xvfb-run -s "-screen 0 800x600x24" python3 viz.py 'Humanoid-v2' snapshot_xxxxx.h5`
