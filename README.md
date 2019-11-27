# Using evolution strategies to train environments

This is a fork of the implementation of the algorithm described in
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)
(Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation currently supports all environments which are shipped with the OpenAI Gym, as well as the
[Roboschool](https://github.com/openai/roboschool/) from OpenAI and the
[PyBullet Robotics Environments](https://github.com/openai/gym/blob/master/docs/environments.md#pybullet-robotics-environments).


## Installation

The implementation uses Jupyter Notebooks. To run them, a Dockerfile is provided to build a Docker image. Inside this image
all dependencies which are needed will be installed. Then a Docker container can be created and started,
which will start a Jupyter Lab server at the address `127.0.0.1:8888`. From there the Notebooks can be accessed and
tested with a browser.

### Prerequisites

Since the implementation provides a Dockerfile, only Docker is needed on the host.

If you want to test the Notebooks without using Docker you can look into the Dockerfile to find out which packages 
need to be installed.

#### Set a password

For security purposes, the Jupyter Lab uses password authentication. To set a password run

`./generate_password.sh`.

in the main directory of this repository. If there is a `permission denied` error, the script needs execution rights.
To add them, run 

`chmod u+x generate_password.sh`.

This will generate a SHA-1 hash of the password and save it to `hashed_password.txt`. Now the Dockerfile can be used
to build a Docker image, which will use this new password.

#### Use without any security measurements

If you do not want any security measurements (which is not recommended, since the Jupyter Lab allows arbitrary code
execution), then build the Dockerfile as in TODO reference
and run the container like in TODO reference

### Build Docker image

To build the Docker image run the following command inside the root directory of `evolution-strategies`.

`docker build -t evolution-strategies .`

The Docker image will be created and called _evolution-strategies_.

### Start the Docker container

#### Use the previously set password

After building the Docker image you can create and start the Docker container with

`docker run --user $(id -u) --group-add users -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`.

This will automatically set the user ID inside the container to the one you have on the host. The container is started
in the background and the _evolution-strategies_ directory on the host gets mounted inside the container.

Now the Jupyter Lab is up and running and can be accessed at `127.0.0.1:8888`.

#### Run without any security measurements

If you want to run the Jupyter Lab without any security measurements start the Docker container with

`docker run --user $(id -u) --group-add users -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies xvfb-run -a -s='-screen 0 1400x900x24' start.sh jupyter lab --NotebookApp.token=''`

Be aware that potentially any user on your network can access the container and execute commands on it, which in turn
can be executed on the host machine. It is therefore recommended to use the password.