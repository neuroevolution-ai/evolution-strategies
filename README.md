# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments, instead the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

## Installation

### Set password

For security purposes, the Jupyter Lab uses password authentication. To set a password run `generate_password.sh`.
This will generate a SHA-1 hash and saves it to `hashed_password.txt`. Now the Dockerfile can be built, which will
use this new password.

If you do not want any security measurements (which is not recommended, since the Jupyter Lab allows arbitrary code
execution), then build the Dockerfile as in TODO reference
and run the container like in TODO reference

### Build Dockerfile

To build the docker image run the following command inside the root directory of `evolution-strategies`.

`docker build -t evolution-strategies .`

### Start the container

A user was created inside the docker image, which needs to have the same user ID as the one on the host.
Otherwise the created files, e.g. the training data, cannot be accessed from outside the docker container.
Therefore the `UID` argument in the following command is set to your user ID on the host system.


#### Run default image with token authentication
TODO:
Map to your user id on the host to be able to mount a volume where the user inside docker has write access, by default
this is 1000, can be modified with build arguments

`docker run --user root -e NB_UID=$(id -u) -e NB_GID=$(id -g) -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies:testing`

`docker run --user $(id -u) --group-add users -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`

By default, Jupyter uses tokens to authenticate users for accessing the notebooks. By running

`docker run -d -p 8888:8888 --name es-token -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`

a Jupyter Lab will be started. To access it, run `docker logs es-token`. This will print the log when starting 
the container. From there follow the link with the address `127.0.0.1:8888` which automatically uses the generated 
token to log in.

For multiple accesses to the lab this requires having a cookie, which is set automatically. Alternatively one 
can define a password by going to `127.0.0.1:8888` directly and set a password using the token which is retrieved
from the logs.

#### Run with hashed password

If you want to use a password directly run the following command.

`docker run -d -p 8888:8888 --name es-password -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies xvfb-run -a -s='-screen 0 1400x900x24' start.sh jupyter lab --NotebookApp.password='sha1:9eeee5ad359d:b3a4cf67b0e0cbdf8ad4a63d8c2df3702bc26b33'`

This will start the container with a password already set. Currently this is `es-jupyter`. It is recommended to change it
if you are planning on using this method. A guide to generate a password can be found 
[here](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password). It requires
having Python and Jupyter installed. Then the new hash must be set in the command above in the `--NotebookApp.password`
parameter.  

#### Run without any security

If you want to run the Jupyter Notebook without any security measures start the docker container with

`docker run -d -p 8888:8888 --name es-open -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies xvfb-run -a -s='-screen 0 1400x900x24' start.sh jupyter lab --NotebookApp.token=''`

Be aware that any user on your network can access the container and execute commands on it, which in turn can be executed
on the host machine. It is therefore recommended to use either a password or the token mechanism.