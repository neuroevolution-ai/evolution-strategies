# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments, instead the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

## Instructions

### Build Dockerfile

A user will be created inside the docker image, which needs to have the same user ID as the one on the host.
Otherwise the created files, e.g. the training data, cannot be accessed from outside the docker container.
Therefore set `UID` in the following command to your user ID on the host system.

#### Linux

If you are using Linux run `id -u` to get your ID.

#### Windows

In Windows you do not have to change this.

Now you can build the docker image by running the following command inside the root directory of `evolution-strategies`.

`docker build --build-arg UID=$(id -u) -t evolution-strategies .`

### Run built image

#### Run with hashed password

The password is hashed and set in the Dockerfile. Currently it is set to `es-jupyter`. If you want to change it, [here](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password) is a guide to hash a password.

`docker run -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`

You can then access the notebook with your browser at the address `127.0.0.1:8888`.

#### Run without any security

If you want to run the Jupyter Notebook without any security measures replace the argument `--NotebookApp.password=...` with
`--NotebookApp.token=''` in the Dockerfile.
