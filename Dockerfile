FROM jupyter/base-notebook:latest

# Switch to root user to install packages
USER root

# Update the system and install base and roboschool requirements
RUN apt-get update -y && apt-get install -y git xvfb ffmpeg libgl1-mesa-dev libharfbuzz0b libpcre3-dev libqt5x11extras5 build-essential

# Switch back to unprivileged user for python packages. User is defined in base docker image
USER $NB_USER

# NumPy has changed something in version 1.17+ which causes import errors in TensorFlow. Until this fix is merged
# use a slightly older version of NumPy, same with gast
RUN conda install --quiet --yes \
    'gast==0.2.2' \
    'matplotlib' \
    'pandas' \
    'ipywidgets'

RUN conda clean --yes --all -f && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install TensorFlow and NumPy with pip to prevent using the MKL version which in this implementation is slower
RUN pip install --quiet \
    numpy==1.17.2 \
    tensorflow==2.0.0 \
    keras==2.3.1 \
    gym[atari] \
    opencv-python \
    neat-python

WORKDIR work/evolution-strategies/

ADD hashed_password.txt .
ADD launch.sh .

CMD ["/bin/bash", "launch.sh"]
