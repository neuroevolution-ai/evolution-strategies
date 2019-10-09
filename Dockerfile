FROM jupyter/base-notebook:latest

# Switch to root user to install packages
USER root

RUN apt-get update
RUN apt-get dist-upgrade -y

# Install base requirements
RUN apt-get install -y git xvfb ffmpeg

# Roboschool Requirements
RUN apt-get install -y libgl1-mesa-dev libharfbuzz0b libpcre3-dev libqt5x11extras5

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

# Roboschool is deprecated after version 1.0.48
# Install TensorFlow and NumPy with pip to prevent using the MKL version which in this implementation is slower
RUN pip install --quiet \
    tensorflow \
    numpy==1.16.4 \
    gym \
    roboschool==1.0.48

# $NB_USER == jovyan, docker does not support dynamic substitution in chown
ADD --chown=jovyan:root . work/evolution-strategies/

WORKDIR work/evolution-strategies/

# Run jupyter notebook with a fake display to allow rendering in roboschool as suggested here:
# https://github.com/openai/gym#rendering-on-a-server
CMD ["xvfb-run", "-s", "-screen 0 1400x900x24", "start-notebook.sh"]