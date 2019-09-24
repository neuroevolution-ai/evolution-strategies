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

RUN conda install --quiet --yes \
    'matplotlib' \
    'gast==0.2.2'

# Use pip for packages that cannot be installed with conda and for TensorFlow and NumPy becaue we do not want the version with MKL
RUN pip install --quiet \
    numpy==1.16.4 \
    tensorflow \
    gym \
    roboschool

RUN conda clean --yes --all -f && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


# $NB_USER == jovyan, docker does not support dynamic substitution in chown
ADD --chown=jovyan:root . work/evolution-strategies/

WORKDIR work/evolution-strategies/

# Run jupyter notebook with a fake display to allow rendering in roboschool as suggested here:
# https://github.com/openai/gym#rendering-on-a-server
CMD ["xvfb-run", "-s", "-screen 0 1400x900x24", "start-notebook.sh"]