FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg locales

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG="en_US.UTF-8" LANGUAGE="en_US:en" LC_ALL="en_US.UTF-8"
