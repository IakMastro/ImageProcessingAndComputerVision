#!/bin/sh

docker build -f Dockerfile -t xelatex-image-processing .
docker run -it -v $PWD/src:/home -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY  xelatex-image-processing /bin/bash