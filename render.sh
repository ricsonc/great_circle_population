#!/usr/bin/env bash

# echo {0..359} | xargs -n 1 -P 20 python go.py

cd out

ffmpeg -y -r 30 -f image2 -i %06d.png \
       -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
       -tune animation \
       -vcodec libx265 -crf 20 -pix_fmt yuv420p ../out.mp4

# ffmpeg -y -r 60 -f image2 -i %06d.png \
#        -vcodec libvpx-vp9 -crf 25 -b:v 0 ../out.webm
