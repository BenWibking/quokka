#!/bin/sh

CONVERT_EXE=$(which convert)  # install with: "spack install imagemagick"
RENAME_EXE=$(which rename)    # install with: "spack install rename"
FFMPEG_EXE=$(which ffmpeg)    # install with: "spack install ffmpeg+libx264+drawtext"

set -x

## rename *.png files

$RENAME_EXE 's/\d+/sprintf("%07d",$&)/e' -- *.png

## resize frames

for file in *.png;
do
    $CONVERT_EXE $file -resize 2048x2048 $(basename -s .png $file).resize_large.png
done

## make movie
FPS=10
FILTERS="pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white, drawtext=fontfile=/usr/share/fonts/google-droid/DroidSans.ttf:text='QUOKKA Simulation':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.6:boxborderw=5:x=0.9*(w-text_w):y=0.9*(h-text_h)"

$FFMPEG_EXE -framerate $FPS -pattern_type glob -i "*.resize_large.png" -r $FPS -vcodec libx264 -vf "$FILTERS" -pix_fmt yuv420p -preset slow -tune animation -crf 18 movie_large.mov
