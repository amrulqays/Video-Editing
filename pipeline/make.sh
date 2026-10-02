#!/usr/bin/env bash
# Rebuild the UMBRAZOR fight edit from the two Seedance source clips.
# Usage: pipeline/make.sh <clip_A.mp4 (Daus take)> <clip_B.mp4 (UMBRAZOR take)> [workdir]
# Needs: ffmpeg, python3 with numpy scipy soundfile pyloudnorm opencv-python-headless pillow
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
A=$(realpath "$1"); B=$(realpath "$2"); W=${3:-build}
mkdir -p "$W" && cd "$W"
cp "$A" A.mp4 && cp "$B" B.mp4
ffmpeg -v error -y -i A.mp4 -vn -ac 2 A_orig_st.wav
ffmpeg -v error -y -i B.mp4 -vn -ac 2 B_orig_st.wav
python3 "$HERE/render_video.py"          # -> video_main.mp4 (picture only)
python3 "$HERE/score.py"                 # -> score_mix.wav (music + SFX + mix)
ffmpeg -v error -y -i video_main.mp4 -c:v libx264 -preset slow -tune film -b:v 11M -maxrate 16M \
  -bufsize 22M -pass 1 -passlogfile x264pass -an -f null /dev/null
ffmpeg -v error -y -i video_main.mp4 -i score_mix.wav -map 0:v -map 1:a -c:v libx264 -preset slow \
  -tune film -b:v 11M -maxrate 16M -bufsize 22M -pass 2 -passlogfile x264pass -pix_fmt yuv420p \
  -c:a aac -b:a 320k -ar 48000 -movflags +faststart UMBRAZOR_Peak_Fight.mp4
echo "done: $W/UMBRAZOR_Peak_Fight.mp4"
