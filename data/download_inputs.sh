#!/bin/bash
set -eu

# download files from onedrive, or you can acess directrly to
# https://1drv.ms/u/s!AnwqImTfVnOs22WmAZQF_Y4FGkkM
wget --no-check-certificate -O ./inputs.7z "https://onedrive.live.com/download?cid=AC7356DF64222A7C&resid=AC7356DF64222A7C%2111749&authkey=AAHuOht2wFTuE5o"
7z x ./inputs.7z
rm ./inputs.7z
