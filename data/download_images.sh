#!/bin/bash
set -eu

# download images form flickr (this takes some time)
echo prepareing folders...
for i in $(seq 10 99); do
	mkdir -p ./images/${i}
done

echo downloading images...
cut -d' ' -f2 ./image_urls.txt | xargs -L 1 -P 1 -I{} \
	bash -c '[ ! -f ./images/$(basename {} | cut -c1-2)/$(basename {}) ] && wget --wait=2.5 --max-redirect 0 --random-wait -P ./images/$(basename {} | cut -c1-2) {}'

#find ./images/ -name "*jpg" | xargs -I"{}" bash -c 'convert -colorspace RGB {} `echo {} | sed -e "s/images/color_images/g"`'
