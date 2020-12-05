#!/usr/bin/env bash

# taken from http://rpg.ifi.uzh.ch/teaching.html

mkdir -p data
cd data
rm -rf parking
curl http://rpg.ifi.uzh.ch/docs/teaching/2016/parking.zip --output parking.zip
unzip -q parking.zip
rm parking.zip

# TODO: remove commas from K.txt

cd parking
mv K.txt K.csv
cat K.csv | sed "s/,//g" > K.txt
