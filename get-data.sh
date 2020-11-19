#!/usr/bin/env bash

# taken from http://rpg.ifi.uzh.ch/teaching.html

mkdir -p data
cd data
curl http://rpg.ifi.uzh.ch/docs/teaching/2016/parking.zip --output parking.zip
unzip parking.zip
rm parking.zip
