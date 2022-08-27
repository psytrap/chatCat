#!/bin/sh

rm -v nohup.out
sleep 1
nohup python3 -u ./chatCat.py --url $(cat webhook) &

