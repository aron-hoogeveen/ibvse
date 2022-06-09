#!/bin/bash

# global model
mkdir -p featureextraction/solar/data/networks/
wget -nc https://imperialcollegelondon.box.com/shared/static/fznpeayct6btel2og2wjjgvqw0ziqnk4.pth -O featureextraction/solar/data/networks/resnet101-solar-best.pth

# local model
mkdir -p featureextraction/solar/solar_local/weights/
wget -nc https://imperialcollegelondon.box.com/shared/static/4djweum6gs30os243zqzplhafxlys31z.pth -O featureextraction/solar/solar_local/weights/local-solar-345-liberty.pth

# SuperGlue utils
mkdir -p featureextraction/superglue/
wget -nc https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/utils.py -O featureextraction/superglue/utils.py
