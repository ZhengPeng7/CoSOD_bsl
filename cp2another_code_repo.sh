#!/bin/bash
# Set dst repo here.
repo="Conensus1-DUTS"
mkdir ../${repo}
mkdir ../${repo}/evaluation
mkdir ../${repo}/models

cp ./*.sh ../${repo}
cp ./*.py ../${repo}
cp ./*.ipynb ../${repo}
cp ./evaluation/*.py ../${repo}/evaluation
cp ./models/*.py ../${repo}/models
cp -r ./.git* ../${repo}
