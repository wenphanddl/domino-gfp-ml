#!/bin/bash
python partition-data.py
python train-column-transformer.py
python train-model-logreg.py
