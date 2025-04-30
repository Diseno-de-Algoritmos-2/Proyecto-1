#!/bin/bash

# Loop through values of SIM from 1 to 500
for SIM in {1..500}
do
    python Comparación/main.py $SIM
done