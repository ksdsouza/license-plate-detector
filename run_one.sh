#!/bin/bash

for i in {0..8}
do
  python3 LPSelectiveSearch.py $i &
done
wait
for i in {0..8}
do
  python3 LicensePlateLearner2.py $i
done