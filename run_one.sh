#!/bin/bash

for i in {0..8}
do
  python3 LPSelectiveSearch.py $i &
done
wait
python3 LicensePlateLearner2.py