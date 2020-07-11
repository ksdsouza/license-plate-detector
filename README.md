# license-plate-detector



## Getting Started

Unzip `resources.zip` for the images and annotations. The folder structure after unzipping should be:

```
license-plate-detector/
|---airplanes/
|---Images/
|---Annotations/
|---LPSelectiveSearch.py
etc
```



Due to heavy memory usage, selective search and model training are done in chunks of 20 images at a time. The chunk is specified by a CLI argument.



To generate the selective search results, run `python3 LPSelectiveSearch.py $CHUNK_NUM`

And to train the model on the chunk, run `python3 LicensePlateLearner2.py $CHUNK_NUM`. The model is stored in `ieeercnn_vgg16_1.h5`. If this file is detected when this command is run, the model will be loaded, and further trained. Otherwise a new model will be created.

To check the performance of the trained model, run `python3 LPModelCheck.py`.