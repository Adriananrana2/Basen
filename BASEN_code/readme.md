# This is the pytorch implementation of BASEN

## dataset and data processing

The Cocktail_Party sub-dataset is chosen to be the dataset. You can obtain the dataset from [this page](https://doi.org/10.5061/dryad.070jc). The data processing code is in ```data_processing``` directory, which comes from the [UBESD](https://github.com/MaryamHoss/UBESD). The matlab scripts to proccess the EEG data are in ```data_processing/EEG_preprocessing``` while the script for generating the dataset is in ```data_processing/dataset_generate```. After processing, place the dataset in ```data``` directory. Below is the structure of dataset after processing:

```
└─Cocktail_Party
    ├─Behavioural Data
    ├─EEG
    ├─Normalized
    │  ├─20s
    │  │  ├─eeg
    │  │  │  └─new
    │  │  └─fbc
    │  │      └─new
    │  ├─2s
    │  │  ├─eeg
    │  │  │  └─new
    │  │  └─fbc
    │  │      └─new
    │  │       
    │  └─60s
    │      ├─eeg
    │      │  └─new
    │      └─fbc
    │          └─new
    ├─preprocessed_EEG
    └─Stimuli
```





## train

To train the BASEN,  run this:

```
python distributed.py -c configs/BASEN.json
```

You can change the config in ```config/BASEN.json```, for example, the dataset path.

The checkpoint will be saved in ```exp/BASEN/checkpoint```.

## test

To test, run:

```
python test.py -c configs/experiments.json
```

You can change the test config in ```config/experiments.json```.
