## DISUNet

> This repo is for the following paper.
> DISUNet: Denoising Inertial Sensor Data with UNet and Diffusion Framework for Pedestrian Tracking

### Training
1. Direct training without overwrite the config from `config.json` file
```bash
python trainScript.py 
```

2. overwitte the config in `config.json` file with `[name_of_datasets]` as the target datasets, and `[model_weight]` as the starting model weight.
```bash
python trainScript.py -d [name_of_datasets] -b [model_weight]
```

> [name_of_datasets] can be ADVIO, RoNIN, RIDI, OIOD(OxIOD), OIOD_tango (OxIOD tango), where the OIOD is the one with only VICON label.
> [model_weight] should be the path to the weight directory.
> The default config.json file hyperparameter can be applied to all datasets with the same result of original paper.

### Testing
<!-- 
1. Download the weight from authors. **Please install the gdown in advance, or you may download it direct from the [website](https://drive.google.com/file/d/1nz5rLhDdMe4x0pHpOBPIikZcsC25VMEL/view?usp=sharing)**

```bash
gdown 1nz5rLhDdMe4x0pHpOBPIikZcsC25VMEL
``` -->

1. Direct testinig from the config file without overwrite the config in the `config.json` file. **Notice: If the `base_model_weight` parameter is `""` then the testing will select the baseline testing to generate the logs for baseline**  

```bash
python testScript.py
```

2. Testing with overwrite the config in the `config.json` file with `[name_of_datasets]` as the target datasets, and `[model_weight]` as the starting model weight. **Notice: If the `base_model_weight` parameter is `""` then the testing will select the baseline testing to generate the logs for baseline**  

```bash
python testScript.py -d [name_of_datasets] -b [model_weight]
```
> [name_of_datasets] can be ADVIO, RoNIN, RIDI, OIOD(OxIOD), OIOD_tango (OxIOD tango), where the OIOD is the one with only VICON label.
> > [model_weight] should be the path to the weight directory.

### Environment Setup

1. Please use conda to setup the virtual environment with the following command.

```bash
conda env create -f config.yaml
```

> Please encure the cuda compability and nvidia-driver compability.

### Datasets Setup

1. Please arrange the datasets as the following directory and download them from source respectively.
   - [ADVIO](https://github.com/AaltoVision/ADVIO)
   - [RIDI](https://www.kaggle.com/datasets/kmader/ridi-robust-imu-double-integration)
   - [RoNIN](https://armanasq.github.io/datsets/ronin-datset/)
   - [OxOIOD](http://deepio.cs.ox.ac.uk/)

> The directory should have the similar structure as the following under the directory `datasets`.
```bash
./datasets
├── ADVIO
│   ├── datasets
│   │   ├── advio-01
│   │   ├── advio-02
│   │   ├── advio-03
│   │   ├── advio-04
│   │   ├── advio-05
│   │   ├── advio-06
│   │   ├── advio-07
│   │   ├── advio-08
│   │   ├── advio-09
│   │   ├── advio-10
│   │   ├── advio-11
│   │   ├── advio-12
│   │   ├── advio-13
│   │   ├── advio-14
│   │   ├── advio-15
│   │   ├── advio-16
│   │   ├── advio-17
│   │   ├── advio-18
│   │   ├── advio-19
│   │   ├── advio-20
│   │   ├── advio-21
│   │   ├── advio-22
│   │   ├── advio-23
│   ├── test_list.txt
│   ├── train_list.txt
│   └── val_list.txt
├── OIOD
│   ├── datasets
│   │   ├── __MACOSX
│   │   ├── Oxford Inertial Odometry Dataset
│   │   └── test_list.txt
│   ├── lists
│   │   ├── test_list.txt
│   │   ├── train_list.txt
│   │   └── val_list.txt
├── RIDI
│   ├── datasets
│   │   └── data_publish_v2
│   ├── test_list.txt
│   ├── train_list.txt
│   └── val_list.txt
└── RoNIN
    ├── Data
    │   ├── a000_1
    │   ├── a000_10
    │   ├── a000_11
    │   ├── a000_2
    │   ├── a000_3
    │   ├── a000_4
    │   ├── a000_5
    │   ├── a000_6
    │   ├── a000_7
    │   ├── a000_8
    │   ├── a000_9
    │   ├── a001_1
    │   ├── a001_2
    │   ├── a001_3
    │   ├── a002_1
    │   ├── a002_2
    │   ├── a003_1
    │   ├── a003_2
    │   ├── a003_3
    │   ├── a004_2
    │   ├── a004_3
    │   ├── a005_1
    │   ├── a005_3
    │   ├── a006_2
    │   ├── a007_2
    │   ├── a009_1
    │   ├── a009_2
    │   ├── a009_3
    │   ├── a010_1
    │   ├── a010_2
    │   ├── a010_3
    │   ├── a011_1
    │   ├── a011_2
    │   ├── a011_3
    │   ├── a012_1
    │   ├── a012_2
    │   ├── a012_3
    │   ├── a013_1
    │   ├── a013_2
    │   ├── a013_3
    │   ├── a014_1
    │   ├── a014_2
    │   ├── a014_3
    │   ├── a015_1
    │   ├── a015_2
    │   ├── a015_3
    │   ├── a016_1
    │   ├── a016_3
    │   ├── a017_1
    │   ├── a017_2
    │   ├── a017_3
    │   ├── a018_1
    │   ├── a018_2
    │   ├── a018_3
    │   ├── a019_3
    │   ├── a020_1
    │   ├── a020_2
    │   ├── a020_3
    │   ├── a021_1
    │   ├── a021_2
    │   ├── a021_3
    │   ├── a022_1
    │   ├── a022_2
    │   ├── a022_3
    │   ├── a023_1
    │   ├── a023_2
    │   ├── a023_3
    │   ├── a024_1
    │   ├── a024_3
    │   ├── a025_1
    │   ├── a025_2
    │   ├── a025_3
    │   ├── a026_1
    │   ├── a026_2
    │   ├── a026_3
    │   ├── a027_1
    │   ├── a027_2
    │   ├── a027_3
    │   ├── a028_1
    │   ├── a028_3
    │   ├── a029_1
    │   ├── a029_2
    │   ├── a030_1
    │   ├── a030_3
    │   ├── a031_1
    │   ├── a031_2
    │   ├── a031_3
    │   ├── a032_1
    │   ├── a032_3
    │   ├── a033_1
    │   ├── a033_2
    │   ├── a033_3
    │   ├── a034_1
    │   ├── a034_2
    │   ├── a034_3
    │   ├── a035_1
    │   ├── a035_3
    │   ├── a036_1
    │   ├── a036_2
    │   ├── a036_3
    │   ├── a037_1
    │   ├── a037_3
    │   ├── a038_1
    │   ├── a038_2
    │   ├── a038_3
    │   ├── a039_1
    │   ├── a039_2
    │   ├── a040_2
    │   ├── a040_3
    │   ├── a042_2
    │   ├── a043_1
    │   ├── a043_3
    │   ├── a044_1
    │   ├── a044_2
    │   ├── a044_3
    │   ├── a045_1
    │   ├── a045_2
    │   ├── a045_3
    │   ├── a046_1
    │   ├── a046_2
    │   ├── a046_3
    │   ├── a047_1
    │   ├── a047_2
    │   ├── a047_3
    │   ├── a049_1
    │   ├── a049_2
    │   ├── a049_3
    │   ├── a050_1
    │   ├── a050_3
    │   ├── a051_1
    │   ├── a051_2
    │   ├── a051_3
    │   ├── a052_2
    │   ├── a053_1
    │   ├── a053_2
    │   ├── a053_3
    │   ├── a054_1
    │   ├── a054_2
    │   ├── a054_3
    │   ├── a055_2
    │   ├── a055_3
    │   ├── a056_1
    │   ├── a056_3
    │   ├── a057_1
    │   ├── a057_2
    │   ├── a057_3
    │   ├── a058_1
    │   ├── a058_2
    │   ├── a058_3
    │   ├── a059_1
    │   ├── a059_2
    │   ├── a059_3
    │   ├── seen_subjects_test_set.zip
    │   ├── train_dataset_1.zip
    │   ├── train_dataset_2.zip
    │   └── unseen_subjects_test_set.zip
    ├── frdr-checksums-and-filetypes.md
    ├── FRDR_dataset_538_download_587_202404040428.zip
    ├── freq_statics.csv
    ├── LICENSE.txt
    └── lists
        ├── list_pred.txt
        ├── list_test_seen.txt
        ├── list_test_unseen.txt
        ├── list_train.txt
        └── list_val.txt
```
