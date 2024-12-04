# TGRNet-REP
This is the official code for TGRNet-REP.
## Requirements

```python==3.8```

```einops==0.3.0```

```cryptography==43.0.0```

```torch==2.3.0```

```numpy==1.24.4```

```pandas==2.0.3```

```opencv-python==4.4.0.46```

## Dataset Preparation
* [WHU-Stereo](https://github.com/Sheng029/WHU-Stereo)
* [US3D](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
  
  Note：If you want to use our preprocessed US3D dataset, please download：

## Pretrained weights
The pretrained weights for WHU-Stereo and US3D datasets are available at: 

## Evaluation WHU-Stereo
```Shell
python evaluate_WHU.py --test_left_dir dataset/WHU-Stereo/with_GT/test_all/left --test_right_dir dataset/WHU-Stereo/test_all/right --test_disp_dir dataset/WHU-Stereo/test_all/disp --test_save_path results/whu --device cuda
```
```test_left_dir```: Directory containing the left image

```test_right_dir```: Directory containing the right image

```test_disp_dir```: Directory containing the disparity ground truth

## Evaluation US3D
```Shell
python evaluate_US3D.py --test_left_dir dataset/US3D/test_all/left --test_right_dir dataset/US3D/test_all/left --test_disp_dir dataset/US3D/test_all/left --test_save_path results/us3d --device cuda
```
```test_left_dir```: Directory containing the left image

```test_right_dir```: Directory containing the right image

```test_disp_dir```: Directory containing the disparity ground truth

