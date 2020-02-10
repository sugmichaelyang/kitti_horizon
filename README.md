
If you use this code, please cite [our paper](https://arxiv.org/abs/1907.10014):
```
@article{kluger2019temporally,
  title={Temporally Consistent Horizon Lines},
  author={Kluger, Florian and Ackermann, Hanno and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={arXiv preprint arXiv:1907.10014},
  year={2019}
}
```

## Prerequisites
Get the code:
```
git clone https://github.com/fkluger/kitti_horizon.git
cd kitti_horizon
```

Install required libraries, e.g. via pip or conda:
* pykitti, numpy, scikit-image
* *optional:* pytorch, Pillow, opencv

Download the [KITTI *raw* dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php).

## Usage
### kitti_horizon_raw.py
Contains the *KITTIHorizonRaw* class, which extracts horizon lines in image coordinates from the KITTI raw data. 
It also applies image scaling and padding in order to achieve consistent image resolutions. 
See `python kitti_horizon_raw.py --help` for a demo.

### process_kitti_horizon_raw.py
Pre-processes the KITTI Horizon dataset (using the KITTIHorizonRaw class) and stores it as pickle files, in order to 
speed up the training process. See `python process_kitti_horizon_raw.py --help` for usage.

### kitti_horizon_torch.py
Provides the *KITTIHorizon* class, derived from the PyTorch *Dataset* class. Requires the pre-processed KITTI Horizon 
dataset. For usage as a training/validation dataset in PyTorch. 
See `python kitti_horizon_torch.py --help` for a demo.
