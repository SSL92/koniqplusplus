# KonIQ++: Boosting No-Reference Image Quality Assessment in the Wild by Jointly Predicting Image Quality and Defects

Source code for the BMVC'21 oral "[KonIQ++: Boosting No-Reference Image Quality Assessment in the Wild by Jointly Predicting Image Quality and Defects](https://www.bmvc2021-virtualconference.com/assets/papers/0868.pdf)".

The KonIQ++ database can be found [here](http://database.mmsp-kn.de/koniq-image-defects-database.html).

Our pretrained model can be found [here](https://drive.google.com/file/d/1NVFPPFM7n2P1i_LSw-F3JX2mn-Gn3PKw/view?usp=sharing).

## Installation

Install dependencies

```
pip install -r requirements.txt
```

The code also requires apex, please refer to this [repository](https://github.com/NVIDIA/apex) for instructions.

## Usages

### Training & Testing on IQA databases

To reproduce reported results in the paper, first download the koniq++database.csv file from [KonIQ++ Webpage](http://database.mmsp-kn.de/koniq-image-defects-database.html) and put it in your own path for KonIQ-10k dataset, then run:

```
python main.py --dataset KonIQ-10k --resize --lr 1e-4 -bs 8 -e 25 --ft_lr_ratio 0.1 --loss_type norm-in-norm --p 1 --q 2 --koniq_root 'YOUR OWN PATH'
```

Model will be trained and tested on the KonIQ++ dataset.

If you want to cross test on either CLIVE or SPAQ dataset, add following options:
* `--test_on_clive`: Add option to cross test on CLIVE dataset after training on KonIQ++. 
* `--clive_root 'YOUR OWN PATH'`: Root path for CLIVE dataset.
* `--test_on_spaq`: Add option to cross test on SPAQ dataset after training on KonIQ++. 
* `--spaq_root 'YOUR OWN PATH'`: Root path for SPAQ dataset.

Note: We have conducted user screenings to the released version KonIQ++ labels, so the results might have small fluctuations regarding to the reported performances. To perfectly reproduce performances in the paper, you can use the Koniq++_old.csv label in the old_label folder as substitution.

### Predicting Single Image Quality and Visualizing 

To predict single image quality, run:

```
python test_image.py --root_path 'IMAGE PATH' --img_name 'TEST IMAGE NAME' --resize --p 1 --q 2
```

To visualize feature heatmaps from two side networks, add option `--save_heatmap`. Heatmaps will be saved under the image root path.

## Citation
If you find this work useful for your research, please cite our paper.


## Acknowledgements
This code is largely implemented based upon [norm-in-norm IQA](https://github.com/lidq92/LinearityIQA), we thank them for their sharing.
