<img src="img/logo_ot.png" width=400, padding=100>

## Oncological TranscripTomE Recognition
### v 1.0

[![Licence](https://img.shields.io/github/license/fcomitani/otter?style=flat-square)](https://github.com/fcomitani/otter/blob/main/LICENSE)
<!--
[![GitHub top language](https://img.shields.io/github/languages/top/fcomitani/otter?style=flat-square)](https://github.com/fcomitani/otter/search?l=python)
%[![Documentation Status](https://readthedocs.org/projects/aroughcun/badge/?version=latest&style=flat-square)](https://aroughcun.readthedocs.io/en/latest/?badge=latest)
-->

This repository contains code to train and run OTTER v1.0, an ensemble of CNN for transcriptome-based classification of tumour subtypes.
For details on how it works and its aims, please see the publication at the end of this file.

To run the pre-trained classifier on your expression samples for inference please visit our [OTTER web app](https://otter.ccm.sickkids.ca/)

### How it works

There are two main files, `otter_train.py` allows you to train a single CNN model at a time, `otter_predict.py` allows you to run inference with an ensemble of trained models.

```
python otter_train.py --data to_train.h5 --labels input_labels.h5 --output output_folder --hparam hyperparameters.json --epochs 50 --batchsize 64 --patience 3 --split .2 --lowvar .99

python otter_predict.py --data to_predict.h5 --outputo output_folder --models models_folder
```

`-h`, `--help` will summon a help message with details on the available options.

The output is a pandas dataframe with the prediction probabilities. The Youden correction awill have to be applied independently after calculating the proper index. 

### Dependencies

The classifiers were originally built and trained with the following libraries.
More recent versions can be used, but beware you will need to update the code accordingly.

```
- scikit-learn	== 0.22.1
- tensorflow	== 1.12.0
- keras		== 2.2.4
- hyperopt	== 0.1.1
```

### Citation

When using these files or the associated web app, please cite

> F. Comitani, J. O. Nash, S. Cohen-Gogo, A. Chang, T. T. Wen, A. Maheshwari, B. Goyal, E. S. L. Tio, K. Tabatabaei, L. Brunga, J. E. G. Lawrence, P. Balogh, A. Flanagan, S. Teichmann, V. Ramaswamy, J. Hitzler, J. Wasserman, R. A. Gladdy, B. C. Dickson, U. Tabori, M. J. Cowley, S. Behjati, D. Malkin, A. Villani, M. S. Irwin and A. Shlien, "Multi-scale transcriptional clustering and heterogeneity analysis reveal diagnostic classes of childhood cancer" (under review). 

