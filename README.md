# Merging Weak and Active Supervision for Semantic Parsing (WASSP)

This repository contains the experiment code for the AAAI 2020 paper, [Merging Weak and Active Supervision for Semantic Parsing](https://arxiv.org/abs/1911.12986).

<div align="middle"><img src="https://github.com/niansong1996/wassp/blob/master/images/wassp_overview.png" width="80%"></div>

### Note: Ansong Ni has moved from CMU to Yale, please see his new contact info [here](https://github.com/niansong1996/wassp#contact).

## Memory Augmented Policy Optimization (MAPO)
The semantic parsing model we used in our paper is MAPO. If you are looking for more information about MAPO, please refer to this [paper](https://arxiv.org/abs/1807.02322) and [repository](https://github.com/crazydonkey200/neural-symbolic-machines).

## Preparation
To run our code, you need to set up the environment with the following steps:

```
# Go to a convenient location and clone this repo
git clone git@github.com:niansong1996/wassp
cd wassp

# Create the conda environment and install the requirements
conda create --name wassp python=2.7
source activate wassp
pip install requirements.txt
```

Then you need to download the data and pretrained MAPO models (which we use as baseline) from [here](https://drive.google.com/drive/folders/1D3YVStX-DWzZxTyqTAcIfVYXOGaP-20K?usp=sharing). Unzip the downloaded file and put the resulting `data` folder under the `wassp` directory so it looks like this:
```
wassp
    ├── data
        └── ...
    ├── images
    ├── nsm
    ├── nsm.egg-info
    └── table
        └── ...
```
Or you could simply do:
```
cd wassp
bash get_data.sh
```

Finally you need to run `setup.py` so the dependencies are set correctly:
```
source activate wassp
cd wassp
python setup.py develop
```

## Running experiments

### Starting WikiSQL Experiment
```
source activate wassp
cd wassp/table/wikisql/
./run.sh active_learning your_experiment_name
```

### Starting WikiTableQuestions Experiment
```
source activate wassp
cd wassp/table/wtq/
./run.sh active_learning your_experiment_name
```

### Different settings for active learning
To change:
 - Active learning selection heuristic;
 - Forms of extra supervision;
 - Querying budget
 
please see relevant options described in the `run.sh` file.


### Hardware

Our experiments are run on g3.4xlarge AWS instance, which has 16 vCPUs and 122 GiB of memory as well as a M60 GPU with ~8GiB of GPU memory. It takes ~10 hours to run WikiSQL experiments and ~4 hours to run WikiTableQuestions experiments.

If you are running the experiments on a machine with less CPU Computing Power/RAM, we recommend you to decrease the `n_actors`(default=30) parameter in `run.sh`.

### Monitoring training process
You can monitor the training process with tensorboard, specifically:
```
source activate wassp
cd wassp/data/wikisql # or wtq, depending on which dataset are you using
tensorboard --logdir=ouput
```
To see the tensorboard, got to [your AWS public DNS]:6006 and `avg_return_1` is the main metric (accuracy).

An example of our training process is shown in the screenshot below:

<div align="middle"><img src="https://github.com/niansong1996/wassp/blob/master/images/tensorboard_exp_output.png" width="80%"></div>

## Citation
If you use the code in your research, please cite:

    @inproceedings{ni20aaai,
    title = {Merging Weak and Active Supervision for Semantic Parsing},
    author = {Ansong Ni and Pengcheng Yin and Graham Neubig},
    booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
    address = {New York, USA},
    month = {February},
    year = {2020}
    }

    @inproceedings{liang2018memory,
      title={Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing},
      author={Liang, Chen and Norouzi, Mohammad and Berant, Jonathan and Le, Quoc V and Lao, Ni},
      booktitle={Advances in Neural Information Processing Systems},
      pages={10014--10026},
      year={2018}
    }

    @inproceedings{liang2017neural,
      title={Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision},
      author={Liang, Chen and Berant, Jonathan and Le, Quoc and Forbus, Kenneth D and Lao, Ni},
      booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      volume={1},
      pages={23--33},
      year={2017}
    }

## Contact
This code is developed by [Ansong Ni](https://niansong1996.github.io) while he was at CMU but he is now at Yale. So if you find issues in running the code or would like to discuss some part of this work, feel free to contact Ansong at this new email address: [ansong.ni@yale.edu](mailto:ansong.ni@yale.edu).
