# Parameter-Agnostic Optimization under Relaxed Smoothness

This repository contains the experiments supporting our theoretical findings. See the paper for more details: [ Parameter-Agnostic Optimization under Relaxed Smoothness ]( https://arxiv.org/abs/2311.03252 ).

The repository is based on the language modeling part of [ this repo]( https://github.com/zbh2047/clipping-algorithms/tree/master ), which in turn is based on [ the AWD-LSTM repo ]( https://github.com/manuvn/lpRNN-awd-lstm-lm ).

## Training

For our considered algorithm (NSGD-M), simply run

```
python main_lstm.py --batch_size 20 --data [data_folder] --dropouti 0.4 --dropouth 0.25 --seed 2020 --lr 30 --gamma 7.5 --momentum 0.0 --algo sgd_clip --epochs 250
```

Here the `[data_folder]` is the data folder containing training set and validation set.

For other algorithms, change `--algo` command.

##  Citation

If you use this code or our results in your research, please cite as appropriate: 

```
@article{hubler2023parameter,
  title={Parameter-Agnostic Optimization under Relaxed Smoothness},
  author={H{\"u}bler, Florian and Yang, Junchi and Li, Xiang and He, Niao},
  journal={arXiv preprint arXiv:2311.03252},
  year={2023}
}
```