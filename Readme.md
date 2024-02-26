# Parameter-Agnostic Optimization under Relaxed Smoothness

This repository contains the experiments supporting our theoretical findings. See the paper for more details: [ Parameter-Agnostic Optimization under Relaxed Smoothness ]( https://arxiv.org/abs/2311.03252 ).

The repository is based on the language modeling part of [this repository]( https://github.com/zbh2047/clipping-algorithms ), which in turn is based on [ the AWD-LSTM repository]( https://github.com/manuvn/lpRNN-awd-lstm-lm ).

## Training

For our considered algorithm (NSGD-M), simply run

```
python main_lstm.py --data [data_folder] --result_dir result/ --epochs 300 --algo nsgdm --lr 25.0 --lr_decay 0.75 --mom_decay 0.5 --seed 1970
```

Here the `[data_folder]` is the data folder containing training set and validation set.

For other algorithms, change the `--algo` parameter.

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