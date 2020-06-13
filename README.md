# NLU Project 1 - Group 35 - ETHZ 2019

## Running Instructions

#### Setup on Leonhard

1. Load Modules
```
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
module load cuda/10.0.130
module load cudnn/7.5
```

2. Create Virtual Environmentwith the files requirements.txt

#### Exercise 1a
```
python main.py --experiment a --mode train evaluate --epochs 5
```

#### Exercise 1b
```
python main.py --experiment b --mode train evaluate --epochs 5
```

#### Exercise 1c and Exercise 2
```
python main.py --experiment c --mode train evaluate generate --epochs 5
```

##### Run Modes Individually
The different modes can also be run individually. Simply use the --id argument. (if id argument not given in train mode, will create id automatically)

```
python main.py --experiment c --mode train --epochs 5 --id final

python main.py --experiment c --mode evaluate --id final

python main.py --experiment c --mode generate --id final
```

##### Run Additional Training Epochs
To run additional epochs in train mode, simply call the command again with the same id.
```
python main.py --experiment c --mode train --epochs 5 --id final

python main.py --experiment c --mode train --epochs 2 --id final
```

## Authors
- Brunner Lucas	brunnelu@student.ethz.ch
- Blatter Philippe	pblatter@student.ethz.ch
- Küchler Nicolas	kunicola@student.ethz.ch
- Fynn Faber	faberf@student.ethz.ch

## Notes

- To it edit the run.sh file to have the rights commands for the run
- tensorboard is supported: tensorboard --logdir logs
- module load cuda/10.0.130 module load cudnn/7.5
