# NLU Project 1 - ETHZ 2019

## Running Instructions

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


## Notes

- To it edit the run.sh file to have the rights commands for the run
- tensorboard is supported: tensorboard --logdir logs
- module load cuda/10.0.130 module load cudnn/7.5
