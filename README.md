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


## Notes

- To it edit the run.sh file to have the rights commands for the run
- tensorboard is supported: tensorboar --logdir logs
- module load cuda/10.0.130 module load cudnn/7.5
