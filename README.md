##To it edit the run.sh file to have the rights commands for the run
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" < run.sh 

#tensorboard is supported
tensorboar --logdir logs