#load python and cuda version
module load python_gpu/3.6.4 cuda/10.0.130 cudnn/7.5

#enter virtual env
source ./venv/bin/activate

#actualy running something
cd code/
python main.py --experiment a --mode train evaluate --epochs 5

#cleanup
deactivate
#python
