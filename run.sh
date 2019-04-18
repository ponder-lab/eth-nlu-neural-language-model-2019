#load python and cuda version
module load python_gpu/3.6.4 cuda/10.0.130

#enter virtual env
source ./venv/bin/activate

#actualy running something
cd code/
python3 main.py 1 a

#cleanup
deactivate
#python
