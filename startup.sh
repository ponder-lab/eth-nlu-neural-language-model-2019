#only when running for the first time

module load python_gpu/3.6.4 cuda/10.0.130 cudnn/7.5

pip install --user pipenv
virtualenv -p python3 ./venv
source ./venv/bin/activate
pip3 install tf-nightly-gpu-2.0-preview
pip3 install tensorflow==2.9.3
pip3 install gensim
pip3 install pandas
pip3 install scipy==1.10.0
deactivate
