#only when running for the first time

module load python_gpu/3.6.4 cuda/10.0.130

pip install --user pipenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip3 install tf-nightly-gpu-2.0-preview
pip3 install gensim
deactivate