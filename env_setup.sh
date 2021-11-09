conda create -n clpf python=3.7.10 --file requirements.txt -c defaults -c conda-forge 
conda activate clpf
conda install -c pytorch pytorch==1.7.1
pip install torchdiffeq==0.2.1
pip install git+https://github.com/google-research/torchsde.git@f965bc9a716a86bce45c3d410bc9eaf22283037e
