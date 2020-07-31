
# code to run onnx on 100M samples
sbatch --array=0-600 ONYX_BERT_labels_deploy_100M_random.sbatch

- you might be able to explore larger arrays - 1000 give me an slurm admin error, but I also get 162 machines max at the same time so it doens't matter much
- fyi the smallest memory I was able to use to run code was 5gb


# git clone manu's branch of simple transformers
git clone git@github.com:manueltonneau/simpletransformers.git
cd simpletransformers
pip install -e .

# do the same for transformers
https://github.com/manueltonneau/transformers.git

# make sure pytorch 1.5 is installed (other versions are weird)
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install onyx
pip install onnxruntime_tools onnxruntime


(# my pyenv is here but don't use it while i'm using it for a job array as it can corrupt my job (happened with when I tried to use yours before)
(# source /scratch/da2734/pyenv_dval_wb_twitter/py3.7/bin/activate)