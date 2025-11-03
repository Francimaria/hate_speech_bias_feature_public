#!/bin/bash

#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=exp_wh
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH --partition short
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output_wh.txt
#SBATCH --error=job_error_wh.txt

#carregar versão python
module load Python/3.10

# criar ambiente
python -m venv $HOME/envteste/

# ativar ambiente
source $HOME/envteste/bin/activate

#instalar dependencias
pip install -r $HOME/sparse_ae_bias/requirements.txt


#executar .py
python $HOME/hate_speech_bias_feature/feature_extraction.py 
python $HOME/hate_speech_bias_feature/train_clf_param_select.py
python $HOME/hate_speech_bias_feature/train_clf.py 


