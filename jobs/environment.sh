module purge
module load 2022
module load Anaconda3/2022.05

cd $HOME/fast_robust_early_exit

export CUBLAS_WORKSPACE_CONFIG=:4096:8

conda activate dl2

pip install -r requirements.txt
