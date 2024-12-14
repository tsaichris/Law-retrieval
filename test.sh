#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o out.out				
#PBS -e err.out		
#PBS -N law_retreived

cd ~/law

source ~/.bashrc			
conda activate law	

module load cuda-12.4
#python3 test.py
python3 embed_database_e5_chunk.py  2>&1 | tee log.txt 




