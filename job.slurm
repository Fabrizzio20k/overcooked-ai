#!/bin/bash

#SBATCH --job-name=train_gpu          # Nombre del trabajo
#SBATCH --output=res_%j.txt           # Archivo de salida (%j se reemplaza por el ID del trabajo)
#SBATCH --error=err_%j.txt            # Archivo de errores
#SBATCH --ntasks=1                    # Número de tareas (procesos)
#SBATCH --cpus-per-task=4             # Número de cores de CPU asignados
#SBATCH --mem=32G                     # Memoria RAM solicitada
#SBATCH --partition=gpu               # Obligatorio para acceder a los nodos con GPU
#SBATCH --gres=gpu:a100:1             # Pide exclusivamente la poderosa NVIDIA A100

module load python3/3.13.2 

source .venv/bin/activate

# Ejecutar el script de entrenamiento
python3 train.py