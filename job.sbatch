#!/bin/bash

#SBATCH --job-name=train_gpu          # Nombre del trabajo
#SBATCH --output=res_%j.txt           # Archivo de salida (%j se reemplaza por el ID del trabajo)
#SBATCH --error=err_%j.txt            # Archivo de errores
#SBATCH --ntasks=1                    # Número de tareas (procesos)
#SBATCH --cpus-per-task=4             # Número de cores de CPU asignados
#SBATCH --mem=32G                     # Memoria RAM solicitada


SBATCH --nodelist=ag001
SBATCH --gres=gpu:1

module load python3/3.13.2 

source .venv/bin/activate

# Ejecutar el script de entrenamiento
python3 train.py