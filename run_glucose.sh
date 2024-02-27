#!/bin/bash

for ((i=1;i<=50;i++)); 
    do 
    echo "================ STRONGER SEED: ${i} ================"


    python main.py --env "glucose" --seed $i \
    --timing_smart --timing_dumb --model_free \
    --cpu --epochs 200 \
    --epsilon_max 0.1 --epsilon_min 0.1 \
    --T_lr 0.01 \
    --action_cost 0.5 \
    --transition_convergence 1e-5 \
    --gamma 0.99 \
    --greedy_delay uniform \
    --cpu

    done


# exploration phase
for ((i=1;i<=50;i++)); 
    do 
    echo "================ STRONGER SEED: ${i} ================"


    python main.py --env "glucose" --seed $i \
    --timing_smart \
    --cpu --epochs 200 \
    --epsilon_max 0.1 --epsilon_min 0.1 \
    --T_lr 0.01 \
    --action_cost 0.5 \
    --transition_convergence 1e-5 \
    --gamma 0.99 \
    --explore 50 \
    --cpu

    done


