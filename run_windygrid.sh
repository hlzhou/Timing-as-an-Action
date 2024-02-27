#!/bin/bash

for ((i=1;i<=50;i++)); 
    do 

    echo "================ WINDY SEED: ${i} ================"
    python main.py --env "custom_windygrid" --seed $i \
    --timing_smart --timing_dumb --model_free \
    --gamma 0.99 \
    --action_cost 1 \
    --greedy_delay uniform;
done


# exploration phase
for ((i=1;i<=50;i++)); 
    do 

    echo "================ WINDY SEED: ${i} ================"
    python main.py --env "custom_windygrid" --seed $i \
    --timing_smart --timing_dumb --model_free \
    --gamma 0.99 \
    --action_cost 1 \
    --explore 50;
done