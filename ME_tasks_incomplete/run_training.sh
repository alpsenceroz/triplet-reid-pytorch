#!/bin/bash

# Define arrays of hyperparameters
learning_rates=(1e-4 3e-5)
triplet_weights=(0.7 0.5)
kl_weights=(0.02 0.01)
reconstruction_weights=(0.5 0.3)
bce_weights=(1.0 0.8)
epoch_numbers=(5000 10000)

# Loop over each combination of hyperparameters
for lr in "${learning_rates[@]}"; do
    for triplet in "${triplet_weights[@]}"; do
        for kl in "${kl_weights[@]}"; do
            for reconstruction in "${reconstruction_weights[@]}"; do
                for bce in "${bce_weights[@]}"; do
                    for epoch in "${epoch_numbers[@]}"; do
                        echo "Training with lr=$lr, triplet=$triplet, kl=$kl, reconstruction=$reconstruction, bce=$bce, epochs=$epoch"
                        python train.py --lr $lr --triplet $triplet --kl $kl --reconstruction $reconstruction --bce $bce --epochNumber $epoch
                    done
                done
            done
        done
    done
done
