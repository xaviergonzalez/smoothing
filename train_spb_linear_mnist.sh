#!/bin/bash
# train models with specific pairings of noise
g=0
in_noise=(0.12 0.25 1.0)
out_noise=(0.55 0.535 0.4)
for index in "${!in_noise[@]}";
do
        i=${in_noise[$index]}
        j=${out_noise[$index]}
        python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --gpu $g --noise_std_lst $i $j &
        g=$((g+1))
done
