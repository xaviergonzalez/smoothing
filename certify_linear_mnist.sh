#!/bin/bash
# certify models with varying amounts of noise
#command line arg: size of hidden layer
g=0
for i in 0.12 0.25 0.5 1; # train noise, data
do
   for j in 0 0.12 0.25 0.5 1; #train noise, hidden layer
   do
	   froot="mnist_results/linear/hlayer${1}/"
	   fmin="${froot}min/train-${i}-${j}"
	   fmean="${froot}min/train-${i}-${j}"
           mkdir -p mnist_results/linear/hlayer444/min/train-$i-$j/
           mkdir -p mnist_results/linear/hlayer444/mean/train-$i-$j/
	   for k in 0.12 0.25 0.5 1; #test noise, data
	   do
		   for l in 0 0.12 0.25 0.5 1; #test noise, hidden layer
		   do
			python code/certify.py mnist mnist_models/linear/hlayer444/$i-$j/checkpoint.pth.tar $k mnist_results/linear/hlayer444/min/train-$i-$j/test-$k-$l --alpha 0.001 --N0 100 --N 1000000 --skip 100 --batch 400 --gpu $g --noise_std_lst 0 $l --layered_GNI --min &
                   	g=$(((g+1)%4))
                   	if (($g==3))
                   	then
                           python code/certify.py mnist mnist_models/linear/hlayer444/$i-$j/checkpoint.pth.tar $k mnist_results/linear/hlayer444/mean/train-$i-$j/test-$k-$l --alpha 0.001 --N0 100 --N 1000000 --skip 100 --batch 400 --gpu $g --noise_std_lst 0 $l --layered_GNI --mean
                   	else
                           python code/certify.py mnist mnist_models/linear/hlayer444/$i-$j/checkpoint.pth.tar $k mnist_results/linear/hlayer444/mean/train-$i-$j/test-$k-$l --alpha 0.001 --N0 100 --N 1000000 --skip 100 --batch 400 --gpu $g --noise_std_lst 0 $l --layered_GNI --mean &
                   	fi
			g=$(((g+1)%4))
		   done
 	   done
   done
done

