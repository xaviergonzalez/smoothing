#!/bin/bash
# certify models with varying amounts of noise
#command line first arg: hidden layer size
#command line second arg: linear or nonlinear (0 is linear, 1 is nonlinear)
#g gives the gpu
#N gives the number of examples to use for certify
#S gives how many examples to skip through for certify
g=0
N=100000
S=100
for i in 0.12 0.25 0.5 1; # train noise, data
do
   for j in 0 0.12 0.25 0.5 1; #train noise, hidden layer
   do
	   if (($2==0))
           then
		   fm="mnist_models/linear/hlayer${1}/${i}-${j}/checkpoint.pth.tar"
                   fr="mnist_results/linear"
	   else
                   fm="mnist_models/nonlinear/hlayer${1}/${i}-${j}/checkpoint.pth.tar"
                   fr="mnist_results/nonlinear"
	   fi
	   fr="${fr}/hlayer${1}"
	   fmin="${fr}/min/train-${i}-${j}"
	   fmean="${fr}/mean/train-${i}-${j}"
           mkdir -p $fmin
           mkdir -p $fmean
	   CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist $fm $i $fmin/test-$i-$j --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j --layered_GNI --min &
	   g=$(((g+1)%4))
           if (($g==3))
           then
		   CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist $fm $i $fmean/test-$i-$j --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j --layered_GNI --mean
		   wait
	   else
                   CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist $fm $i $fmean/test-$i-$j --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j --layered_GNI --mean &
           fi
	   g=$(((g+1)%4))
   done
done

