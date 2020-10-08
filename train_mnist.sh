#!/bin/bash
# train models with varying amounts of noise
#first command line argument gives size of hidden layer
#second command line argument gives whether nonlinearity applied or not (0 is linear, 1 is nonlinear)
#g gives the gpu
g=0
for i in 0 0.12 0.25 0.5 1;
do	
   for j in 0 0.12 0.25 0.5 1;
   do
	   if (($2==0)) 
	   then
		   f="mnist_models/linear"
	   else
		   f="mnist_models/nonlinear"
	   fi
	   f="${f}/hlayer${1}/${i}-${j}"
	   mkdir -p $f
	   python code/train.py mnist mnist_mlp $f --batch 400 --noise_sd 0 --gpu $g --hidden_size $1 --nonlinear $2 --noise_std_lst $i $j > $f/train_log.txt &
	   if (($g==3))
	   then
		   wait
	   fi
	   g=$(((g+1)%4))
   done
done

