#!/bin/bash
#sanity check for whether certify output files are accurate
#command line first arg: hidden layer size
#command line second arg: linear or nonlinear (0 is linear, 1 is nonlinear)
#g gives the gpu
g=0
for i in 0.12 0.25 0.5 1; # train noise, data
do
   for j in 0 0.12 0.25 0.5 1; #train noise, hidden layer
   do
           if (($2==0))
           then
		   fm="mnist_models/linear/hlayer${1}/${i}-${j}/checkpoint.pth.tar"
                   fr="mnist_results/linear"
		   fch="mnist_checks/linear"
           else
                   fm="mnist_models/nonlinear/hlayer${1}/${i}-${j}/checkpoint.pth.tar"
		   fr="mnist_results/nonlinear"
		   fch="mnist_checks/nonlinear"
           fi
           fr="${fr}/hlayer${1}"
	   fch="${fch}/hlayer${1}"
           fmin="${fr}/min/train-${i}-${j}"
	   fchmin="${fch}/min/train-${i}-${j}"
           fmean="${fr}/mean/train-${i}-${j}"
	   fchmean="${fch}/mean/train-${i}-${j}"
           mkdir -p $fmin
           mkdir -p $fmean
	   mkdir -p $fchmin
           mkdir -p $fchmean
           CUDA_VISIBLE_DEVICES=$g python code/check.py mnist $fm $i $fmin/test-$i-$j $fchmin/test-$i-$j --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j &
           g=$(((g+1)%4))
           if (($g==3))
           then
                CUDA_VISIBLE_DEVICES=$g python code/jac_check.py mnist $fm $i $fmean/test-$i-$j $fchmean/test-$i-$j --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j
                wait
           else
           	CUDA_VISIBLE_DEVICES=$g python code/jac_check.py mnist $fm $i $fmean/test-$i-$j $fchmean/test-$i-$j --batch 400 --hidden_size $1 --nonlinear $2 --noise_std_lst 0 $j & 
	   fi
           g=$(((g+1)%4))
   done
done

