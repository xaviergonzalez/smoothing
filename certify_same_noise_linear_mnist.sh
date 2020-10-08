#!/bin/bash
# certify with different amounts of test time noise injection that give equal pullback noise (determinant)
#g gives the gpu
#N gives the number of examples to use for certify
#S gives how mnay examples to skip through for certify
#E gives how many epochs to train for 
g=0
N=100000
S=100
E=40
#noise equivalent to 0.12, 0.12
n=0.12
in_noise=(0.12 0.25 0.5 0.556)
out_noise=(0.12 0.094 0.02 0)
#for index in "${!in_noise[@]}";
#do
 #       i=${in_noise[$index]}
  #      j=${out_noise[$index]}
   #     if (($g==3))
    #    then
     #           CUDA_VISIBLE_DEVICES=$g python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
      #          wait
       # else
        #        python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
        #fi
        #g=$(((g+1)%4))
#done
#for index in "${!in_noise[@]}";
#do
 #       k=${in_noise[$index]}
  #      l=${out_noise[$index]}
   #     mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min
    #    mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean
     #   CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --min &
   #     g=$(((g+1)%4))
    #    if (($g==3))
     #   then
      #          CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean
       #         wait
        #else
         #       CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean &
        #fi
        #g=$(((g+1)%4))
# done
#noise equivalent to 0.25, 0.25
n=0.25
in_noise=(0.12 0.25 0.5 1)
out_noise=(0.27 0.25 0.2 0.056)
#for index in "${!in_noise[@]}";
#do
 #       i=${in_noise[$index]}
  #      j=${out_noise[$index]}
   #     if (($g==3))
    #    then
     #           CUDA_VISIBLE_DEVICES=$g python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
      #          wait
      #  else
       #         python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
       # fi
       # g=$(((g+1)%4))
#done
for index in "${!in_noise[@]}";
do
        k=${in_noise[$index]}
        l=${out_noise[$index]}
        mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min
        mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean
        CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --min &
        g=$(((g+1)%4))
        if (($g==3))
        then
                CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean
                wait
        else
                CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean &
        fi
        g=$(((g+1)%4))
done
#noise equivalent to 0.5, 0.5
n=0.5
in_noise=(0.12 0.25 0.5 1)
out_noise=(0.55 0.535 0.5 0.4)
#for index in "${!in_noise[@]}";
#do
 #       i=${in_noise[$index]}
  #      j=${out_noise[$index]}
#	if (($g==3))
#	then
 #       	CUDA_VISIBLE_DEVICES=$g python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
#		wait
#	else
#		python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
#	fi
 #       g=$(((g+1)%4))
#done
#for index in "${!in_noise[@]}";
#do
#	k=${in_noise[$index]}
#	l=${out_noise[$index]}
#	mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min
#	mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean
#	CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --min &
#	g=$(((g+1)%4))
#	if (($g==3))
#	then
#		CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean
#		wait
#	else
#		CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean &
#	fi
 #       g=$(((g+1)%4))
#done
#noise equivalent to 1,1
n=1
in_noise=(0.12 0.25 0.5 1)
#out_noise=(1.11 1.1 1.07 1)
#for index in "${!in_noise[@]}";
#do
 #       i=${in_noise[$index]}
  #      j=${out_noise[$index]}
   #     if (($g==3))
        #then
         #       CUDA_VISIBLE_DEVICES=$g python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
          #      wait
#        else
 #               python code/train.py mnist mnist_mlp mnist_models/linear/hlayer444/$i-$j --batch 400 --noise_std_lst $i $j --epochs $E
  #      fi
   #     g=$(((g+1)%4))
# done
#for index in "${!in_noise[@]}";
#do
 #       k=${in_noise[$index]}
  #      l=${out_noise[$index]}
   #     mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min
 #       mkdir -p same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean
  #      CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/min/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --min &
   #     g=$(((g+1)%4))
    #    if (($g==3))
     #   then
      #          CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean
#                wait
 #       else
  #              CUDA_VISIBLE_DEVICES=$g python code/certify.py mnist mnist_models/linear/hlayer444/$k-$l/checkpoint.pth.tar $k same_pb/mnist/linear/hlayer444/$n/train-$k-$l/mean/test-$k-$l --alpha 0.001 --N0 100 --N $N --skip $S --batch 400 --noise_std_lst 0 $l --layered_GNI --mean &
   #     fi
    #    g=$(((g+1)%4))
#done

