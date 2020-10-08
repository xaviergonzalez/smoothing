#!/bin/bash
# check the certification results of models with varying hidden layer sizes, and whether or not nonlinearity applied
for h in 20 44 200 444;
do
   for n in 0 1;
   do
           ./check_mnist.sh $h $n
           wait
   done
done

