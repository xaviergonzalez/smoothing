#!/bin/bash
# ceritfy models with varying hidden layer sizes, and whether or not nonlinearity applied
for h in 20 44 200 444;
do
   for n in 0 1;
   do
           ./certify_mnist.sh $h $n
           wait
   done
done

