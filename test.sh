#!/bin/bash
#test file for writing scripts

i=1
echo $i
i=$((i+1))
echo $i
i=$(((i+1)%2))
echo $i

