#!/bin/bash
valid=true
count=1
while [ $valid ]
do
echo $count
if [ $count -eq $((10**99999999)) ];
then
break
fi
((count++))
done
