#!/bin/bash

FILE=$1

while read -ru 3 LINE; do
    #echo "$LINE"
    #IFS=' '; arrIN=($LINE); unset IFS;
    #echo ${arrIN[5]}${arrIN[7]//'='}
    #sbatch train.sh  "$LINE" "${arrIN[5]}${arrIN[7]//'='}"
    sbatch inference.sh  "$LINE"
done 3< "$FILE"
