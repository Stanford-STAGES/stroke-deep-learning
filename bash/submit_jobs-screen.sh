#!/bin/bash

FILE=$1

while read -ru 3 LINE; do
    sbatch train-screen.sh  "$LINE"
done 3< "$FILE"
