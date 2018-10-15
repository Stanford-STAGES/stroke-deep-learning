#!/bin/bash
FILE=$1
while read -ru 3 LINE; do
    sbatch inference.sh  "$LINE"
done 3< "$FILE"
