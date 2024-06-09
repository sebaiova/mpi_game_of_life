#!/bin/bash

index_row=$(ls subgrid*.out | sort -V | tail -1 | sed 's/_/ /g' | awk '{print $2}')

echo -n > grid.out
for ((i = 0; i <= index_row; i++))
do
   array=($(ls subgrid_${i}_*.out | sort -V))
   echo ${array[@]}
   paste ${array[@]} -d '' >> grid.out
done