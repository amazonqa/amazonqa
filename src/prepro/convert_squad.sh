#!/bin/bash

data_dir="../../data"
parts=('train' 'val' 'test')
parts=('test')
num_process=80

for part in "${parts[@]}"; do
	echo "processing $part"
	qar_all="$data_dir/$part-qar_all.jsonl"
	
	num_lines=($(wc -l $qar_all))
	((num_splits = num_lines/$num_process))
	echo "Total number of lines $num_lines"
	
	rm $data_dir/$part-split-*
	split -l $num_splits $qar_all "$data_dir/$part-split-"
	echo "Split Completed"

	for input_file in $data_dir/$part-split-*; do
		echo $input_file
		output_file="$input_file.out"
		python3 convert_squad.py --input_file $input_file --output_file $output_file &
	done

	wait
	echo "Creation Completed"

	qar_squad_all="$data_dir/$part-qar_squad_all.jsonl" 
	for output_file in $data_dir/$part-split-*.out; do
		cat $output_file
	done > $qar_squad_all
	echo "Cat Completed"

	shuf $qar_squad_all -o $qar_squad_all
	rm $data_dir/$part-split-*
	echo "Shuffle and Delete Completed"
done


