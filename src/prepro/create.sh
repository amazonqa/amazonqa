#!/bin/bash

data_dir="../../data"
parts=('train' 'dev' 'test')
parts=('train')
num_process=1

for part in "${parts[@]}"; do
	echo $part
	qar_all_categories="$data_dir/$part-qar_all_categories.jsonl"
	num_lines=($(wc -l $qar_all_categories))
	echo $num_lines
	((num_splits = num_lines/$num_process))
	echo $num_splits
	split -l $num_splits $qar_all_categories "$data_dir/$part-input-"

	for input_file in $data_dir/$part-input-*; do
		echo $input_file
		output_file="$input_file.out"
		python3 create_data.py --input_file $input_file --output_file $output_file --review_select_mode "bm25" --review_select_num 10 --review_max_len 100 &
	done

	wait

	final_output_file="$data_dir/$part-qar_all.jsonl" 
	for output_file in $data_dir/$part-input-*.out; do
		echo $output_file
		cat $output_file > $final_output_file
	done

done


