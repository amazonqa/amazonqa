#!/bin/bash

data_dir="../../data"
parts=('train' 'dev' 'test')
parts=('train')
num_process=32

for part in "${parts[@]}"; do
	echo "processing $part"
	qar_products_all="$data_dir/$part-qar_products_all.jsonl"
	
	num_lines=($(wc -l $qar_products_all))
	((num_splits = num_lines/$num_process))
	echo "Total number of lines $num_lines"
	
	rm $data_dir/$part-split-*
	split -l $num_splits $qar_products_all "$data_dir/$part-split-"
	echo "Split Completed"

	for input_file in $data_dir/$part-split-*; do
		echo $input_file
		output_file="$input_file.out"
		python3 create_data.py --input_file $input_file --output_file $output_file --review_select_mode "bm25" --review_select_num 10 --review_max_len 100 &
	done

	wait
	echo "Creation Completed"

	qar_all="$data_dir/$part-qar_all.jsonl" 
	for output_file in $data_dir/$part-split-*.out; do
		echo $output_file
		cat $output_file > $qar_all
	done
	echo "Cat Completed"

	shuf $qar_all -o $qar_all
	rm $data_dir/$part-split-*
	echo "Shuffle and Delete Completed"
done


