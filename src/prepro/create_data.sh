#!/bin/bash

data_dir="../../data"
parts=('train' 'val' 'test')
parts=('train')
num_process=72

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

	qar_all_temp="$data_dir/$part-qar_all.jsonl_temp" 
	qar_all="$data_dir/$part-qar_all.jsonl" 

	for output_file in $data_dir/$part-split-*.out; do
		cat $output_file
	done > $qar_all_temp
	echo "Cat Completed"

	shuf $qar_all_temp -o $qar_all_temp
	rm $data_dir/$part-split-*
	echo "Shuffle and Delete Completed"

	python3 convert0.py --input_file $qar_all_temp --output_file $qar_all
	rm $qar_all_temp

	echo "qid done"
done


