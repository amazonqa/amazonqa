#!/bin/bash

data_dir="../../../data"
out_dir="data"
parts=('train' 'val' 'test')
parts=('val')
num_process=70

for part in "${parts[@]}"; do
	echo "processing $part"
	qar_all=$data_dir/$part"-qar_squad_all.jsonl"
	
	num_lines=($(wc -l $qar_all))
	((num_splits = num_lines/$num_process))
	echo "Total number of lines $num_lines"
	
	rm $out_dir/$part-split-*
	split -l $num_splits $qar_all "$out_dir/$part-split-"
	echo "Split Completed"

	for input_file in $out_dir/$part-split-*; do
		echo $input_file
		eval_file="$input_file.eval_file"
		examples_file="$input_file.examples_file"
		python3 convert_rnet.py --file $input_file --eval_file $eval_file --examples_file $examples_file &
	done
	wait

	echo "Creation Completed"

	eval_all=$out_dir/$part"_eval.jsonl"
	examples_all=$out_dir/$part"_examples.jsonl"
	echo $eval_all
	echo $examples_all

	for output_file in $out_dir/$part-split-*.eval_file; do
		cat $output_file
	done > $eval_all
	wait

	for output_file in $out_dir/$part-split-*.examples_file; do
		cat $output_file
	done > $examples_all
	wait

	echo "Cat Completed"

	shuf $eval_all -o $eval_all
	shuf $examples_all -o $examples_all

	rm $out_dir/$part-split-*
	echo "Shuffle and Delete Completed"
done


