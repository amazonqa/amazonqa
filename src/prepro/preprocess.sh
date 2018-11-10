#!/bin/bash

categories=("Automotive" "Baby" "Beauty" "Cell Phones and Accessories" "Clothing Shoes and Jewelry" "Electronics" "Grocery and Gourmet Food" "Health and Personal Care" "Home and Kitchen" "Musical Instruments" "Office Products" "Patio Lawn and Garden" "Pet Supplies" "Sports and Outdoors" "Tools and Home Improvement" "Toys and Games" "Video Games")
categories=("Baby")

data_dir="../../data/"

mkdir -p $data_dir

for i in "${categories[@]}"; do
    echo "$i"
    python3 preprocess_data.py --download 1 --data_dir $data_dir --category $i &
done

wait

output_file="$data_dir/qa_reviews_all.jsonl.gz"

for i in "${categories[@]}"; do
    echo "$i"
    file="$data_dir/qa_reviews$i.jsonl.gz"
    cat $file > $output_file
done