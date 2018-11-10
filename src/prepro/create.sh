#!/bin/bash

categories=("Automotive" "Baby" "Beauty" "Cell Phones and Accessories" "Clothing Shoes and Jewelry" "Electronics" "Grocery and Gourmet Food" "Health and Personal Care" "Home and Kitchen" "Musical Instruments" "Office Products" "Patio Lawn and Garden" "Pet Supplies" "Sports and Outdoors" "Tools and Home Improvement" "Toys and Games" "Video Games")
categories=("Baby")

data_dir="../../data/"

for i in "${categories[@]}"; do
    echo "$i"
    input_file="$data_dir/qa_reviews_$i.jsonl.gz"
    output_file="$data_dir/qar_$i.jsonl"
    python3 create_data.py --input_file $input_file --output_file $output_file --review_select_mode "bm25" --review_select_num 10
done
