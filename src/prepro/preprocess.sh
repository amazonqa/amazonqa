#!/bin/bash

#CATEGORIES="'Automotive', 'Baby', 'Beauty', 'Cell Phones and Accessories', 'Clothing Shoes and Jewelry', 'Electronics', 'Grocery and Gourmet Food', 'Health and Personal Care', 'Home and Kitchen', 'Musical Instruments', 'Office Products', 'Patio Lawn and Garden', 'Pet Supplies', 'Sports and Outdoors', 'Tools and Home Improvement', 'Toys and Games', 'Video Games'"
# array=("item 1" "item 2" "item 3")

categories=("Automotive")
for i in "${categories[@]}"; do
    echo "$i"
    python3 process_raw_qa.py --download 1 --raw_qa_dir '../../data/raw_qa/' --clean_qa_dir '../../data/clean_qa/' --category $i
done
