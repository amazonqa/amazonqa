#!/bin/bash

categories=("Automotive" "Baby" "Beauty" "Cell_Phones_and_Accessories" "Clothing_Shoes_and_Jewelry" "Electronics" "Grocery_and_Gourmet_Food" "Health_and_Personal_Care" "Home_and_Kitchen" "Musical_Instruments" "Office_Products" "Patio_Lawn_and_Garden" "Pet_Supplies" "Sports_and_Outdoors" "Tools_and_Home_Improvement" "Toys_and_Games" "Video_Games")
#categories=("Baby")

data_dir="../../data"

mkdir -p $data_dir

for i in "${categories[@]}"; do
    echo "$i"
    python3 preprocess_data.py --download 1 --data_dir $data_dir --category "$i" &
done

wait
echo "Preprocess Completed"

qar_products_all="$data_dir/qar_products_all.jsonl"
rm $qar_products_all

for i in "${categories[@]}"; do
    file="$data_dir/qar_products_$i.jsonl"
    cat $file
done > $qar_products_all

echo "Cat Completed"

shuf $qar_products_all -o $qar_products_all
echo "Shuffle Completed"

num_lines=($(wc -l $qar_products_all))
parts=('train' 'val' 'test')

part=0
percentSum=0
currentLine=1
for percent in 80 10 10; do
        ((percentSum += percent))
        ((nextLine = num_lines * percentSum / 100))
        if ((nextLine < currentLine)); then
                printf "" # create empty file
        else
                sed -n "$currentLine,$nextLine"p "$qar_products_all"
        fi > "$data_dir/${parts[$part]}-qar_products_all.jsonl"
        ((currentLine = nextLine + 1))
        ((part++))
done
echo "Split Completed"

