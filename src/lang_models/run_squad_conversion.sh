NUM_PROCESSES=5
for ((i = 0 ; i < $NUM_PROCESSES ; i++ )); do
    echo "Running cmd: python convert_squad.py --num_processes $NUM_PROCESSES --process_idx $i > temp/out_$i.log &;"
    python convert_squad.py --category Video_Games --mode train --max_num_products 10 --num_processes $NUM_PROCESSES --process_idx $i > temp/out_$i.log &
done
