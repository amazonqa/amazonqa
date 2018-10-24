NUM_PROCESSES=5
for ((i = 0 ; i < $NUM_PROCESSES ; i++ )); do
    echo "Running cmd: python convert_squad.py --num_processes $NUM_PROCESSES --process_idx $i &;"
    python convert_squad.py --num_processes $NUM_PROCESSES --process_idx $i &
done
