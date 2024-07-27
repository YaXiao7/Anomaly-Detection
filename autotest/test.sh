mkdir -p results

for i in $(seq 200); do
    python demo.py  > results/log_$i.txt 2>&1 &
    # sleep $((RANDOM % 5 + 1))
done