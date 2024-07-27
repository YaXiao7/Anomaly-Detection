# python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 48 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --CSP --dilated --passthrough --itr 10

python -u main_informer.py --model informer --data custom --root_path data/exp --features MS \
    --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --train_epochs 1000 --patience 5 \
    --CSP --dilated --passthrough --itr 10 --freq t --pred_len 1 --do_predict \
    --dataname $1 --clientname $2
