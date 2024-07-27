python -u server.py --model informer --data custom --root_path data/exp --features MS \
    --enc_in 6 --dec_in 6 --c_out 2 \
    --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --train_epochs 10 --patience 5 \
    --CSP --dilated --passthrough --itr 10 --freq t --pred_len 2 --do_predict