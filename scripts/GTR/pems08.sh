model_name=GTR

root_path_name=./dataset/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS


seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --cycle 288 \
      --train_epochs 30 \
      --use_revin 1 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done
