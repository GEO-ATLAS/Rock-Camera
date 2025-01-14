#示例: 16 帧 随机间距--模式2-均值1.5，正负0.25，均匀分布 rand_flag=2

## predrnn_v3
python -u my_runner2.py --is_training 1 --device cuda:2 --dataset_name PBAll  --train_data_ratio 0.90 --save_dir ./checkpoints/c2srand2/16/ --gen_frm_dir ./results/c2srand2/16/ --model_name predrnn_v3 --reverse_input 0  --img_width 128 --img_channel 1 --px 1 --img_norm 0 --rand_or_fix 1 --rand_flag 2 --data_type capture2slice --input_length 8 --total_length 16 --num_hidden 128,128,128,128   --filter_size 5 --stride 1 --patch_size 4 --layer_norm 0 --scheduled_sampling 1  --sampling_stop_iter 80000 --sampling_start_value 1.0 --sampling_changing_rate 0.00002   --lr 0.0003 --batch_size 4 --max_iterations 400000 --start_ite 0 --display_interval 1000 --test_interval 4000 --snapshot_interval 2000 

## predrnn
python -u my_runner2.py --is_training 1 --device cuda:1 --dataset_name PBAll  --train_data_ratio 0.90 --save_dir ./checkpoints/rand2/16/ --gen_frm_dir ./results/c2srand2/16/ --model_name predrnn --reverse_input 0  --img_width 128 --img_channel 1 --px 1 --img_norm 0 --rand_or_fix 1 --rand_flag 2 --data_type capture2slice --input_length 8 --total_length 16 --num_hidden 128,128,128,128   --filter_size 5 --stride 1 --patch_size 4 --layer_norm 0 --scheduled_sampling 1  --sampling_stop_iter 80000 --sampling_start_value 1.0 --sampling_changing_rate 0.00002   --lr 0.0003 --batch_size 4 --max_iterations 200000 --start_ite 0 --display_interval 1000 --test_interval 4000 --snapshot_interval 2000 

## convlstm

python -u my_runner2.py --is_training 1 --device cuda:0 --dataset_name PBAll  --train_data_ratio 0.90 --save_dir ./checkpoints/rand2/16/ --gen_frm_dir ./results/c2srand2/16/ --model_name convlstm --reverse_input 0  --img_width 128 --img_channel 1 --px 1 --img_norm 0 --rand_or_fix 1 --rand_flag 2 --data_type capture2slice --input_length 8 --total_length 16 --num_hidden 128,128,128,128   --filter_size 5 --stride 1 --patch_size 4 --layer_norm 0 --scheduled_sampling 1  --sampling_stop_iter 80000 --sampling_start_value 1.0 --sampling_changing_rate 0.00002   --lr 0.0003 --batch_size 4 --max_iterations 200000 --start_ite 0 --display_interval 1000 --test_interval 4000 --snapshot_interval 2000 

