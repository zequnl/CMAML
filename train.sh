python CMAML.py --cuda --batch_size 16 --use_sgd --lr 0.01 --meta_lr 0.0003 --meta_batch_size 16 --meta_optimizer adam --pretrain_emb --weight_sharing --emb_dim 300 --hidden_dim 300 --save_path save/cmaml/