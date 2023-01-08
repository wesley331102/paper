# B-GCN-PyTorch

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training

You can also adjust the `--seq_len` parameters.

```bash
# Basketball
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --linear_transformation --applying_player --applying_attention --loss nba_rmse --settings supervised
```