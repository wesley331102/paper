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

```bash
====================================================================================================
# BGCN no player
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_team --loss nba_mae --settings supervised 
====================================================================================================
```

```bash
====================================================================================================
# BGCN no team
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --loss nba_mae --settings supervised 
====================================================================================================
```

```bash
====================================================================================================
# BGCN no co-attention
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_team --applying_player --output_attention encoder --loss nba_mae --settings supervised 
====================================================================================================
```

```bash
====================================================================================================
# BGCN no encoder
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_team --applying_player --applying_co_attention --loss nba_mae --settings supervised 
====================================================================================================
```

```bash
# BGCN self attention
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --output_attention self --loss nba_mae --settings supervised 
```

```bash
# BGCN attention V1
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --output_attention V1 --loss nba_mae --settings supervised 
```

```bash
# BGCN attention V2 (add diff and mul)
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --output_attention V2 --loss nba_rmse --settings supervised 
```

```bash
# BGCN attention V2_reverse
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --output_attention V2_reverse --loss nba_mae --settings supervised 
```

```bash
# BGCN co-attention
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_player --output_attention co --loss nba_mae --settings supervised 
```

```bash
====================================================================================================
# BGCN encoder
python main.py --model_name BGCN --max_epochs 100 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --aspect_num 4 --hidden_dim 64 --co_attention_dim 16 --applying_team --applying_player --applying_co_attention --output_attention encoder --loss nba_mae --settings supervised 
====================================================================================================
```
