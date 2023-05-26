import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.logging
import os

def get_model(args, dm):
    model = None
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0]+dm.adj_1.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "BGCN":
        model = models.BGCN(adj=dm.adj, adj_1=dm.adj_1, adj_2=dm.adj_2, adj_3=dm.adj_3, adj_4=dm.adj_4, adj_5=dm.adj_5, feat=dm.feat, team_2_player=dm.player_2_team, aspect_num=args.aspect_num, hidden_dim=args.hidden_dim, co_attention_dim=args.co_attention_dim, applying_team=args.applying_team, applying_player=args.applying_player, applying_co_attention=args.applying_co_attention, applying_attention=args.applying_attention)
    if args.model_name == "T2TGRU":
        model = models.T2TGRU(hidden_dim=args.hidden_dim)
    return model

def get_attentionLayer(args):
    if args.output_attention == "self":
        return models.SelfAttentionLayer(hidden_dim=args.hidden_dim)
    elif args.output_attention == "V1":
        return models.OutputAttentionV1Layer(hidden_dim=args.hidden_dim)
    elif args.output_attention == "V2":
        return models.OutputAttentionV2Layer(hidden_dim=args.hidden_dim, attention_dim=((args.hidden_dim)*3), attention_mul=False)
    elif args.output_attention == "V2_reverse":
        return models.OutputAttentionV2Layer(hidden_dim=args.hidden_dim, attention_dim=((args.hidden_dim)*7), attention_mul=False)
    elif args.output_attention == "co":
        return models.OutputCoAttentionLayer(hidden_size=7, history_hidden_size=12, hidden_dim=args.hidden_dim)
    else:
        return None

def get_task(args, model, attentionLayer, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, attentionLayer=attentionLayer, t_dim=dm.adj.shape[0] , p_dim=dm.adj_1.shape[0], **vars(args)
    )
    return task

def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    if args.model_name == "T2TGRU":
        feat_path = os.path.join('data', 'T2T', 'x_feat_train.p')
        y_path = os.path.join('data', 'T2T', 'y_feat_odds_train.p')
        T2T_flag = True
    else:
        feat_path = os.path.join('data', '21_22', 'new_team_list_other_n.p')
        y_path = os.path.join('data', '21_22', 'team_list_y_namenum_non_ave_odds.p') if args.output_attention in ["V2_reverse", "co", "encoder_all"] else os.path.join('data', '21_22', 'team_list_y_namenum_ave_odds.p')
        T2T_flag=False

    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=feat_path, 
        p_feat_path=os.path.join('data', '21_22', 'new_player_list_other_n.p'),
        player_team_path=os.path.join('data', '21_22', 'member_schedule.p'),
        y_path=y_path,
        T2T=T2T_flag,
        feat_path_test = os.path.join('data', 'T2T', 'x_feat.p'),
        y_path_test = os.path.join('data', 'T2T', 'y_feat_odds.p'),
        adj_path=os.path.join('data', '21_22', 'team_adj.csv'), 
        adj_1_path=os.path.join('data', '21_22', 'pass_adj.csv'),
        adj_2_path=os.path.join('data', '21_22', 'ast_adj.csv'),
        adj_3_path=os.path.join('data', '21_22', 'def_adj.csv'),
        adj_4_path=os.path.join('data', '21_22', 'blk_adj.csv'),
        adj_5_path=os.path.join('data', '21_22', 'pf_adj.csv'),
        **vars(args)
    )
    model = get_model(args, dm)
    attentionLayer = get_attentionLayer(args)
    task = get_task(args, model, attentionLayer, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for nba score difference prediction",
        choices=("BGCN", "GRU", "T2TGRU"),
        default="BGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:
        traceback.print_exc()
        exit(-1)
