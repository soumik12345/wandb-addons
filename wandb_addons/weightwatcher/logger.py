import pandas as pd
import wandb


def log_details(details: pd.DataFrame):
    details_table = wandb.Table(dataframe=details)
    wandb.log(details_table)
