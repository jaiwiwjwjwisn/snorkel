config_updates = {
    "n_epochs": 5,
    "optimizer_config": {
        "lr": 0.001,
    }
}
trainer_config = merge_config(TrainerConfig(), config_updates)
