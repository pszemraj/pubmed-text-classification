import os

from knockknock import telegram_sender

def get_knockknock_notifier(
    trainer,
    datamodule,
    model,
    train_strategy: str = "freeze",
    UNFREEZE_EPOCH: int = 1,
    api_key: str = "KNOCK_TELEGRAM_API",
    chat_id: str = "KNOCK_TELEGRAM_CHAT",
):
    """
    get_knockknock_notifier - returns a knockknock.telegram_sender.TelegramSender object for use with PyTorch Lightning-Flash trainer.
                                optional dependencies for this function are: knockknock (pip install knockknock)

    Args:
        trainer (PyTorch Lightning LightningFlashTrainer): required, trainer object
        datamodule (PyTorch Lightning LightningDataModule): required, datamodule object
        model (PyTorch Lightning LightningModule): required, model object
        train_strategy (str, optional): train strategy, either "freeze", "no_freeze", "freeze_unfreeze", or "unfreeze_freeze"
        UNFREEZE_EPOCH (int, optional): number of epochs to wait before unfreezing the model
        api_key (str, optional): telegram api key in your PATH / env, default "KNOCK_TELEGRAM_API"
        chat_id (str, optional): telegram chat id in your PATH / env, default "KNOCK_TELEGRAM_CHAT"

    Returns:
        knockknock.telegram_sender.TelegramSender: object for sending telegram messages
    """
    BOT_API: str = os.environ.get(api_key)
    CHAT_ID: int = os.environ.get(chat_id)

    @telegram_sender(token=BOT_API, chat_id=CHAT_ID)
    def knockknock_test_wrap(verbose=False):

        if train_strategy == "full_train":
            trainer.fit(
                model,
                datamodule=datamodule,
            )
        else:
            trainer.finetune(
                model,
                datamodule=datamodule,
                strategy=("freeze_unfreeze", UNFREEZE_EPOCH)
                if train_strategy == "freeze_unfreeze"
                else train_strategy,
                # 'freeze_unfreeze' is a special case
            )

        eval_metrics = trainer.test(
            verbose=verbose,
            datamodule=datamodule,
        )
        return eval_metrics

    return knockknock_test_wrap
