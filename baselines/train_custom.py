from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from custom_dssm import CustomDSSM
from recbole.config import Config
from recbole.data import create_dataset, data_preparation


if __name__ == "__main__":

    config = Config(
        model=CustomDSSM, config_file_list=["configs/dssm_custom_plain.yaml"]
    )
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = CustomDSSM(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, show_progress=True)

    logger.info("best valid result: {}".format(best_valid_result))
    logger.info("test result: {}".format(test_result))
