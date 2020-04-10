import click
import click_pathlib
import logging
from hotelbooking.models import models_utils
from hotelbooking.models import model_utils_GS

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
    pass


@main.command()
@click.option("--data-path", type=click_pathlib.Path(exists=True))
@click.option("--model-version", type=int)
def train_model(data_path, model_version):
    models_utils.run(data_path, model_version)
    logger.info('Finished with training the model.')


@main.command()
@click.option("--data-path", type=click_pathlib.Path(exists=True))
@click.option("--model-version", type=int)
def optimise_model(data_path, model_version):
    model_utils_GS.run(data_path, model_version)
    logger.info('Finished with optimising the model.')



