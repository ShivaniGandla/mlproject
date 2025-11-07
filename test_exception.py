import sys
from src.exception import CustomException
from src.logger.logger_config import logger

try:
    x = 1 / 0
except Exception as e:
    logger.error("An error occurred:")
    raise CustomException(e, sys)