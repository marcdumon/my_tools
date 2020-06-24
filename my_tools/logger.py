# --------------------------------------------------------------------------------------------------------
# 2020/06/12
# Twitter_Scraping - logger.py
# md
# --------------------------------------------------------------------------------------------------------
import logging
import coloredlogs

# from config import LOGGING_LEVEL
LOGGING_LEVEL = 'Debug'
if LOGGING_LEVEL == 'Debug':
    level = logging.DEBUG
elif LOGGING_LEVEL == 'Info':
    level = logging.INFO
elif LOGGING_LEVEL=='Warning':
    level = logging.WARNING
elif LOGGING_LEVEL=='Error':
    level = logging.ERROR
else:
    level = logging.NOTSET

# create logger
logger = logging.getLogger('scraper')
logger.setLevel(level)

logger.propagate = False


# create console handler and set level to debug
handler = logging.StreamHandler()
handler.setLevel(level)

# create formatter
format = '[%(filename)s:%(lineno)s %(funcName)20s]: %(asctime)s [%(levelname)8s]: %(message)s'
formatter = coloredlogs.ColoredFormatter(format)

# add formatter to console handler
handler.setFormatter(formatter)

# add console handler to logger
logger.addHandler(handler)
