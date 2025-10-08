# src/config.py
import yaml
import os
import logging

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, encoding="utf-8") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]

        STUDY_NAME = _cfgGeneral.get("STUDY_NAME")
        DATA_PATH = _cfg.get("DATA_PATH")
        SEMILLA = _cfg.get("SEMILLA")
        MES_TRAIN = _cfg.get("MES_TRAIN")
        MES_VALIDACION = _cfg.get("MES_VALIDACION")
        MES_TEST = _cfg.get("MES_TEST")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO")
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO")
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN")
        FINAL_PREDIC = _cfg.get("FINAL_PREDIC")
        KFOLDS = _cfg.get("KFOLDS")
        EARLY_STOPPING_ROUNDS = _cfg.get("EARLY_STOPPING_ROUNDS")
        NUM_BOOST_ROUND = _cfg.get("NUM_BOOST_ROUND")
        UMBRAL = _cfg.get("UMBRAL")
        logger.info("Archivo de configuracion cargado correctamente.")

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise