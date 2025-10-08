import os
import joblib
from src.config import *
import logging

logger = logging.getLogger(__name__)
dump_path = "models/trained_model.sav"

def guardar_modelo(model):
    """
    Guarda el modelo entrenado en el path especificado usando joblib.
  
    Args:
        model: Modelo entrenado (por ejemplo, un objeto LightGBM)
    """
    joblib.dump(model, dump_path)
    logger.info(f"Modelo guardado en: {dump_path}")