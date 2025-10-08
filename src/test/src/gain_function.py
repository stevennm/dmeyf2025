# src/gain_function.py
import numpy as np
import pandas as pd
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred, clase_peso=None):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.
    Usa clase_peso para identificar BAJA+2 (1.00002) y BAJA+1 (1.00001).
 
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)
        clase_peso: Pesos de clase (1.0, 1.00001, 1.00002) - opcional
  
    Returns:
        float: Ganancia total
    """
    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if clase_peso is not None and isinstance(clase_peso, pd.Series):
        clase_peso = clase_peso.values
  
    if clase_peso is not None:
        # Exact replication of professor's approach
        ganancia_individual = (
            np.where(clase_peso == 1.00002, GANANCIA_ACIERTO, 0) -
            np.where(clase_peso < 1.00002, COSTO_ESTIMULO, 0)
        )
        ganancia_total = np.sum(ganancia_individual * y_pred)
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        ganancia_total = tp * GANANCIA_ACIERTO - fp * COSTO_ESTIMULO
    
    return ganancia_total

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
    Ahora extrae y usa clase_peso si está disponible.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()
  
    # Intentar obtener clase_peso si está disponible
    try:
        clase_peso = y_true.get_weight()
        if clase_peso is None or len(clase_peso) == 0:
            clase_peso = None
    except:
        clase_peso = None
  
    # Convertir probabilidades a predicciones binarias (umbral 0.025)
    y_pred_binary = (y_pred > 0.025).astype(int)
  
    # Calcular ganancia usando configuración y pesos
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary, clase_peso)
  
    # Retornar en formato esperado por LightGBM
    return 'ganancia', ganancia_total, True  # True = higher is better