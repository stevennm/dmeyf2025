# src/final_training.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, peso_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    if isinstance(FINAL_TRAIN, list):
        df_train = df[df['foto_mes'].isin(FINAL_TRAIN)].copy()
    else:
        df_train = df[df['foto_mes'] == FINAL_TRAIN].copy()    
    # Datos de predicción: período FINAL_PREDIC 
    df_predict = df[df['foto_mes'] == FINAL_PREDIC].copy()

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df
    if df_train.empty:
        logger.error("DataFrame de entrenamiento final está vacío. Revise FINAL_TRAIN y la columna 'foto_mes'.")
        raise ValueError("df_train vacío en preparar_datos_entrenamiento_final")

    if df_predict.empty:
        logger.error("DataFrame de predicción final está vacío. Revise FINAL_PREDIC y la columna 'foto_mes'.")
        raise ValueError("df_predict vacío en preparar_datos_entrenamiento_final")
    # Preparar features y target para entrenamiento
  
    # Columnas a excluir de features
    exclude_cols = {'target', 'clase_peso', 'foto_mes', 'numero_de_cliente'}
    features_cols = [c for c in df.columns if c not in exclude_cols]

    # Preparar features y target para entrenamiento
    X_train = df_train[features_cols].reset_index(drop=True)
    y_train = df_train['target'].reset_index(drop=True)
    peso_train = df_train['clase_peso'].reset_index(drop=True)



    # Preparar features para predicción
    X_predict = df_predict[features_cols].reset_index(drop=True)
    if 'numero_de_cliente' in df_predict.columns:
        clientes_predict = df_predict['numero_de_cliente'].values
    else:
        clientes_predict = df_predict.index.values

    logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

    return X_train, y_train, peso_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, peso_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")

    # Cargar mejores params si no fueron proporcionados
    if mejores_params is None:
        try:
            mejores_params = cargar_mejores_hiperparametros() or {}
            logger.info("Mejores hiperparámetros cargados desde almacenamiento")
        except Exception as e:
            logger.warning(f"No se pudieron cargar mejores hiperparámetros, se usarán valores por defecto: {e}")
            mejores_params = {}

    # Parche para semilla
    random_state = SEMILLA[0] if isinstance(SEMILLA, (list, tuple)) else SEMILLA

    # Parámetros base y combinación con mejores_params
    params = {
        'objective': 'binary',
        'metric': 'None',
        'random_state': random_state,
        'verbose': -1,
        **(mejores_params or {})
    }

    # Número de iteraciones
    num_boost_round = int(params.get('num_boost_round', 1000))

    logger.info(f"Parámetros finales para entrenamiento: { {k: params[k] for k in params if k != 'random_state'} }")
    logger.info(f"Num boost rounds: {num_boost_round}")
  
    # Crear dataset de LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train, weight=peso_train)

    
    # Entrenar modelo con lgb.train()
    modelo = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
        feval=ganancia_lgb_binary,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    logger.info("Entrenamiento finalizado")
    return modelo

def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.025) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.
  
    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
  
    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")

    if X_predict is None or X_predict.shape[0] == 0:
        logger.error("X_predict está vacío. No se pueden generar predicciones.")
        raise ValueError("X_predict vacío en generar_predicciones_finales")

    # Asegurar que clientes_predict tenga la misma longitud que X_predict
    clientes_predict = np.asarray(clientes_predict)
    if clientes_predict.shape[0] != X_predict.shape[0]:
        logger.error("La longitud de clientes_predict no coincide con X_predict.")
        raise ValueError("Mismatch entre clientes_predict y X_predict en generar_predicciones_finales")

    # Generar probabilidades con el modelo entrenado
    probs = modelo.predict(X_predict)

    # Convertir a predicciones binarias con el umbral establecido
    y_pred_binary = (probs > umbral).astype(int)

    # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'predict': y_pred_binary,
        'probability': probs
    })
    
    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
  
    return resultados