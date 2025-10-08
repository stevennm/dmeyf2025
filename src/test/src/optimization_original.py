# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
    'objective': 'binary',
    'metric': 'None',
    
    # Core tree parameters
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 16, 255),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 200),
    
    # Sampling parameters
    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
    
    # Regularization
    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 5),
    
    # Bin parameters
    'max_bin': trial.suggest_int('max_bin', 63, 255),
    
    # Tree depth (alternative to num_leaves)
    # 'max_depth': trial.suggest_int('max_depth', 3, 12),  # Use EITHER num_leaves OR max_depth, not both
    
    # Additional sampling
    'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.5, 1.0),
    
    # For imbalanced data (you have 0.7% positive class)
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
    
    'random_state': SEMILLA,
    'verbose': -1,
    }
  

    # Debug / sanity checks
    logger.info(f"Trial {getattr(trial,'number', 'NA')} - iniciando objetivo")
    logger.debug(f"Parametros sugeridos (preview): {trial.params if hasattr(trial,'params') else 'NA'}")
    logger.debug(f"Valores únicos en foto_mes (preview): {df['foto_mes'].unique()[:20]}")
    logger.debug(f"MES_TRAIN={MES_TRAIN} ({type(MES_TRAIN)}), MES_VALIDACION={MES_VALIDACION} ({type(MES_VALIDACION)})")


    # Completar!!!!!!
    # Preparar dataset para entrenamiento y validación
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)].copy()
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN].copy()

    df_valid = df[df['foto_mes'] == MES_VALIDACION]

    X_train = df_train.drop(['target', 'clase_peso', 'foto_mes'], axis=1)
    y_train = df_train['target']
    X_valid = df_valid.drop(['target', 'clase_peso', 'foto_mes'], axis=1)
    y_val = df_valid['target']

    # Entrenar modelo LightGBM
    peso_train = df_train['clase_peso']
    peso_valid = df_valid['clase_peso']

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=peso_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_val, weight=peso_valid, reference=lgb_train)


    try:
        modelo = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_valid],
            feval=ganancia_lgb_binary,
            callbacks=[lgb.log_evaluation(period=0)]
        )
    except Exception as e:
        logger.exception(f"Error entrenando modelo en trial {getattr(trial,'number','NA')}: {e}")
        return float("-inf")

    # Predecir y calcular ganancia
    y_pred = modelo.predict(X_valid)
    y_pred_binary = (y_pred > 0.025).astype(int)




    ganancia_total = calcular_ganancia(y_val, y_pred_binary, df_valid['clase_peso'])

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total








def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")









def optimizar(df, n_trials) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME
    
    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Completar!!!!!!!!
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name
    )

    study.optimize(lambda trial: objetivo_ganancia(trial, df), n_trials=n_trials)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study






# src/optimization.py (implementación simplificada)

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]
  
    # Entrenar modelo con mejores parámetros
    X_train = df_train_completo.drop(['target', 'clase_peso', 'foto_mes'], axis=1)
    y_train = df_train_completo['target']
    X_test = df_test.drop(['target', 'clase_peso', 'foto_mes'], axis=1)
    y_test = df_test['target']

    peso_train = df_train_completo['clase_peso']
    peso_test = df_test['clase_peso']

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=peso_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, weight=peso_test, reference=lgb_train)


    modelo = lgb.train(
        mejores_params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_test],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # Predecir en test
    y_pred = modelo.predict(X_test)
    y_pred_binary = (y_pred > 0.025).astype(int)

    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary, peso_test)

  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }
  
    return resultados

def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo para resultados de test
    archivo = f"resultados/{archivo_base}_test_results.json"

    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar nueva evaluación
    resultados_test['datetime'] = datetime.now().isoformat()
    resultados_test['configuracion'] = {
        'semilla': SEMILLA,
        'mes_train': MES_TRAIN,
        'mes_validacion': MES_VALIDACION,
        'mes_test': MES_TEST
    }
    datos_existentes.append(resultados_test)

    # Guardar todas las evaluaciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Resultados de test guardados en {archivo}")