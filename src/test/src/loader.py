# src/loader.py
import pandas as pd
import logging

logger = logging.getLogger("__name__")

## Funcion para cargar datos
def cargar_datos(path: str) -> pd.DataFrame | None:

    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise



def convertir_clase_ternaria_con_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a binario PERO mantiene información mediante pesos.
    
    Binary target:
    - CONTINUA = 0
    - BAJA+1 = 1
    - BAJA+2 = 1
    
    Weights (para diferenciar en la función de ganancia):
    - CONTINUA = 1.0
    - BAJA+1 = 1.00001
    - BAJA+2 = 1.00002
    
    Args:
        df: DataFrame con columna 'clase_ternaria'
    
    Returns:
        pd.DataFrame: DataFrame con target binario y columna de pesos
    """
    df_result = df.copy()
    
    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()
    
    # Crear target binario
    df_result['target'] = df_result['clase_ternaria'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })
    
    # Crear pesos para diferenciar las clases
    df_result['clase_peso'] = df_result['clase_ternaria'].map({
        'CONTINUA': 1.0,
        'BAJA+1': 1.00001,
        'BAJA+2': 1.00002
    })
    
    # Eliminar clase_ternaria original (ya no se necesita)
    df_result = df_result.drop('clase_ternaria', axis=1)
    
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Target - 0: {(df_result['target'] == 0).sum()}, 1: {(df_result['target'] == 1).sum()}")
    logger.info(f"  Pesos - 1.0: {(df_result['clase_peso'] == 1.0).sum()}, "
                f"1.00001: {(df_result['clase_peso'] == 1.00001).sum()}, "
                f"1.00002: {(df_result['clase_peso'] == 1.00002).sum()}")
    
    return df_result