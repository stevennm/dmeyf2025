# src/features.py
import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def _ensure_columns(df: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    """
    Asegura que existan todas las columnas requeridas. Si falta alguna,
    la crea como NA (nullable), para que el SQL no falle.
    """
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def feature_engineering_financiero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica en Python/DuckDB los macros y ratios de tu script SQL (pct_util, shares, flags, etc.)
    Devuelve df con t.* + columnas nuevas.
    """
    # columnas referenciadas en los cálculos (para no romper si falta alguna)
    needed_cols = [
        "numero_de_cliente","foto_mes",
        "Visa_msaldototal","Master_msaldototal",
        "Visa_mlimitecompra","Master_mlimitecompra",
        "Visa_mpagominimo","Visa_msaldopesos",
        "Master_mpagominimo","Master_msaldopesos",
        "mtarjeta_visa_consumo","mtarjeta_master_consumo",
        "mcuentas_saldo","Visa_mconsumospesos","Master_mconsumospesos",
        "ctarjeta_visa_transacciones","ctarjeta_master_transacciones",
        "ctarjeta_visa_debitos_automaticos","ctarjeta_master_debitos_automaticos",
        "mcaja_ahorro","mpayroll","ctrx_quarter",
        "chomebanking_transacciones","cmobile_app_trx","ccajas_transacciones",
        "cpagodeservicios","cpagomiscuentas","ccuenta_debitos_automaticos",
        "cliente_antiguedad","mtransferencias_emitidas","mtransferencias_recibidas",
        "ctransferencias_emitidas","ctransferencias_recibidas",
        "mpasivos_margen","mactivos_margen","mcuenta_corriente",
        "mprestamos_personales","mprestamos_hipotecarios"
    ]
    df = _ensure_columns(df, needed_cols)

    con = duckdb.connect(database=":memory:")
    try:
        con.register("t", df)

        # ---------- Macros (seguros/robustos) ----------
        con.execute("""
        CREATE OR REPLACE MACRO ratio_seguro(n, d) AS (
            CASE WHEN d IS NULL OR d = 0 THEN NULL
                 ELSE CAST(n AS DOUBLE) / CAST(d AS DOUBLE) END
        );

        CREATE OR REPLACE MACRO ratio_pct_seguro(n, d) AS (
            CASE WHEN d IS NULL OR d = 0 THEN NULL
                 ELSE 100.0 * CAST(n AS DOUBLE) / CAST(d AS DOUBLE) END
        );

        CREATE OR REPLACE MACRO div_segura(n, d) AS (
            CASE WHEN d IS NULL OR d = 0 THEN NULL
                 ELSE CAST(n AS DOUBLE) / CAST(d AS DOUBLE) END
        );

        -- Nota: en tu comentario decía log1p, pero el código usaba log10(...).
        -- Mantengo log10 para ser fiel al script original.
        CREATE OR REPLACE MACRO ratio_log1p_seguro(n, d) AS (
            CASE
                WHEN n IS NULL OR d IS NULL OR d = 0 THEN NULL
                WHEN div_segura(n, d) IS NULL THEN NULL
                WHEN div_segura(n, d) <= (-1.0 + 1e-12) THEN NULL
                ELSE log10(div_segura(n, d))
            END
        );

        CREATE OR REPLACE MACRO pct_change_seguro(curr, prev) AS (
            CASE WHEN prev IS NULL OR prev = 0 THEN NULL
                 ELSE 100.0 * (CAST(curr AS DOUBLE) - CAST(prev AS DOUBLE)) / CAST(prev AS DOUBLE) END
        );
        """)

        # ---------- SELECT con derivados (fiel al SQL que pegaste) ----------
        sql = """
        SELECT
          t.*,

          -- USO DE CRÉDITO
          ratio_pct_seguro(
            COALESCE(Visa_msaldototal,0) + COALESCE(Master_msaldototal,0),
            NULLIF(COALESCE(Visa_mlimitecompra,0) + COALESCE(Master_mlimitecompra,0),0)
          ) AS pct_util_limite_tc,

          ratio_pct_seguro(Visa_msaldototal,  NULLIF(Visa_mlimitecompra,0))   AS pct_util_limite_visa,
          ratio_pct_seguro(Master_msaldototal,NULLIF(Master_mlimitecompra,0)) AS pct_util_limite_master,

          ratio_pct_seguro(Visa_mpagominimo,  NULLIF(Visa_msaldopesos,0))     AS pct_pago_min_vs_saldo_visa,
          ratio_pct_seguro(Master_mpagominimo,NULLIF(Master_msaldopesos,0))   AS pct_pago_min_vs_saldo_master,

          ratio_seguro(
            COALESCE(mtarjeta_visa_consumo,0) + COALESCE(mtarjeta_master_consumo,0),
            mcuentas_saldo
          ) AS gasto_tc_vs_saldo,

          ratio_seguro(Visa_mconsumospesos,  NULLIF(ctarjeta_visa_transacciones,0))    AS ticket_medio_visa,
          ratio_seguro(Master_mconsumospesos,NULLIF(ctarjeta_master_transacciones,0))  AS ticket_medio_master,

          ratio_seguro(
            COALESCE(ctarjeta_visa_debitos_automaticos,0) + COALESCE(ctarjeta_master_debitos_automaticos,0),
            NULLIF(COALESCE(ctarjeta_visa_transacciones,0) + COALESCE(ctarjeta_master_transacciones,0),0)
          ) AS share_deb_auto_en_tc,

          -- LIQUIDEZ / SUELDO
          ratio_seguro(mcaja_ahorro,  mpayroll)        AS caja_vs_sueldo,
          ratio_seguro(mcuentas_saldo, mpayroll)       AS saldo_vs_sueldo,
          ratio_seguro(mcaja_ahorro,  mcuentas_saldo)  AS caja_sobre_saldo,

          -- ACTIVIDAD TRANSACCIONAL
          ratio_seguro(ctarjeta_debito_transacciones,                         ctrx_quarter) AS share_trx_debito,
          ratio_seguro(COALESCE(ctarjeta_visa_transacciones,0)+COALESCE(ctarjeta_master_transacciones,0), ctrx_quarter) AS share_trx_credito,
          ratio_seguro(COALESCE(chomebanking_transacciones,0)+COALESCE(cmobile_app_trx,0), ctrx_quarter)  AS share_trx_digital,
          ratio_seguro(ccajas_transacciones,                                  ctrx_quarter) AS share_trx_cajas,
          ratio_seguro(cpagodeservicios,                                      ctrx_quarter) AS share_trx_pago_servicios,
          ratio_seguro(cpagomiscuentas,                                       ctrx_quarter) AS share_trx_pagomiscuentas,
          ratio_seguro(ccuenta_debitos_automaticos,                           ctrx_quarter) AS share_trx_deb_auto_cc,

          ratio_seguro(mcuentas_saldo, ctrx_quarter) AS saldo_por_trx,
          ratio_seguro(ctrx_quarter,  NULLIF(cliente_antiguedad,0)) AS trx_por_mes_de_antig,

          -- TRANSFERENCIAS
          ratio_seguro(mtransferencias_emitidas,   mtransferencias_recibidas) AS ratio_out_in_monto,
          ratio_seguro(ctransferencias_emitidas,   ctrx_quarter)              AS share_trx_out,
          ratio_seguro(ctransferencias_recibidas,  ctrx_quarter)              AS share_trx_in,
          ratio_seguro(mtransferencias_recibidas,  mpayroll)                  AS inflow_transf_vs_sueldo,

          ratio_seguro(
            COALESCE(mtransferencias_recibidas,0) - COALESCE(mtransferencias_emitidas,0),
            NULLIF(mcuentas_saldo,0)
          ) AS neto_transf_sobre_saldo,

          -- ESTRUCTURA FINANCIERA
          ratio_seguro(mpasivos_margen,   mactivos_margen)  AS pasivos_sobre_activos_margen,
          ratio_seguro(mcuenta_corriente, mcuentas_saldo)   AS ccorriente_vs_saldo,

          -- PRÉSTAMOS / DEUDA
          ratio_seguro(mprestamos_personales, mpayroll)       AS cuota_personal_vs_sueldo_aprox,
          ratio_seguro(mprestamos_personales, mcuentas_saldo) AS personales_vs_saldo,
          ratio_seguro(mprestamos_hipotecarios, mpayroll)     AS cuota_hipo_vs_sueldo_aprox,

          -- CANALES / ADOPCIÓN (flags 0/1)
          CASE WHEN COALESCE(chomebanking_transacciones,0) > 0 THEN 1 ELSE 0 END AS flag_usa_hb,
          CASE WHEN COALESCE(cmobile_app_trx,0)            > 0 THEN 1 ELSE 0 END AS flag_usa_app,
          CASE WHEN COALESCE(ccajas_transacciones,0)       > 0 THEN 1 ELSE 0 END AS flag_usa_caja,
          CASE WHEN COALESCE(ctarjeta_debito_transacciones,0) > 0 THEN 1 ELSE 0 END AS flag_usa_debito,
          CASE WHEN COALESCE(ctarjeta_visa_transacciones,0)+COALESCE(ctarjeta_master_transacciones,0) > 0 THEN 1 ELSE 0 END AS flag_usa_tc,

          (CASE WHEN COALESCE(chomebanking_transacciones,0) > 0 THEN 1 ELSE 0 END
           + CASE WHEN COALESCE(cmobile_app_trx,0) > 0 THEN 1 ELSE 0 END
           + CASE WHEN COALESCE(ccajas_transacciones,0) > 0 THEN 1 ELSE 0 END
           + CASE WHEN COALESCE(ctarjeta_debito_transacciones,0) > 0 THEN 1 ELSE 0 END
           + CASE WHEN COALESCE(ctarjeta_visa_transacciones,0)+COALESCE(ctarjeta_master_transacciones,0) > 0 THEN 1 ELSE 0 END
          ) AS canales_usados_cnt

        FROM t AS t
        """
        out = con.execute(sql).df()
        logger.info(f"Feature financiero completado. DataFrame resultante con {out.shape[1]} columnas")
        return out
    finally:
        con.close()


def feature_engineering_lag(
    df: pd.DataFrame, columnas: list[str], cant_lag: int = 1
) -> pd.DataFrame:
    """
    Tu misma función (idéntica), solo le agrego un pequeño guard contra columnas inexistentes.
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Filtrar solo columnas que existan
    columnas_validas = [c for c in columnas if c in df.columns]
    faltantes = set(columnas) - set(columnas_validas)
    for c in faltantes:
        logger.warning(f"El atributo {c} no existe en el DataFrame")

    if not columnas_validas:
        return df

    sql = "SELECT *"
    for attr in columnas_validas:
        for i in range(1, cant_lag + 1):
            sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {out.shape[1]} columnas")
    return out
