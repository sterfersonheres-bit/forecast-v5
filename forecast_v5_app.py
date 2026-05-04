#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║           FORECAST INTELIGENTE V5 — Dashboard Streamlit      ║
║    IA + Estatística | Backtesting | Análise por SKU          ║
╚══════════════════════════════════════════════════════════════╝

INSTALAÇÃO:
    pip install streamlit pandas numpy plotly scipy scikit-learn
    pip install statsmodels openpyxl xlsxwriter

EXECUÇÃO:
    streamlit run forecast_v5_app.py

ESTRUTURA ESPERADA DO EXCEL (abas):
    Base_Limpa       → Histórico de demanda (SKU | Período | Demanda)
    Estatistica_SKU  → Estatísticas por SKU
    Previsao_Modelo  → Previsões dos modelos estatísticos
    Avaliacao_Modelo → Melhores previsões com WMAPE
    Avaliacao_SKU    → Avaliação por SKU
"""

import os, glob, warnings, io, datetime, unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    st.sidebar.warning("⚠️ statsmodels não instalado. Alguns métodos serão desabilitados.\npip install statsmodels")

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Forecast Inteligente V5",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        color: white;
    }
    .badge-green { background:#00b894; color:white; padding:2px 8px; border-radius:12px; font-size:12px; }
    .badge-red   { background:#d63031; color:white; padding:2px 8px; border-radius:12px; font-size:12px; }
    .badge-yellow{ background:#fdcb6e; color:#2d3436; padding:2px 8px; border-radius:12px; font-size:12px; }
    .stTabs [data-baseweb="tab"] { font-size:15px; font-weight:600; }
    div[data-testid="metric-container"] { background:#f8f9fa; border-radius:10px; padding:10px; border:1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

PASTA_DEFAULT = r"C:\Users\sborges\OneDrive - Energisa\Área de Trabalho\Pensar no futuro\GPT"

# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════

def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto.replace(" ", "").replace("_", "").replace("-", "")

def encontrar_coluna(df, *palavras):
    """Busca coluna por palavras-chave normalizadas"""
    cols_norm = {normalizar(c): c for c in df.columns}
    for palavra in palavras:
        pn = normalizar(palavra)
        for norm, orig in cols_norm.items():
            if pn in norm:
                return orig
    return None

def wmape(actual, forecast):
    """WMAPE = Σ|real - prev| / Σ|real|"""
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return np.nan
    return np.sum(np.abs(a[mask] - f[mask])) / np.sum(a[mask])

def cor_wmape(val):
    """Retorna cor HTML baseada no WMAPE"""
    if pd.isna(val):
        return "#95a5a6"
    if val < 0.20:
        return "#00b894"
    if val < 0.40:
        return "#f9ca24"
    if val < 0.60:
        return "#e17055"
    return "#d63031"

# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════

def carregar_excel(arquivo) -> dict:
    try:
        # Aceita tanto caminho de arquivo quanto objeto de upload do Streamlit
        xls = pd.ExcelFile(arquivo)
        abas = xls.sheet_names

        def get_aba(*nomes):
            for nome in nomes:
                for aba in abas:
                    if normalizar(nome) in normalizar(aba):
                        df = pd.read_excel(arquivo, sheet_name=aba)
                        orig_cols = list(df.columns)
                        df.columns = [normalizar(c) for c in df.columns]
                        df.attrs['colunas_originais'] = orig_cols
                        return df
            return None

        return {
            'base':           get_aba('baselimpa', 'base', 'historico', 'hist'),
            'estatistica':    get_aba('estatisticasku', 'estatistica', 'estat'),
            'previsao':       get_aba('previsaomodelo', 'previsao', 'modelo'),
            'avaliacao':      get_aba('avaliacaomodelo', 'avaliacao', 'aval'),
            'avaliacao_sku':  get_aba('avaliacaosku', 'avalsku'),
            'abas':           abas,
            'erro':           None
        }
    except Exception as e:
        return {'erro': f'❌ Erro ao carregar: {str(e)}'}

# ══════════════════════════════════════════════════════════════
# MÉTODOS DE FORECAST
# ══════════════════════════════════════════════════════════════

def _safe_positive(vals):
    return [max(0.0, float(v)) if pd.notna(v) else 0.0 for v in vals]

def forecast_naive(serie, h=1):
    return _safe_positive([serie.iloc[-1]] * h)

def forecast_ma3(serie, h=1):
    return _safe_positive([serie.tail(3).mean()] * h)

def forecast_ma6(serie, h=1):
    return _safe_positive([serie.tail(min(6, len(serie))).mean()] * h)

def forecast_wma(serie, h=1, window=3):
    tail = serie.tail(window).values
    w = np.arange(1, len(tail)+1, dtype=float)
    return _safe_positive([np.average(tail, weights=w)] * h)

def forecast_ses(serie, h=1):
    if not HAS_STATSMODELS or len(serie) < 3:
        return forecast_ma3(serie, h)
    try:
        m = SimpleExpSmoothing(serie.astype(float).values).fit(optimized=True, disp=False)
        return _safe_positive(m.forecast(h).tolist())
    except:
        return forecast_ma3(serie, h)

def forecast_holt(serie, h=1):
    if not HAS_STATSMODELS or len(serie) < 5:
        return forecast_ses(serie, h)
    try:
        m = Holt(serie.astype(float).values).fit(optimized=True, disp=False)
        return _safe_positive(m.forecast(h).tolist())
    except:
        return forecast_ses(serie, h)

def forecast_hw(serie, h=1):
    sp = 12
    if not HAS_STATSMODELS or len(serie) < sp * 2:
        return forecast_holt(serie, h)
    try:
        m = ExponentialSmoothing(
            serie.astype(float).values,
            trend='add', seasonal='add', seasonal_periods=sp
        ).fit(optimized=True, disp=False)
        return _safe_positive(m.forecast(h).tolist())
    except:
        return forecast_holt(serie, h)

def forecast_croston(serie, h=1):
    """Croston para demanda intermitente"""
    d = serie.dropna().values.astype(float)
    if len(d) == 0 or d.max() == 0:
        return [0.0] * h
    alpha = 0.15
    non_zero_vals = d[d > 0]
    a = float(non_zero_vals[0]) if len(non_zero_vals) > 0 else 1.0
    p = 1.0
    last_nz = -1
    for i, val in enumerate(d):
        if val > 0:
            interval = i - last_nz if last_nz >= 0 else 1
            a = alpha * val + (1 - alpha) * a
            p = alpha * interval + (1 - alpha) * p
            last_nz = i
    result = max(0.0, a / max(p, 1e-6))
    return [result] * h

def forecast_trim_heres(serie, h=1):
    """TriM-Heres: média do mesmo trimestre (M-2,M-1,M) dos últimos 2 anos.
    Captura sazonalidade sem exigir série longa. Fallback ponderado se < 6 obs."""
    s = serie.dropna().reset_index(drop=True)
    n = len(s)
    resultados = []
    for step in range(1, h + 1):
        n_pred = n + step - 1  # índice da posição sendo prevista
        # Mesmo trimestre ano -1: posições n_pred-14, n_pred-13, n_pred-12
        # Mesmo trimestre ano -2: posições n_pred-26, n_pred-25, n_pred-24
        idx_y1 = [n_pred - 14, n_pred - 13, n_pred - 12]
        idx_y2 = [n_pred - 26, n_pred - 25, n_pred - 24]
        vals = []
        for i in idx_y1 + idx_y2:
            if 0 <= i < n:
                vals.append(float(s.iloc[i]))
        n_found = len(vals)
        if n_found == 6:
            pred = float(np.mean(vals))
        elif n_found > 0:
            trim_pred  = float(np.mean(vals))
            anual_pred = float(s.tail(min(12, n)).mean())
            peso_trim  = n_found / 6.0
            pred = peso_trim * trim_pred + (1 - peso_trim) * anual_pred
        else:
            pred = float(s.mean()) if n > 0 else 0.0
        resultados.append(max(0.0, pred))
    return resultados

METODOS = {
    'Naive':       forecast_naive,
    'MA-3':        forecast_ma3,
    'MA-6':        forecast_ma6,
    'WMA-3':       forecast_wma,
    'SES':         forecast_ses,
    'Holt':        forecast_holt,
    'Holt-Winters': forecast_hw,
    'Croston':     forecast_croston,
    'TriM-Heres':  forecast_trim_heres,
}

METODOS_DESC = {
    'Naive':        'Usa o último valor observado. Boa referência baseline.',
    'MA-3':         'Média simples dos 3 últimos períodos.',
    'MA-6':         'Média simples dos 6 últimos períodos. Suaviza mais.',
    'WMA-3':        'Média ponderada (maior peso nos períodos recentes).',
    'SES':          'Suavização Exponencial Simples. Alfa otimizado.',
    'Holt':         'Holt duplo. Captura tendência linear.',
    'Holt-Winters': 'Triple ES. Captura tendência e sazonalidade anual.',
    'Croston':      'Ideal para demanda intermitente com muitos zeros.',
    'TriM-Heres':   'Média do mesmo trimestre dos 2 últimos anos. Captura sazonalidade.',
}

# ══════════════════════════════════════════════════════════════
# BACKTESTING — WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════

def backtest_sku(serie: pd.Series, fn, n_test: int = 3):
    """Walk-forward validation para um SKU e um método"""
    s = serie.dropna().reset_index(drop=True)
    n = len(s)
    if n < n_test + 4:
        return np.nan, [], []

    actuals, preds = [], []
    for i in range(n_test, 0, -1):
        train = s.iloc[:n - i]
        try:
            pred = fn(train, h=1)[0]
        except:
            pred = float(train.mean())
        actuals.append(float(s.iloc[n - i]))
        preds.append(max(0.0, pred))

    return wmape(actuals, preds), actuals, preds


@st.cache_data(show_spinner=False)
def rodar_backtesting_cached(df_hash: str, n_test: int):
    """Versão cacheada — o hash é passado para invalidar cache se dados mudarem"""
    # Placeholder: real computation happens in rodar_backtesting()
    pass


def rodar_backtesting(df_base, col_sku, col_demanda, n_test=3):
    skus = df_base[col_sku].dropna().unique()
    resultados = []
    barra = st.progress(0, text="Rodando backtesting...")

    for idx, sku in enumerate(skus):
        serie = df_base[df_base[col_sku] == sku][col_demanda].reset_index(drop=True)
        row = {'sku': sku}
        melhor_wmape, melhor_nome = np.inf, 'MA-3'

        for nome, fn in METODOS.items():
            w, _, _ = backtest_sku(serie, fn, n_test=n_test)
            row[f'wmape_{nome}'] = w
            if pd.notna(w) and w < melhor_wmape:
                melhor_wmape, melhor_nome = w, nome

        row['melhor_metodo'] = melhor_nome
        row['melhor_wmape'] = melhor_wmape if melhor_wmape < np.inf else np.nan
        resultados.append(row)
        barra.progress((idx + 1) / len(skus), text=f"Backtesting: {idx+1}/{len(skus)} SKUs")

    barra.empty()
    return pd.DataFrame(resultados)

# ══════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO DA DEMANDA
# ══════════════════════════════════════════════════════════════

def classificar_demanda(serie: pd.Series) -> str:
    s = serie.dropna()
    if len(s) == 0:
        return "Indefinida"
    zeros = (s == 0).sum() / len(s)
    mean_v = s.mean()
    cv = s.std() / mean_v if mean_v > 0 else np.inf

    if zeros > 0.50:
        return "Intermitente"
    if zeros > 0.25 and cv > 0.5:
        return "Esporádica"
    if cv < 0.25:
        return "Estável"
    if cv < 0.60:
        return "Variável"
    return "Errática"

CLASSE_ICONES = {
    "Estável":      "🟢",
    "Variável":     "🟡",
    "Errática":     "🔴",
    "Intermitente": "⚪",
    "Esporádica":   "🔵",
    "Indefinida":   "⬛",
}

def detectar_tendencia(serie: pd.Series):
    s = serie.dropna()
    if len(s) < 4:
        return 0.0, "➡️ Sem dados"
    x = np.arange(len(s))
    slope, _, _, p_value, _ = stats.linregress(x, s.values)
    if p_value > 0.1:
        return slope, "➡️ Estável"
    return (slope, "📈 Crescente") if slope > 0 else (slope, "📉 Decrescente")

def detectar_sazonalidade(serie: pd.Series) -> str:
    s = serie.dropna()
    if len(s) < 13:
        return "Dados insuficientes"
    # Autocorrelação no lag 12
    acf12 = s.autocorr(lag=12)
    if pd.notna(acf12) and abs(acf12) > 0.4:
        return f"🔄 Sazonal (ACF lag12={acf12:.2f})"
    return "— Sem sazonalidade detectada"

# ══════════════════════════════════════════════════════════════
# IA — GRADIENT BOOSTING COM FEATURES DE LAG
# ══════════════════════════════════════════════════════════════

def criar_features(serie: pd.Series, n_lags=6) -> pd.DataFrame:
    df = pd.DataFrame({'y': serie.values})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['roll_mean_3'] = df['y'].shift(1).rolling(3).mean()
    df['roll_std_3']  = df['y'].shift(1).rolling(3).std().fillna(0)
    df['roll_mean_6'] = df['y'].shift(1).rolling(6).mean()
    df['roll_min_3']  = df['y'].shift(1).rolling(3).min()
    df['roll_max_3']  = df['y'].shift(1).rolling(3).max()
    return df.dropna()

N_LAGS = 6
FEAT_NAMES = (
    [f'lag_{i}' for i in range(1, N_LAGS+1)] +
    ['roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_min_3', 'roll_max_3']
)

def treinar_ia(serie: pd.Series):
    s = serie.dropna().reset_index(drop=True)
    if len(s) < 14:
        return None
    df_f = criar_features(s, N_LAGS)
    if len(df_f) < 6:
        return None
    X = df_f[FEAT_NAMES].values
    y = df_f['y'].values
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42, min_samples_leaf=2
    )
    try:
        model.fit(X, y)
        return model
    except:
        return None

def prever_ia(model, serie: pd.Series) -> float | None:
    if model is None:
        return None
    s = serie.dropna().reset_index(drop=True)
    if len(s) < N_LAGS:
        return None
    try:
        last = s.tail(N_LAGS + 3).values.astype(float)
        lags = [last[-(i)] for i in range(1, N_LAGS+1)]
        roll3 = np.mean(last[-3:])
        std3  = np.std(last[-3:]) if len(last) >= 3 else 0.0
        roll6 = np.mean(last[-min(6, len(last)):])
        min3  = np.min(last[-3:])
        max3  = np.max(last[-3:])
        feats = np.array([*lags, roll3, std3, roll6, min3, max3], dtype=float).reshape(1, -1)
        return float(max(0.0, model.predict(feats)[0]))
    except:
        return None

def prever_ia_multistep(model, serie: pd.Series, h: int = 4):
    """Rolling forecast com IA para h passos.
    Cada previsão alimenta o passo seguinte (sem data leakage futuro)."""
    if model is None:
        return [None] * h
    s = list(serie.dropna().values.astype(float))
    if len(s) < N_LAGS:
        return [None] * h
    preds = []
    for _ in range(h):
        try:
            buf = s[-(N_LAGS + 3):] if len(s) >= N_LAGS + 3 else s[:]
            buf = np.array(buf, dtype=float)
            lags  = [buf[-(i)] for i in range(1, N_LAGS + 1)]
            r3    = np.mean(buf[-3:]) if len(buf) >= 3 else np.mean(buf)
            s3    = np.std(buf[-3:])  if len(buf) >= 3 else 0.0
            r6    = np.mean(buf[-min(6, len(buf)):])
            mn3   = np.min(buf[-3:])  if len(buf) >= 3 else np.min(buf)
            mx3   = np.max(buf[-3:])  if len(buf) >= 3 else np.max(buf)
            feats = np.array([*lags, r3, s3, r6, mn3, mx3], dtype=float).reshape(1, -1)
            pred  = float(max(0.0, model.predict(feats)[0]))
            preds.append(pred)
            s.append(pred)          # <<< alimenta o próximo passo
        except Exception:
            preds.append(None)
    return preds

def wmape_ia_insample(model, serie: pd.Series):
    """WMAPE in-sample do modelo de IA"""
    s = serie.dropna().reset_index(drop=True)
    df_f = criar_features(s, N_LAGS)
    if len(df_f) < 4:
        return np.nan, np.array([]), np.array([])
    X = df_f[FEAT_NAMES].values
    y = df_f['y'].values
    preds = np.maximum(0, model.predict(X))
    return wmape(y, preds), y, preds

# ══════════════════════════════════════════════════════════════
# EXPORTAÇÃO EXCEL
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# EXPORTAÇÃO ESTILIZADA — usada por todos os botões de download
# ──────────────────────────────────────────────────────────

# Paleta de cores padrão do dashboard
_XL = {
    'navy':      '0F2B4F',
    'teal':      '0D9488',
    'teal_lt':   '14B8A6',
    'amber':     'F59E0B',
    'white':     'FFFFFF',
    'off_white': 'F8FAFC',
    'slate':     '64748B',
    'green':     '22C55E',
    'red':       'EF4444',
    'yellow':    'EAB308',
    'light_green': 'C8E6C9',
    'light_yellow':'FFF9C4',
    'light_red':   'FFCDD2',
    'light_teal':  'E0F2F1',
    'light_amber': 'FEF9EC',
}

def _xl_fmt(wb, **kw):
    """Cria formato xlsxwriter a partir de kwargs."""
    return wb.add_format(kw)

def _escrever_aba_estilizada(
    writer, df: pd.DataFrame, nome_aba: str,
    col_wmape=None,          # nome da coluna de WMAPE (para coloração condicional)
    col_wmape_escala=1.0,    # 1.0 se decimal, 100.0 se já em %
    col_metodo=None,         # coluna do melhor método (teal)
    col_classe=None,         # coluna de classificação (badge)
    freeze=True,
    col_widths: dict = None, # {nome_col: largura} override
):
    if df is None or len(df) == 0:
        return
    wb  = writer.book

    # ── Formatos base ─────────────────────────────────────
    hdr = _xl_fmt(wb, bold=True, bg_color=_XL['navy'], font_color=_XL['white'],
                  border=1, border_color='CCCCCC', align='center', valign='vcenter',
                  font_size=10, font_name='Calibri')
    row_par  = _xl_fmt(wb, bg_color=_XL['white'],     border=1, border_color='E2E8F0',
                       font_size=9, font_name='Calibri', valign='vcenter')
    row_impar= _xl_fmt(wb, bg_color=_XL['off_white'], border=1, border_color='E2E8F0',
                       font_size=9, font_name='Calibri', valign='vcenter')
    # WMAPE condicional
    fmt_excelente = _xl_fmt(wb, bg_color=_XL['light_green'],  border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', bold=True, font_color='1B5E20')
    fmt_bom       = _xl_fmt(wb, bg_color=_XL['light_teal'],   border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', bold=True, font_color='004D40')
    fmt_regular   = _xl_fmt(wb, bg_color=_XL['light_yellow'], border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', bold=True, font_color='E65100')
    fmt_critico   = _xl_fmt(wb, bg_color=_XL['light_red'],    border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', bold=True, font_color='B71C1C')
    fmt_metodo    = _xl_fmt(wb, bg_color=_XL['light_teal'],   border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', bold=True, font_color=_XL['teal'])
    fmt_num       = _xl_fmt(wb, bg_color=_XL['white'],        border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', num_format='#,##0.00', valign='vcenter')
    fmt_num_impar = _xl_fmt(wb, bg_color=_XL['off_white'],    border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', num_format='#,##0.00', valign='vcenter')
    fmt_pct       = _xl_fmt(wb, bg_color=_XL['white'],        border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', num_format='0.0%',     valign='vcenter')
    fmt_pct_impar = _xl_fmt(wb, bg_color=_XL['off_white'],    border=1, border_color='E2E8F0',
                            font_size=9, font_name='Calibri', num_format='0.0%',     valign='vcenter')
    # Título da aba
    titulo_fmt = _xl_fmt(wb, bold=True, bg_color=_XL['teal'], font_color=_XL['white'],
                         font_size=11, font_name='Calibri', valign='vcenter', align='left')
    # Legenda inferior
    leg_hdr = _xl_fmt(wb, bold=True, bg_color=_XL['slate'], font_color=_XL['white'],
                      font_size=8, font_name='Calibri')

    # ── Detectar colunas de tipo numérico / % ────────────
    cols = list(df.columns)
    n_cols = len(cols)
    wmape_idx  = cols.index(col_wmape)  if col_wmape  and col_wmape  in cols else None
    metodo_idx = cols.index(col_metodo) if col_metodo and col_metodo in cols else None

    # Colunas que devem ser tratadas como texto (SKU)
    sku_idx_list = [i for i, c in enumerate(cols)
                    if c.upper() in ('SKU', 'COD. MATERIAL', 'CODIGO', 'MATERIAL', 'ITEM')
                    or 'sku' in c.lower()]

    # Escrever na aba (dados brutos para manter valores numéricos)
    df.to_excel(writer, sheet_name=nome_aba, index=False, startrow=2, header=False)
    ws = writer.sheets[nome_aba]

    # ── Linha de título da aba (row 0) ───────────────────
    ws.set_row(0, 22)
    ws.merge_range(0, 0, 0, max(n_cols-1, 0), f'  {nome_aba.replace("_", " ").upper()}', titulo_fmt)

    # ── Cabeçalhos (row 1) ───────────────────────────────
    ws.set_row(1, 20)
    for ci, col in enumerate(cols):
        ws.write(1, ci, str(col), hdr)

    # ── Dados (a partir de row 2) ────────────────────────
    for ri, row in enumerate(df.itertuples(index=False)):
        ws.set_row(ri + 2, 16)
        bg_par = ri % 2 == 0
        for ci, val in enumerate(row):
            col_name = cols[ci]
            # Determinar formato
            is_wmape_col = (ci == wmape_idx)
            is_metodo_col= (ci == metodo_idx)
            is_pct_col   = isinstance(val, str) and str(val).strip().endswith('%')
            is_num       = isinstance(val, (int, float)) and not pd.isna(val)

            if ci in sku_idx_list:
                # SKU sempre como texto
                fmt_sku = _xl_fmt(wb, bg_color=_XL['white'] if bg_par else _XL['off_white'],
                                  border=1, border_color='E2E8F0', font_size=9,
                                  font_name='Calibri', valign='vcenter', num_format='@')
                ws.write_string(ri+2, ci, str(int(val)) if isinstance(val, float) and val == int(val)
                                else str(val) if val is not None else '—', fmt_sku)
            elif is_wmape_col:
                # Coloração condicional por faixa
                raw = val
                if isinstance(raw, str) and raw.endswith('%'):
                    try: raw = float(raw.replace('%','').replace(',','.').strip()) / 100
                    except: raw = np.nan
                if pd.isna(raw):
                    ws.write(ri+2, ci, '—', row_par if bg_par else row_impar)
                else:
                    v_pct = raw * col_wmape_escala if col_wmape_escala != 1.0 else raw
                    if v_pct < 0.20:   fmt_w = fmt_excelente
                    elif v_pct < 0.35: fmt_w = fmt_bom
                    elif v_pct < 0.60: fmt_w = fmt_regular
                    else:              fmt_w = fmt_critico
                    ws.write(ri+2, ci, f'{v_pct*100:.1f}%', fmt_w)
            elif is_metodo_col:
                ws.write(ri+2, ci, str(val) if val is not None else '—', fmt_metodo)
            elif is_num and ('wmape' in col_name.lower() or 'erro' in col_name.lower() or '%' in col_name):
                fmt_use = fmt_pct if bg_par else fmt_pct_impar
                ws.write(ri+2, ci, val, fmt_use)
            elif is_num:
                fmt_use = fmt_num if bg_par else fmt_num_impar
                ws.write(ri+2, ci, val, fmt_use)
            else:
                fmt_use = row_par if bg_par else row_impar
                ws.write(ri+2, ci, str(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else '—', fmt_use)

    # ── Larguras das colunas ─────────────────────────────
    default_w = 16
    col_w_map = col_widths or {}
    for ci, col in enumerate(cols):
        w = col_w_map.get(col, default_w)
        ws.set_column(ci, ci, w)

    # ── Freeze pane ─────────────────────────────────────
    if freeze:
        ws.freeze_panes(2, 0)

    # ── Legenda de WMAPE ─────────────────────────────────
    leg_row = len(df) + 4
    ws.write(leg_row, 0, 'Legenda WMAPE', leg_hdr)
    for j, (lbl, bg, fc) in enumerate([
        ('< 20%  — Excelente',    _XL['light_green'],  '1B5E20'),
        ('20–35% — Bom ⭐',       _XL['light_teal'],   '004D40'),
        ('35–60% — Regular',      _XL['light_yellow'], 'E65100'),
        ('> 60%  — Crítico',      _XL['light_red'],    'B71C1C'),
    ], 1):
        ws.write(leg_row + j, 0, lbl, _xl_fmt(wb, bg_color=bg, font_color=fc,
                                               font_size=8, font_name='Calibri', bold=True,
                                               border=1, border_color='CCCCCC'))


def exportar_excel_visual(abas: dict, filename_prefix: str = 'forecast') -> io.BytesIO:
    """
    abas: dict { nome_aba: { 'df': DataFrame, 'col_wmape': str|None,
                              'col_metodo': str|None, 'col_widths': dict|None } }
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        for nome_aba, cfg in abas.items():
            _escrever_aba_estilizada(
                writer,
                df=cfg.get('df'),
                nome_aba=nome_aba,
                col_wmape=cfg.get('col_wmape'),
                col_wmape_escala=cfg.get('col_wmape_escala', 1.0),
                col_metodo=cfg.get('col_metodo'),
                col_widths=cfg.get('col_widths'),
            )
        # Aba de legenda geral
        wb = writer.book
        ls = wb.add_worksheet('Legenda_Geral')
        ls.set_column(0, 0, 35)
        hf = wb.add_format({'bold':True,'bg_color':_XL['navy'],'font_color':_XL['white'],
                            'font_size':10,'font_name':'Calibri','border':1})
        ls.write(0, 0, 'Legenda Geral do Arquivo', hf)
        for i, (txt, bg, fc) in enumerate([
            ('WMAPE < 20%  — Excelente (verde escuro)',    _XL['light_green'],  '1B5E20'),
            ('WMAPE 20–35% — Bom ⭐ (teal)',              _XL['light_teal'],   '004D40'),
            ('WMAPE 35–60% — Regular (amarelo)',           _XL['light_yellow'], 'E65100'),
            ('WMAPE > 60%  — Crítico (vermelho)',          _XL['light_red'],    'B71C1C'),
            ('Método Base  — Melhor método (teal)',        _XL['light_teal'],   _XL['teal']),
        ], 1):
            ls.write(i, 0, txt, wb.add_format({'bg_color':bg,'font_color':fc,'font_size':9,
                                               'font_name':'Calibri','bold':True,'border':1}))
    buf.seek(0)
    return buf


# Manter compatibilidade com código legado
def exportar_excel(df_bt, df_ia, df_top10):
    abas = {
        'Backtesting_Completo': {'df': df_bt,   'col_wmape': 'melhor_wmape', 'col_metodo': 'melhor_metodo'},
        'Sugestoes_IA':         {'df': df_ia,   'col_wmape': 'wmape_melhor', 'col_metodo': 'melhor_metodo'},
        'Top10_Piores_WMAPE':   {'df': df_top10,'col_wmape': 'wmape_pct',    'col_wmape_escala': 100.0,
                                 'col_metodo': 'melhor_metodo'},
    }
    return exportar_excel_visual(abas)

# ══════════════════════════════════════════════════════════════
# GERAÇÃO DE TEXTO DE SUGESTÃO
# ══════════════════════════════════════════════════════════════

def gerar_sugestao(row_ia, wmape_val):
    sugestoes = []
    classe = str(row_ia.get('classificacao', ''))
    tend   = str(row_ia.get('tendencia', ''))
    metodo = str(row_ia.get('melhor_metodo', ''))

    if classe == 'Intermitente':
        sugestoes.append("Use **Croston** ou **SBA (Syntetos-Boylan)** — ideal para demanda esporádica com muitos zeros.")
    elif classe == 'Errática':
        sugestoes.append("Demanda errática: avalie **SES com alpha alto** ou revisão manual do histórico para remover outliers.")
    elif classe == 'Estável':
        sugestoes.append(f"Demanda estável: **{metodo}** é adequado. Considere ampliar janela de previsão.")

    if '📈' in tend:
        sugestoes.append("Tendência **crescente** identificada — use **Holt** ou **Holt-Winters** para capturar a tendência.")
    elif '📉' in tend:
        sugestoes.append("Tendência **decrescente** — use Holt e revise se o SKU está em declínio de ciclo de vida.")

    pred_ia   = row_ia.get('previsao_ia')
    pred_stat = row_ia.get('previsao_estatistica', 0)
    if pred_ia and pred_stat > 0:
        diff_pct = abs(pred_ia - pred_stat) / (pred_stat + 1e-9)
        if diff_pct > 0.35:
            sugestoes.append(f"⚠️ IA e modelo estatístico divergem em **{diff_pct*100:.0f}%** — investigar padrão recente nos dados.")

    if pd.notna(wmape_val):
        if wmape_val > 0.60:
            sugestoes.append("🔴 WMAPE crítico (>60%) — revisar dados históricos, verificar outliers ou demanda atípica.")
        elif wmape_val > 0.40:
            sugestoes.append("🟡 WMAPE moderado — a **previsão combinada IA+estatística** pode reduzir o erro.")

    if not sugestoes:
        sugestoes.append(f"Manter **{metodo}**. SKU com bom comportamento de previsão — monitorar mensalmente.")

    return " ".join(sugestoes)

# ══════════════════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    # ─── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 Forecast V5")
        st.markdown("*IA + Backtesting + Análise SKU*")
        st.divider()

        uploaded_file = st.file_uploader(
            "📂 Faça upload do arquivo Excel",
            type=["xlsx"],
            help="Selecione o arquivo .xlsx com as abas: Base_Limpa, Estatistica_SKU, etc."
        )

        if uploaded_file is None:
            st.info("👆 Faça o upload do arquivo Excel para começar.")
            st.stop()

        arquivo = uploaded_file

        st.divider()
        st.markdown("**⚙️ Configurações**")
        n_test = st.slider("Períodos de backtesting (walk-forward)", 2, 6, 3,
                           help="Quantos períodos finais usar para avaliar cada método")
        peso_ia = st.slider("Peso da IA na previsão combinada (%)", 0, 100, 50,
                            help="O restante é dado ao melhor método estatístico") / 100
        st.divider()
        rodar = st.button("🚀 Rodar Pipeline Completo", type="primary", use_container_width=True)
        if st.button("🗑️ Limpar Cache", use_container_width=True):
            st.cache_data.clear()
            for k in ['df_backtest', 'df_ia']:
                st.session_state.pop(k, None)
            st.rerun()

    # ─── CABEÇALHO ─────────────────────────────────────────────
    st.title("📈 Forecast Inteligente V5")
    st.caption("Pipeline de previsão de demanda com IA | Backtesting Retrospectivo | Análise por SKU")

    # ─── CARREGAR DADOS ────────────────────────────────────────
    with st.spinner("Carregando arquivo Excel..."):
        dados = carregar_excel(arquivo)

    if dados.get('erro'):
        st.error(dados['erro'])
        st.stop()

    df_base = dados['base']
    _av1 = dados.get('avaliacao')
    _av2 = dados.get('avaliacao_sku')
    df_aval = _av1 if _av1 is not None else _av2

    if df_base is None:
        st.error("Aba com histórico de demanda não encontrada. Verifique o nome da aba (ex: Base_Limpa).")
        with st.expander("Abas encontradas no arquivo"):
            st.write(dados.get('abas', []))
        st.stop()

    col_sku     = encontrar_coluna(df_base, 'sku', 'material', 'codigo', 'cod', 'produto', 'item')
    col_periodo = encontrar_coluna(df_base, 'periodo', 'mes', 'data', 'competencia', 'referencia')
    col_ano     = encontrar_coluna(df_base, 'ano', 'year')
    col_demanda = encontrar_coluna(df_base, 'demanda', 'consumo', 'quantidade', 'qtd', 'qtde', 'realizado')

    if not col_sku or not col_demanda:
        st.error(f"Colunas de SKU e/ou demanda não encontradas. Colunas disponíveis: {list(df_base.columns)}")
        st.stop()

    df_base[col_demanda] = pd.to_numeric(df_base[col_demanda], errors='coerce').fillna(0)

    # Criar coluna de período combinado Ano+Mês para contagem correta
    if col_ano and col_periodo:
        df_base['_periodo_combined'] = df_base[col_ano].astype(str) + '_' + df_base[col_periodo].astype(str)
        col_periodo_count = '_periodo_combined'
    elif col_periodo:
        col_periodo_count = col_periodo
    else:
        col_periodo_count = None

    # Colunas WMAPE do arquivo original (se existir)
    wmape_original = None
    if df_aval is not None:
        col_wmape_orig = encontrar_coluna(df_aval, 'wmape')
        col_sku_aval   = encontrar_coluna(df_aval, 'sku', 'material', 'codigo')
        if col_wmape_orig and col_sku_aval:
            wmape_original = df_aval.groupby(col_sku_aval)[col_wmape_orig].mean().to_dict()

    # ─── MÉTRICAS RÁPIDAS ──────────────────────────────────────
    n_skus = df_base[col_sku].nunique()
    n_reg  = len(df_base)
    n_per  = df_base[col_periodo_count].nunique() if col_periodo_count else "—"
    med_dem = df_base[col_demanda].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏷️ SKUs únicos",        n_skus)
    c2.metric("📋 Registros",          f"{n_reg:,}")
    c3.metric("📅 Períodos",            n_per)
    c4.metric("📊 Demanda média",       f"{med_dem:.1f}")
    c5.metric("🔧 Métodos disponíveis", len(METODOS))
    st.divider()

    # ─── ESTADO DA SESSÃO ──────────────────────────────────────
    if 'df_backtest' not in st.session_state:
        st.session_state['df_backtest'] = None
    if 'df_ia' not in st.session_state:
        st.session_state['df_ia'] = None

    # ─── RODAR PIPELINE ────────────────────────────────────────
    if rodar:
        # ETAPA 1: Backtesting
        with st.status("🔄 Executando pipeline...", expanded=True) as status:
            st.write("📊 Etapa 1/3 — Backtesting walk-forward por SKU e método...")
            df_bt = rodar_backtesting(df_base, col_sku, col_demanda, n_test)
            st.session_state['df_backtest'] = df_bt

            # ETAPA 2: IA por SKU
            st.write("🤖 Etapa 2/3 — Treinando modelos de IA por SKU...")
            ia_rows = []
            skus_all = df_base[col_sku].unique()
            barra2 = st.progress(0, text="Treinando IA...")

            for idx, sku in enumerate(skus_all):
                serie = df_base[df_base[col_sku] == sku][col_demanda].reset_index(drop=True)

                model_ia  = treinar_ia(serie)
                pred_ia   = prever_ia(model_ia, serie)
                wmape_ia  = np.nan

                # Melhor método estatístico
                row_bt = df_bt[df_bt['sku'] == sku]
                if len(row_bt) > 0:
                    melhor    = row_bt.iloc[0]['melhor_metodo']
                    melhor_w  = row_bt.iloc[0]['melhor_wmape']
                else:
                    melhor, melhor_w = 'MA-3', np.nan

                fn_melhor = METODOS[melhor]
                pred_stat = fn_melhor(serie, h=1)[0]

                # Previsão combinada
                if pred_ia is not None:
                    pred_comb = peso_ia * pred_ia + (1 - peso_ia) * pred_stat
                    if model_ia:
                        wmape_ia, _, _ = wmape_ia_insample(model_ia, serie)
                else:
                    pred_comb = pred_stat

                # Características da demanda
                classe  = classificar_demanda(serie)
                _, tend = detectar_tendencia(serie)
                sazon   = detectar_sazonalidade(serie)

                row_dict = {
                    'sku':                   sku,
                    'classificacao':         classe,
                    'tendencia':             tend,
                    'sazonalidade':          sazon,
                    'n_periodos':            len(serie),
                    'media_historica':       round(serie.mean(), 2),
                    'cv':                    round(serie.std() / serie.mean(), 3) if serie.mean() > 0 else np.nan,
                    'previsao_estatistica':  round(pred_stat, 2),
                    'previsao_ia':           round(pred_ia, 2) if pred_ia is not None else np.nan,
                    'previsao_combinada':    round(pred_comb, 2),
                    'melhor_metodo':         melhor,
                    'wmape_melhor':          melhor_w,
                    'wmape_ia_insample':     wmape_ia,
                }
                if wmape_original:
                    row_dict['wmape_original'] = wmape_original.get(sku, np.nan)

                ia_rows.append(row_dict)
                barra2.progress((idx+1)/len(skus_all), text=f"IA: {idx+1}/{len(skus_all)}")

            barra2.empty()
            st.session_state['df_ia'] = pd.DataFrame(ia_rows)

            # ETAPA 3: Finalização
            st.write("✅ Etapa 3/3 — Consolidando resultados...")
            status.update(label="✅ Pipeline concluído com sucesso!", state="complete")

        st.rerun()

    df_bt = st.session_state.get('df_backtest')
    df_ia = st.session_state.get('df_ia')

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Simulação Retrospectiva",
        "🎯 Seletor de Método",
        "🔍 Análise por SKU",
        "🤖 IA + Previsão",
        "📋 Top 10 Piores WMAPE",
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1: SIMULAÇÃO RETROSPECTIVA
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("📊 Comparação Retrospectiva — Walk-Forward Validation")
        st.markdown("""
        Para cada SKU, o pipeline simula como **cada método teria se saído nos últimos N períodos**,
        treinando apenas com dados passados (sem *data leakage*). Isso responde: *"Se eu tivesse usado
        este método, qual seria o WMAPE real?"*
        """)

        if df_bt is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        wmape_cols = [c for c in df_bt.columns if c.startswith('wmape_')]
        metodos_nomes = [c.replace('wmape_', '') for c in wmape_cols]

        # Resumo por método
        # WMAPE original (do arquivo) por SKU para calcular ganho
        wmape_orig_por_sku = {}
        if wmape_original:
            wmape_orig_por_sku = wmape_original  # dict sku → wmape
        med_wmape_orig = np.nanmedian(list(wmape_orig_por_sku.values())) if wmape_orig_por_sku else np.nan

        resumo = []
        for col, nome in zip(wmape_cols, metodos_nomes):
            vals      = df_bt[col].dropna()
            n_melhor  = (df_bt['melhor_metodo'] == nome).sum()
            n_abaixo35 = (vals < 0.35).sum()
            med_v5    = vals.median()
            ganho_str = "—"
            if pd.notna(med_wmape_orig) and med_wmape_orig > 0:
                ganho = (med_wmape_orig - med_v5) / med_wmape_orig * 100
                sinal = "▲" if ganho >= 0 else "▼"
                ganho_str = f"{sinal} {abs(ganho):.1f}%"
            resumo.append({
                'Método':              nome,
                'Descrição':           METODOS_DESC.get(nome, ''),
                'SKUs — Melhor':       int(n_melhor),
                '% SKUs':             f"{n_melhor/len(df_bt)*100:.0f}%",
                'SKUs WMAPE < 35%':    int(n_abaixo35),
                '% SKUs < 35%':       f"{n_abaixo35/max(len(vals),1)*100:.0f}%",
                'Ganho vs Original':   ganho_str,
            })

        df_res = pd.DataFrame(resumo).sort_values('SKUs WMAPE < 35%', ascending=False)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Box plot
            melt_data = []
            for col, nome in zip(wmape_cols, metodos_nomes):
                for v in df_bt[col].dropna():
                    melt_data.append({'Método': nome, 'WMAPE (%)': v * 100})
            df_melt = pd.DataFrame(melt_data)

            ordem = df_melt.groupby('Método')['WMAPE (%)'].median().sort_values().index.tolist()
            fig_box = px.box(
                df_melt, x='Método', y='WMAPE (%)',
                title='Distribuição do WMAPE por Método',
                color='Método', category_orders={'Método': ordem},
                template='plotly_white', points='outliers'
            )
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-30)
            fig_box.add_hline(y=20, line_dash='dash', line_color='green',
                              annotation_text='Meta 20%', annotation_position='right')
            st.plotly_chart(fig_box, use_container_width=True)

        with col_b:
            # Distribuição dos melhores métodos
            counts = df_bt['melhor_metodo'].value_counts().reset_index()
            counts.columns = ['Método', 'Qtd']
            fig_pie = px.pie(
                counts, values='Qtd', names='Método',
                title='Melhor Método por SKU (backtesting)',
                template='plotly_white', hole=0.4
            )
            fig_pie.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Comparação com WMAPE original (se disponível)
        if wmape_original and df_ia is not None:
            st.divider()
            st.markdown("### 📉 Ganho vs Arquivo Original")
            df_comp = df_ia[['sku', 'wmape_melhor']].copy()
            df_comp['wmape_original'] = df_comp['sku'].apply(lambda x: wmape_original.get(x, np.nan))
            df_comp = df_comp.dropna(subset=['wmape_original', 'wmape_melhor'])
            if len(df_comp) > 0:
                df_comp['ganho'] = df_comp['wmape_original'] - df_comp['wmape_melhor']
                df_comp['ganho_pct'] = df_comp['ganho'] / (df_comp['wmape_original'] + 1e-9) * 100
                ganho_med = df_comp['ganho_pct'].median()
                n_melhora = (df_comp['ganho'] > 0).sum()

                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("SKUs com WMAPE melhorado", f"{n_melhora}/{len(df_comp)}")
                gc2.metric("Ganho mediano no WMAPE", f"{ganho_med:.1f}%")
                gc3.metric("Total SKUs comparados", len(df_comp))

                fig_ganho = px.scatter(
                    df_comp, x='wmape_original', y='wmape_melhor',
                    hover_data=['sku', 'ganho_pct'],
                    title='WMAPE Original vs WMAPE Novo (pipeline V5)',
                    labels={'wmape_original': 'WMAPE Original', 'wmape_melhor': 'WMAPE Novo (V5)'},
                    template='plotly_white', color='ganho',
                    color_continuous_scale='RdYlGn'
                )
                fig_ganho.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                                    line=dict(dash='dash', color='gray'))
                fig_ganho.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_ganho, use_container_width=True)

        # ── TABELA MENSAL POR SKU ──────────────────────────────
        st.divider()
        st.markdown("### 📋 Comparativo Mensal por SKU")
        st.markdown("Visão mensal com demanda planejada (original), realizado e melhor previsão do pipeline V5.")

        if df_ia is not None and col_periodo and col_ano:
            # Montar períodos disponíveis como Ano+Mês
            meses_ord = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                         'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
            df_mensal = df_base[[col_sku, col_ano, col_periodo, col_demanda]].copy()
            df_mensal.columns = ['sku','ano','mes','realizado']
            df_mensal['mes_num'] = df_mensal['mes'].astype(str).str.lower().str[:3].map(meses_ord).fillna(0).astype(int)
            df_mensal['periodo_lbl'] = df_mensal['mes'].astype(str).str.lower().str[:3] + '/' + df_mensal['ano'].astype(str).str[-2:]

            periodos_disp = (df_mensal[['ano','mes_num','periodo_lbl']]
                             .drop_duplicates()
                             .sort_values(['ano','mes_num'])['periodo_lbl'].tolist())

            col_f1, col_f2 = st.columns([2, 4])
            per_sel = col_f1.selectbox("📅 Filtrar por período", options=periodos_disp,
                                        index=len(periodos_disp)-1 if periodos_disp else 0,
                                        key='tab1_per_sel')

            df_per = df_mensal[df_mensal['periodo_lbl'] == per_sel].copy()

            # Agregar por SKU (caso tenha duplicatas)
            df_per = df_per.groupby('sku', as_index=False)['realizado'].sum()

            # ── Previsão original e WMAPE do período filtrado ────────
            prev_orig_map   = {}   # sku → previsão do arquivo no mês
            wmape_orig_per_map = {}  # sku → WMAPE do arquivo naquele período (Σerro/Σdemanda)

            if df_aval is not None:
                col_prev_orig = encontrar_coluna(df_aval, 'prev', 'previsao', 'forecast', 'hibrido')
                col_mes_aval  = encontrar_coluna(df_aval, 'mes', 'periodo', 'referencia')
                col_sku_aval2 = encontrar_coluna(df_aval, 'sku', 'material', 'codigo')
                col_ano_aval  = encontrar_coluna(df_aval, 'ano', 'year')

                if col_prev_orig and col_mes_aval and col_sku_aval2:
                    df_aval_f = df_aval.copy()
                    df_aval_f['_mes_abr'] = df_aval_f[col_mes_aval].astype(str).str.lower().str[:3]
                    mes_sel_abr = per_sel[:3].lower()

                    # Filtrar também pelo ano se possível
                    ano_sel = per_sel[-2:]  # ex: '25' de 'jan/25'
                    if col_ano_aval:
                        df_aval_f['_ano_abr'] = df_aval_f[col_ano_aval].astype(str).str[-2:]
                        df_aval_per = df_aval_f[
                            (df_aval_f['_mes_abr'] == mes_sel_abr) &
                            (df_aval_f['_ano_abr'] == ano_sel)
                        ]
                        # Fallback: só mês se não achar por ano+mês
                        if len(df_aval_per) == 0:
                            df_aval_per = df_aval_f[df_aval_f['_mes_abr'] == mes_sel_abr]
                    else:
                        df_aval_per = df_aval_f[df_aval_f['_mes_abr'] == mes_sel_abr]

                    prev_orig_map = df_aval_per.groupby(col_sku_aval2)[col_prev_orig].mean().to_dict()

                    # WMAPE do período: Soma_Erro_Hibrido / Soma_Consumo (do arquivo)
                    col_soma_erro = encontrar_coluna(df_aval, 'somaerro', 'errohibrido', 'somaerrohibrido')
                    col_soma_dem  = encontrar_coluna(df_aval, 'somaconsumо', 'somaconsуmo', 'somaconsuma')
                    # busca manual mais robusta
                    _soma_erro_col, _soma_dem_col = None, None
                    for c in df_aval.columns:
                        cn = ''.join(filter(str.isalpha, str(c))).lower()
                        if 'somaerrо' in cn or ('soma' in cn and 'erro' in cn and 'hibrido' in cn):
                            _soma_erro_col = c
                        if 'somaconsumo' in cn or ('soma' in cn and 'consumo' in cn):
                            _soma_dem_col = c

                    # fallback: procurar por nome exato
                    for c in df_aval.columns:
                        if str(c) == 'Soma_Erro_Hibrido':
                            _soma_erro_col = c
                        if str(c) == 'Soma_Consumo':
                            _soma_dem_col = c

                    if _soma_erro_col and _soma_dem_col:
                        grp = df_aval_per.groupby(col_sku_aval2).agg(
                            _se=(  _soma_erro_col, 'sum'),
                            _sd=(_soma_dem_col,  'sum')
                        )
                        grp['_wmape_per'] = np.where(
                            grp['_sd'] > 0,
                            grp['_se'].abs() / grp['_sd'],
                            np.nan
                        )
                        wmape_orig_per_map = grp['_wmape_per'].to_dict()

            # Melhor previsão V5
            melhor_prev_map = df_ia.set_index('sku')['previsao_combinada'].to_dict() if df_ia is not None else {}
            wmape_v5_map    = df_ia.set_index('sku')['wmape_melhor'].to_dict() if df_ia is not None else {}

            df_per['prev_original']  = df_per['sku'].apply(lambda x: prev_orig_map.get(x, np.nan))
            df_per['prev_v5']        = df_per['sku'].apply(lambda x: melhor_prev_map.get(x, np.nan))
            df_per['wmape_v5']       = df_per['sku'].apply(lambda x: wmape_v5_map.get(x, np.nan))
            # WMAPE Original agora é do período, não mais a média anual
            df_per['wmape_original'] = df_per['sku'].apply(
                lambda x: wmape_orig_per_map.get(x,
                    wmape_orig_per_map.get(float(x) if str(x).isdigit() else x, np.nan))
            )

            # Calcular erro percentual do período
            df_per['erro_prev_orig'] = np.where(
                df_per['realizado'] > 0,
                abs(df_per['prev_original'] - df_per['realizado']) / df_per['realizado'] * 100,
                np.nan
            )
            df_per['erro_prev_v5'] = np.where(
                df_per['realizado'] > 0,
                abs(df_per['prev_v5'] - df_per['realizado']) / df_per['realizado'] * 100,
                np.nan
            )

            df_per_show = df_per.rename(columns={
                'sku':           'SKU',
                'realizado':     'Realizado',
                'prev_original': 'Prev. Original (arquivo)',
                'prev_v5':       'Melhor Prev. V5',
                'wmape_v5':      'WMAPE V5 (backtesting)',
                'wmape_original':'WMAPE Original (período)',
                'erro_prev_orig':'Erro % Prev. Original',
                'erro_prev_v5':  'Erro % Prev. V5',
            }).copy()

            # Formatar colunas WMAPE — já estão em decimal (0-1), multiplicar por 100
            for col_fmt in ['WMAPE V5 (backtesting)', 'WMAPE Original (período)']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                    )
            # Erros já estão em % (0-100), não multiplicar
            for col_fmt in ['Erro % Prev. Original', 'Erro % Prev. V5']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                    )
            for col_fmt in ['Prev. Original (arquivo)', 'Melhor Prev. V5', 'Realizado']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(
                        lambda x: f"{x:,.1f}" if pd.notna(x) else "—"
                    )

            st.caption(f"Período selecionado: **{per_sel}** — {len(df_per_show)} SKUs")
            st.dataframe(df_per_show, use_container_width=True, hide_index=True)

            # Mini-resumo do período
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            total_real = df_per['realizado'].sum()
            total_v5   = df_per['prev_v5'].sum()
            r_col1.metric("Total Realizado", f"{total_real:,.0f}")
            r_col2.metric("Total Prev. V5",  f"{total_v5:,.0f}")
            r_col3.metric("Diferença",        f"{total_v5 - total_real:,.0f}",
                          delta=f"{(total_v5-total_real)/max(total_real,1)*100:.1f}%")

            # Export com visual do dashboard
            df_mensal_exp = df_per.rename(columns={
                'sku':'SKU','realizado':'Realizado',
                'prev_original':'Prev. Original (arquivo)',
                'prev_v5':'Melhor Prev. V5',
                'wmape_v5':'WMAPE V5 (backtesting)',
                'wmape_original':'WMAPE Original (período)',
                'erro_prev_orig':'Erro % Prev. Original',
                'erro_prev_v5':'Erro % Prev. V5',
            }).copy()
            buf_mensal = exportar_excel_visual({
                f'Mensal_{per_sel.replace("/","_")}': {
                    'df': df_mensal_exp,
                    'col_wmape': 'WMAPE V5 (backtesting)',
                    'col_widths': {'SKU':12,'Realizado':14,'Prev. Original (arquivo)':22,
                                   'Melhor Prev. V5':18,'WMAPE V5 (backtesting)':22,
                                   'WMAPE Original (período)':24,'Erro % Prev. Original':22,
                                   'Erro % Prev. V5':18},
                }
            })
            r_col4.download_button(
                "📥 Exportar Excel",
                data=buf_mensal,
                file_name=f"mensal_{per_sel.replace('/','_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Rode o pipeline completo para visualizar a tabela mensal.")

    # ─────────────────────────────────────────────────────────
    # TAB 2: SELETOR DE MÉTODO
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🎯 Melhor Método por SKU")
        st.markdown("Baseado no backtesting walk-forward, este painel mostra qual método "
                    "minimiza o WMAPE para cada SKU e como as características da demanda "
                    "influenciam essa escolha.")

        if df_bt is None or df_ia is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        df_sel = df_bt[['sku', 'melhor_metodo', 'melhor_wmape']].merge(
            df_ia[['sku', 'classificacao', 'tendencia', 'cv', 'n_periodos']],
            on='sku', how='left'
        )

        # Filtros
        fc1, fc2, fc3 = st.columns(3)
        f_metodo = fc1.multiselect("Filtrar método",     df_sel['melhor_metodo'].dropna().unique())
        f_classe = fc2.multiselect("Filtrar classificação", df_sel['classificacao'].dropna().unique())
        f_wmape_max = fc3.slider("WMAPE máximo (%)", 0, 200, 100)

        df_f = df_sel.copy()
        if f_metodo: df_f = df_f[df_f['melhor_metodo'].isin(f_metodo)]
        if f_classe: df_f = df_f[df_f['classificacao'].isin(f_classe)]
        df_f = df_f[df_f['melhor_wmape'].fillna(999) <= f_wmape_max / 100]

        df_display = df_f.copy()
        df_display['melhor_wmape'] = (df_display['melhor_wmape'] * 100).round(1).astype(str) + '%'
        df_display['cv'] = df_display['cv'].round(3)

        df_sel_show = df_display.rename(columns={
            'sku': 'SKU', 'melhor_metodo': 'Melhor Método',
            'melhor_wmape': 'WMAPE', 'classificacao': 'Classificação',
            'tendencia': 'Tendência', 'cv': 'CV', 'n_periodos': 'N Períodos'
        })
        st.dataframe(df_sel_show, use_container_width=True, hide_index=True)
        st.caption(f"{len(df_f)} SKUs exibidos")
        # Export estilizado — valores numéricos limpos para o Excel
        df_sel_exp = df_f.rename(columns={
            'sku': 'SKU', 'melhor_metodo': 'Melhor Método',
            'melhor_wmape': 'WMAPE', 'classificacao': 'Classificação',
            'tendencia': 'Tendência', 'cv': 'CV', 'n_periodos': 'N Períodos'
        }).copy()
        buf_sel = exportar_excel_visual({
            'Melhor_Metodo_por_SKU': {
                'df': df_sel_exp,
                'col_wmape': 'WMAPE',
                'col_metodo': 'Melhor Método',
                'col_widths': {'SKU':14,'Melhor Método':18,'WMAPE':14,
                               'Classificação':16,'Tendência':20,'CV':10,'N Períodos':14},
            }
        })
        st.download_button(
            "📥 Exportar Melhor Método por SKU (.xlsx)",
            data=buf_sel,
            file_name=f"melhor_metodo_sku_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        col_l, col_r = st.columns(2)

        with col_l:
            # Heatmap classificação × método
            cross = pd.crosstab(
                df_sel['classificacao'].fillna('N/A'),
                df_sel['melhor_metodo'].fillna('N/A')
            )
            fig_heat = px.imshow(
                cross, text_auto=True,
                title='Classificação da Demanda × Melhor Método',
                color_continuous_scale='Blues', template='plotly_white',
                aspect='auto'
            )
            fig_heat.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_r:
            # WMAPE médio por classificação
            df_cv_wmape = df_sel.copy()
            df_cv_wmape['wmape_pct'] = df_cv_wmape['melhor_wmape'] * 100
            fig_cls = px.box(
                df_cv_wmape.dropna(subset=['wmape_pct', 'classificacao']),
                x='classificacao', y='wmape_pct',
                color='classificacao',
                title='WMAPE por Classificação da Demanda',
                template='plotly_white',
                labels={'classificacao': 'Classificação', 'wmape_pct': 'WMAPE (%)'}
            )
            fig_cls.update_layout(showlegend=False)
            st.plotly_chart(fig_cls, use_container_width=True)

        # Recomendações automáticas
        st.divider()
        st.markdown("### 💡 Regras de Recomendação Automática")
        regras = [
            ("🔵 Intermitente",   "Croston ou SBA",     "> 50% de zeros na série"),
            ("🔴 Errática",       "SES (alpha alto)",   "CV > 0.6"),
            ("🟡 Variável",       "SES ou Holt",        "CV entre 0.25 e 0.6"),
            ("🟢 Estável",        "MA-6 ou SES",        "CV < 0.25"),
            ("📈 Tendência ↑",    "Holt ou Holt-Winters","Slope significativo (p<0.1)"),
            ("📉 Tendência ↓",    "Holt",               "Slope negativo significativo"),
            ("🔄 Sazonal",        "Holt-Winters",       "ACF lag-12 > 0.4"),
        ]
        df_reg = pd.DataFrame(regras, columns=['Perfil', 'Método Recomendado', 'Critério'])
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    # TAB 3: ANÁLISE POR SKU
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("🔍 Análise Detalhada por SKU")

        skus_lista = sorted(df_base[col_sku].dropna().unique())
        sku_sel = st.selectbox("Selecione o SKU", options=skus_lista, key='sku_sel_tab3')

        serie_sku = (df_base[df_base[col_sku] == sku_sel][col_demanda]
                     .reset_index(drop=True).astype(float))

        if col_periodo:
            periodos_sku = (df_base[df_base[col_sku] == sku_sel][col_periodo]
                            .reset_index(drop=True))
            x_labels = periodos_sku.astype(str).tolist()
        else:
            x_labels = list(range(len(serie_sku)))

        # Métricas do SKU
        cls_sku = classificar_demanda(serie_sku)
        slope_sku, tend_sku = detectar_tendencia(serie_sku)
        sazon_sku = detectar_sazonalidade(serie_sku)
        cv_sku = serie_sku.std() / serie_sku.mean() if serie_sku.mean() > 0 else 0
        zeros_pct = (serie_sku == 0).mean() * 100

        ms1, ms2, ms3, ms4, ms5, ms6 = st.columns(6)
        ms1.metric("Média",          f"{serie_sku.mean():.1f}")
        ms2.metric("Desvio Padrão",  f"{serie_sku.std():.1f}")
        ms3.metric("CV",             f"{cv_sku:.3f}")
        ms4.metric("Zeros",          f"{zeros_pct:.0f}%")
        ms5.metric("Classificação",  f"{CLASSE_ICONES.get(cls_sku,'')} {cls_sku}")
        ms6.metric("Tendência",      tend_sku)

        # Gráfico principal
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=x_labels, y=serie_sku.values,
            mode='lines+markers', name='Histórico',
            line=dict(color='#2c3e50', width=2),
            marker=dict(size=5, color='#2c3e50'),
            fill='tozeroy', fillcolor='rgba(44,62,80,0.07)'
        ))

        # Linha de tendência
        if len(serie_sku) >= 4:
            x_arr = np.arange(len(serie_sku))
            s_, i_ = np.polyfit(x_arr, serie_sku.values, 1)
            trend_ = s_ * x_arr + i_
            fig_main.add_trace(go.Scatter(
                x=x_labels, y=trend_,
                mode='lines', name='Tendência linear',
                line=dict(color='#e74c3c', width=1.5, dash='dash')
            ))

        # Média histórica
        fig_main.add_hline(y=serie_sku.mean(), line_dash='dot',
                           line_color='#27ae60', annotation_text='Média')

        # Previsões (ponto futuro)
        cores = px.colors.qualitative.Pastel
        for i, (nome, fn) in enumerate(METODOS.items()):
            try:
                pred_v = fn(serie_sku, h=1)[0]
                fig_main.add_trace(go.Scatter(
                    x=[f"Próx."], y=[pred_v],
                    mode='markers', name=f'{nome}: {pred_v:.1f}',
                    marker=dict(size=13, symbol='diamond',
                                color=cores[i % len(cores)],
                                line=dict(width=1, color='#333'))
                ))
            except:
                pass

        fig_main.update_layout(
            title=f'Histórico + Previsões — SKU: {sku_sel}',
            xaxis_title='Período', yaxis_title='Demanda',
            template='plotly_white', hovermode='x unified',
            height=420, legend=dict(orientation='h', yanchor='bottom', y=-0.4)
        )
        st.plotly_chart(fig_main, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Tabela comparativa de previsões
            rows_prev = []
            best_method = None
            if df_bt is not None:
                row_bt = df_bt[df_bt['sku'] == sku_sel]
                if len(row_bt) > 0:
                    best_method = row_bt.iloc[0]['melhor_metodo']

            for nome, fn in METODOS.items():
                try:
                    pred_v = fn(serie_sku, h=1)[0]
                except:
                    pred_v = np.nan
                w_bt = np.nan
                if df_bt is not None:
                    rb = df_bt[df_bt['sku'] == sku_sel]
                    if len(rb) > 0:
                        w_bt = rb.iloc[0].get(f'wmape_{nome}', np.nan)

                rows_prev.append({
                    'Método':       nome,
                    'Previsão':     round(pred_v, 2) if pd.notna(pred_v) else '—',
                    'WMAPE BT':     f"{w_bt*100:.1f}%" if pd.notna(w_bt) else '—',
                    'Recomendado':  '⭐ Melhor' if nome == best_method else '',
                })

            df_tbl = pd.DataFrame(rows_prev)
            st.markdown("**Previsões próximo período**")
            st.dataframe(df_tbl, use_container_width=True, hide_index=True)

        with col_b:
            # Autocorrelação
            if len(serie_sku) >= 8:
                max_lag = min(12, len(serie_sku)//2)
                acf_vals = [serie_sku.autocorr(lag=i) for i in range(1, max_lag+1)]
                fig_acf = go.Figure(go.Bar(
                    x=[f'L{i}' for i in range(1, max_lag+1)], y=acf_vals,
                    marker_color=['#27ae60' if v > 0 else '#e74c3c' for v in acf_vals]
                ))
                conf_95 = 1.96 / np.sqrt(len(serie_sku))
                fig_acf.add_hline(y=conf_95,  line_dash='dot', line_color='gray')
                fig_acf.add_hline(y=-conf_95, line_dash='dot', line_color='gray')
                fig_acf.update_layout(
                    title='Autocorrelação (ACF)',
                    yaxis_title='ACF', template='plotly_white',
                    height=300
                )
                st.plotly_chart(fig_acf, use_container_width=True)
            else:
                st.info("Série curta — autocorrelação indisponível.")

        # Informações qualitativas
        st.divider()
        ci1, ci2, ci3 = st.columns(3)
        ci1.info(f"**🔍 Classificação:** {cls_sku}\n\n{CLASSE_ICONES.get(cls_sku,'')} Perfil de demanda identificado com base em CV e frequência de zeros.")
        ci2.info(f"**📅 Sazonalidade:**\n\n{sazon_sku}")
        ci3.info(f"**📐 Tendência:**\n\n{tend_sku} (slope = {slope_sku:.3f} unid./período)")

    # ─────────────────────────────────────────────────────────
    # TAB 4: IA + PREVISÃO
    # ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("🤖 IA em Conjunto com Metodologia Estatística")

        if df_ia is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        with st.expander("ℹ️ Como funciona a IA neste pipeline?"):
            st.markdown(f"""
            **Arquitetura:** Gradient Boosting Regressor (scikit-learn)
            
            **Features de entrada** (para cada SKU individualmente):
            - Lags 1 a 6 (últimos 6 períodos observados)
            - Média móvel 3 períodos (shift 1)
            - Desvio padrão 3 períodos (shift 1)
            - Média móvel 6 períodos (shift 1)
            - Mínimo e máximo dos últimos 3 períodos
            
            **Aprendizado com o erro histórico:**  
            O modelo é treinado *por SKU* usando todos os pares (features_t → demanda_t+1).
            Isso significa que a IA aprende o padrão específico daquele SKU e pode capturar
            não-linearidades que os modelos estatísticos clássicos ignoram.
            
            **Previsão combinada:**  
            `prev_comb = {peso_ia*100:.0f}% × IA + {(1-peso_ia)*100:.0f}% × melhor método estatístico`
            
            O peso pode ser ajustado no slider da barra lateral.
            """)

        # Seletor de SKU
        skus_ia = sorted(df_ia['sku'].dropna().unique())
        sku_ia = st.selectbox("Selecione o SKU para análise da IA", options=skus_ia, key='sku_ia')

        row_ia_df = df_ia[df_ia['sku'] == sku_ia]
        if len(row_ia_df) == 0:
            st.warning("SKU não encontrado na tabela de IA.")
            st.stop()

        row_ia = row_ia_df.iloc[0].to_dict()
        serie_ia_full = df_base[df_base[col_sku] == sku_ia][col_demanda].reset_index(drop=True).astype(float)

        # Métricas IA
        mi1, mi2, mi3, mi4 = st.columns(4)
        mi1.metric("Prev. Estatística",    f"{row_ia['previsao_estatistica']:.1f}")
        mi2.metric("Prev. IA",             f"{row_ia['previsao_ia']:.1f}" if pd.notna(row_ia.get('previsao_ia')) else "N/A (série curta)")
        mi3.metric("Prev. Combinada",      f"{row_ia['previsao_combinada']:.1f}")
        mi4.metric("WMAPE Melhor Método",  f"{row_ia['wmape_melhor']*100:.1f}%" if pd.notna(row_ia.get('wmape_melhor')) else "—")

        # Treinar IA on-demand para este SKU (para visualização)
        col_vis1, col_vis2 = st.columns([3, 2])

        with col_vis1:
            model_vis = treinar_ia(serie_ia_full)
            if model_vis is not None:
                wmape_v, y_real, y_pred_ia = wmape_ia_insample(model_vis, serie_ia_full)
                offset = len(serie_ia_full) - len(y_real)

                # Comparar IA vs melhor método vs real
                fn_melhor = METODOS[row_ia['melhor_metodo']]

                # Walk-forward in-sample do melhor método
                preds_stat_is = []
                for j in range(len(y_real)):
                    idx_t = offset + j
                    s_train = serie_ia_full.iloc[:idx_t]
                    if len(s_train) >= 2:
                        preds_stat_is.append(fn_melhor(s_train, h=1)[0])
                    else:
                        preds_stat_is.append(serie_ia_full.mean())

                x_plot = list(range(offset, len(serie_ia_full)))

                # Construir labels de mês para o eixo X
                if col_periodo and col_ano:
                    _df_sku_ia = df_base[df_base[col_sku] == sku_ia][[col_ano, col_periodo]].reset_index(drop=True)
                    _meses_ord_ia = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                                     'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
                    _df_sku_ia['_mn'] = _df_sku_ia[col_periodo].astype(str).str.lower().str[:3].map(_meses_ord_ia).fillna(0)
                    _df_sku_ia['_an'] = pd.to_numeric(_df_sku_ia[col_ano], errors='coerce').fillna(0)
                    _df_sku_ia = _df_sku_ia.sort_values(['_an','_mn']).reset_index(drop=True)
                    x_labels_ia_all = [
                        f"{str(r[col_periodo]).lower()[:3]}/{str(int(r[col_ano]))[-2:]}"
                        for _, r in _df_sku_ia.iterrows()
                    ]
                    # Adicionar label "Próx." para o ponto de previsão
                    _prox_m = (int(_df_sku_ia['_mn'].iloc[-1]) % 12) + 1
                    _prox_a = int(_df_sku_ia['_an'].iloc[-1]) + (1 if _df_sku_ia['_mn'].iloc[-1] == 12 else 0)
                    _mn2 = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',
                            7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}
                    x_labels_ia_prox = [f"{_mn2[_prox_m]}/{str(_prox_a)[-2:]}"]
                else:
                    x_labels_ia_all  = [str(i) for i in range(len(serie_ia_full))]
                    x_labels_ia_prox = ["Próx."]

                x_labels_ia_plot = [x_labels_ia_all[i] for i in x_plot] if x_plot else []

                fig_ia = go.Figure()
                fig_ia.add_trace(go.Scatter(
                    x=x_labels_ia_all,
                    y=serie_ia_full.values,
                    mode='lines+markers', name='Histórico Real',
                    line=dict(color='#2c3e50', width=2), marker=dict(size=4)
                ))
                fig_ia.add_trace(go.Scatter(
                    x=x_labels_ia_plot, y=y_pred_ia, mode='lines',
                    name=f'IA (WMAPE in-sample={wmape_v*100:.1f}%)',
                    line=dict(color='#e67e22', width=2, dash='dot')
                ))
                fig_ia.add_trace(go.Scatter(
                    x=x_labels_ia_plot, y=preds_stat_is, mode='lines',
                    name=f'Método {row_ia["melhor_metodo"]}',
                    line=dict(color='#3498db', width=1.5, dash='dashdot')
                ))

                # Próximo período
                fig_ia.add_trace(go.Scatter(
                    x=x_labels_ia_prox, y=[row_ia['previsao_combinada']],
                    mode='markers', name='Prev. Combinada (próx.)',
                    marker=dict(size=16, symbol='star', color='#e74c3c',
                                line=dict(width=1.5, color='white'))
                ))

                fig_ia.update_layout(
                    title=f'IA vs Modelo Estatístico — SKU {sku_ia}',
                    xaxis_title='Período', yaxis_title='Demanda',
                    template='plotly_white', height=400,
                    legend=dict(orientation='h', y=-0.3)
                )
                st.plotly_chart(fig_ia, use_container_width=True)
            else:
                st.warning(f"⚠️ Série do SKU {sku_ia} tem menos de 14 períodos — IA não pode ser treinada.")

        with col_vis2:
            if model_vis is not None:
                # Feature importance
                imp_df = pd.DataFrame({
                    'Feature': FEAT_NAMES,
                    'Importância': model_vis.feature_importances_
                }).sort_values('Importância', ascending=True)

                fig_imp = px.bar(
                    imp_df, x='Importância', y='Feature',
                    orientation='h', title='Importância das Features (IA)',
                    template='plotly_white', color='Importância',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
                st.plotly_chart(fig_imp, use_container_width=True)

            # Diagnóstico textual
            sugest = gerar_sugestao(row_ia, row_ia.get('wmape_melhor'))
            st.markdown("**💡 Diagnóstico e Sugestão:**")
            st.markdown(sugest)

        # Tabela geral
        st.divider()
        st.markdown("### 📊 Resumo IA — Todos os SKUs")

        df_ia_show = df_ia.copy()
        df_ia_show['wmape_melhor'] = (df_ia_show['wmape_melhor'] * 100).round(1).astype(str) + '%'
        df_ia_show['wmape_ia_insample'] = df_ia_show['wmape_ia_insample'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )

        df_ia_renamed = df_ia_show.rename(columns={
            'sku':                  'SKU',
            'classificacao':        'Classificação',
            'tendencia':            'Tendência',
            'media_historica':      'Média Hist.',
            'cv':                   'CV',
            'previsao_estatistica': 'Prev. Estatística',
            'previsao_ia':          'Prev. IA',
            'previsao_combinada':   'Prev. Combinada',
            'melhor_metodo':        'Melhor Método',
            'wmape_melhor':         'WMAPE',
            'wmape_ia_insample':    'WMAPE IA (in-sample)',
        })
        st.dataframe(df_ia_renamed, use_container_width=True, hide_index=True)
        # Export estilizado com valores numéricos reais
        df_ia_exp = df_ia.rename(columns={
            'sku':'SKU','classificacao':'Classificação','tendencia':'Tendência',
            'media_historica':'Média Hist.','cv':'CV',
            'previsao_estatistica':'Prev. Estatística','previsao_ia':'Prev. IA',
            'previsao_combinada':'Prev. Combinada','melhor_metodo':'Melhor Método',
            'wmape_melhor':'WMAPE','wmape_ia_insample':'WMAPE IA (in-sample)',
        }).copy()
        buf_ia = exportar_excel_visual({
            'Resumo_IA_Todos_SKUs': {
                'df': df_ia_exp,
                'col_wmape': 'WMAPE',
                'col_metodo': 'Melhor Método',
                'col_widths': {'SKU':14,'Classificação':16,'Tendência':18,'Média Hist.':14,
                               'CV':10,'Prev. Estatística':18,'Prev. IA':14,
                               'Prev. Combinada':16,'Melhor Método':18,
                               'WMAPE':14,'WMAPE IA (in-sample)':22},
            }
        })
        st.download_button(
            "📥 Exportar Resumo IA — Todos os SKUs (.xlsx)",
            data=buf_ia,
            file_name=f"resumo_ia_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ── TABELA HORIZONTE 3 MESES ─────────────────────────
        st.divider()
        st.markdown("### 📅 Previsão Horizonte 3 Meses — Todos os SKUs")
        st.markdown("Previsão para os **próximos 3 meses** (M+1, M+2, M+3) a partir do último período fechado na base, "
                    "usando o modelo combinado IA + melhor método estatístico.")

        if df_ia is not None and col_periodo and col_ano:
            meses_ord2 = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                          'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
            meses_nome = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',
                          7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}

            # Descobrir último período da base
            df_ult = df_base[[col_ano, col_periodo]].copy()
            df_ult['_mes_num'] = df_ult[col_periodo].astype(str).str.lower().str[:3].map(meses_ord2).fillna(0).astype(int)
            df_ult['_ano_int'] = pd.to_numeric(df_ult[col_ano], errors='coerce').fillna(0).astype(int)
            df_ult = df_ult.sort_values(['_ano_int','_mes_num'])
            ultimo_ano = int(df_ult['_ano_int'].max())
            ultimo_mes = int(df_ult[df_ult['_ano_int'] == ultimo_ano]['_mes_num'].max())

            # Calcular os 3 próximos meses
            proximos = []
            for h_step in range(1, 4):
                m = (ultimo_mes - 1 + h_step) % 12 + 1
                a = ultimo_ano + ((ultimo_mes - 1 + h_step) // 12)
                proximos.append((a, meses_nome[m], f"{meses_nome[m]}/{str(a)[-2:]}"))

            st.info(f"📌 Último período na base: **{meses_nome[ultimo_mes]}/{ultimo_ano}** → "
                    f"Previsões para: **{' · '.join(p[2] for p in proximos)}**")

            # Calcular previsões para cada SKU × 3 horizontes
            rows_h = []
            skus_h = df_base[col_sku].unique()
            prog_h = st.progress(0, text="Calculando horizonte...")

            for idx_h, sku in enumerate(skus_h):
                serie_h = df_base[df_base[col_sku] == sku][col_demanda].reset_index(drop=True).astype(float)

                # IA
                model_h = treinar_ia(serie_h)

                row_h = {'SKU': sku}

                # Buscar melhor método estatístico do backtesting
                if df_bt is not None:
                    rb_h = df_bt[df_bt['sku'] == sku]
                    melhor_h = rb_h.iloc[0]['melhor_metodo'] if len(rb_h) > 0 else 'MA-3'
                else:
                    melhor_h = 'MA-3'

                fn_h = METODOS[melhor_h]

                # Previsões h=1,2,3
                try:
                    preds_stat_h = fn_h(serie_h, h=3)
                except:
                    preds_stat_h = [float(serie_h.mean())] * 3

                # Rolling forecast IA para todos os passos de uma vez
                preds_ia_h = prever_ia_multistep(model_h, serie_h, h=3)

                for step, (ano_h, mes_h, lbl_h) in enumerate(proximos):
                    pred_stat_step = preds_stat_h[step] if step < len(preds_stat_h) else preds_stat_h[-1]
                    pred_ia_step   = preds_ia_h[step] if (preds_ia_h[step] is not None) else pred_stat_step

                    # Peso da IA (do slider)
                    pred_comb_step = peso_ia * pred_ia_step + (1 - peso_ia) * pred_stat_step
                    row_h[f'Prev {lbl_h}'] = round(max(0.0, pred_comb_step), 1)

                row_h['Método Base'] = melhor_h
                rows_h.append(row_h)
                prog_h.progress((idx_h+1)/len(skus_h))

            prog_h.empty()
            df_horizonte = pd.DataFrame(rows_h)

            # Filtro de SKU
            skus_busca = st.text_input("🔍 Filtrar SKU (deixe vazio para ver todos)",
                                        key='horizonte_sku_filter')
            if skus_busca.strip():
                df_horizonte_show = df_horizonte[
                    df_horizonte['SKU'].astype(str).str.contains(skus_busca.strip())
                ]
            else:
                df_horizonte_show = df_horizonte

            st.dataframe(df_horizonte_show, use_container_width=True, hide_index=True)

            # Totais — respeitam o filtro de SKU aplicado
            n_skus_filtro = len(df_horizonte_show)
            n_skus_total  = len(df_horizonte)
            filtro_ativo  = skus_busca.strip() != ""
            label_filtro  = f" ({n_skus_filtro} SKUs filtrados)" if filtro_ativo else f" ({n_skus_total} SKUs)"
            st.markdown(f"**Totais por mês{label_filtro}:**")
            tot_cols = st.columns(3)
            for i, (_, _, lbl_h) in enumerate(proximos):
                col_name = f'Prev {lbl_h}'
                if col_name in df_horizonte_show.columns:
                    tot_cols[i].metric(lbl_h, f"{df_horizonte_show[col_name].sum():,.0f}")

            # Download estilizado
            buf_hz = exportar_excel_visual({
                'Horizonte_3_Meses': {
                    'df': df_horizonte_show,
                    'col_metodo': 'Método Base',
                    'col_widths': {'SKU':14,'Método Base':18,
                                   **{f'Prev {p[2]}':16 for p in proximos}},
                }
            })
            st.download_button(
                "📥 Exportar Horizonte 3 Meses (.xlsx)",
                data=buf_hz,
                file_name=f"horizonte_3meses_{ultimo_mes:02d}_{ultimo_ano}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Rode o pipeline completo para visualizar as previsões de horizonte.")

    # ─────────────────────────────────────────────────────────
    # TAB 5: TOP 10 PIORES WMAPE
    # ─────────────────────────────────────────────────────────
    with tab5:
        st.subheader("📋 Informativo Automático — Top 10 SKUs com Maior WMAPE")

        if df_ia is None or df_bt is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        # ── Filtro de período histórico ──────────────────────────
        _meses_ord_t5 = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                         'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
        _mn_t5 = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',
                  7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}

        f_periodo_label = st.radio(
            "📅 Janela histórica para cálculo do WMAPE",
            options=["3 Meses","6 Meses","12 Meses"],
            index=2, horizontal=True, key='top10_periodo'
        )
        f_n_meses = int(f_periodo_label.split()[0])

        # Recalcular WMAPE por janela — usando base histórica filtrada
        if col_periodo and col_ano:
            # Identificar os N últimos meses disponíveis na base
            _df_per_t5 = df_base[[col_ano, col_periodo]].copy()
            _df_per_t5['_mn'] = _df_per_t5[col_periodo].astype(str).str.lower().str[:3].map(_meses_ord_t5).fillna(0).astype(int)
            _df_per_t5['_an'] = pd.to_numeric(_df_per_t5[col_ano], errors='coerce').fillna(0).astype(int)
            _periodos_sorted = (_df_per_t5[['_an','_mn']].drop_duplicates()
                                .sort_values(['_an','_mn'])
                                .tail(f_n_meses))
            _periodo_keys = set(zip(_periodos_sorted['_an'], _periodos_sorted['_mn']))

            def _filtrar_serie(sku):
                _df_sku = df_base[df_base[col_sku] == sku].copy()
                _df_sku['_mn'] = _df_sku[col_periodo].astype(str).str.lower().str[:3].map(_meses_ord_t5).fillna(0).astype(int)
                _df_sku['_an'] = pd.to_numeric(_df_sku[col_ano], errors='coerce').fillna(0).astype(int)
                _df_sku = _df_sku[_df_sku.apply(lambda r: (r['_an'], r['_mn']) in _periodo_keys, axis=1)]
                return _df_sku.sort_values(['_an','_mn'])[col_demanda].reset_index(drop=True).astype(float)

            # Recalcular WMAPE backtesting para janela filtrada
            _wmape_filtrado = {}
            for _sku in df_ia['sku'].unique():
                _serie_f = _filtrar_serie(_sku)
                if len(_serie_f) >= 4:
                    _row_bt = df_bt[df_bt['sku'] == _sku] if df_bt is not None else pd.DataFrame()
                    _met = _row_bt.iloc[0]['melhor_metodo'] if len(_row_bt) > 0 else 'MA-3'
                    _fn  = METODOS[_met]
                    _w, _, _ = backtest_sku(_serie_f, _fn, n_test=min(3, len(_serie_f)//2))
                    _wmape_filtrado[_sku] = _w if pd.notna(_w) else np.nan
                else:
                    _wmape_filtrado[_sku] = np.nan

            df_ia_t5 = df_ia.copy()
            df_ia_t5['wmape_janela'] = df_ia_t5['sku'].map(_wmape_filtrado)
            df_top10 = (df_ia_t5.sort_values('wmape_janela', ascending=False, na_position='last')
                        .head(10).reset_index(drop=True).copy())
            df_top10['wmape_pct'] = (df_top10['wmape_janela'].fillna(df_top10['wmape_melhor']) * 100).round(1)
        else:
            df_top10 = (df_ia.sort_values('wmape_melhor', ascending=False)
                        .head(10).reset_index(drop=True).copy())
            df_top10['wmape_pct'] = (df_top10['wmape_melhor'] * 100).round(1)

        st.caption(f"Top 10 piores WMAPE calculado sobre os últimos **{f_n_meses} meses** do histórico.")

        # Converter SKU para string — garante eixo categórico no Plotly
        df_top10['sku_str'] = 'SKU ' + df_top10['sku'].astype(str)
        df_top10['sugestao'] = df_top10.apply(
            lambda r: gerar_sugestao(r.to_dict(), r.get('wmape_janela', r['wmape_melhor'])), axis=1
        )

        # Gráfico de barras horizontal
        df_plot_top = df_top10.sort_values('wmape_pct').copy()
        fig_top = px.bar(
            df_plot_top,
            x='wmape_pct', y='sku_str', orientation='h',
            title='Top 10 SKUs — Pior WMAPE',
            color='wmape_pct', color_continuous_scale='RdYlGn_r',
            template='plotly_white',
            labels={'wmape_pct': 'WMAPE (%)', 'sku_str': 'SKU'},
            text='wmape_pct',
            category_orders={'sku_str': df_plot_top['sku_str'].tolist()}
        )
        fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig_top.update_layout(
            showlegend=False, coloraxis_showscale=False, height=420,
            yaxis=dict(type='category')
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.divider()
        st.markdown("### ⚠️ Detalhamento por SKU")

        for i, row in enumerate(df_top10.itertuples()):
            wmape_color = cor_wmape(row.wmape_melhor)
            badge = f'<span style="background:{wmape_color};color:white;padding:3px 10px;border-radius:10px;font-size:13px">{row.wmape_pct:.1f}%</span>'

            with st.expander(
                f"#{i+1}  |  SKU: {row.sku}  |  WMAPE: {row.wmape_pct:.1f}%  |  {row.classificacao}",
                expanded=(i < 3)
            ):
                st.markdown(badge, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Classificação",   row.classificacao)
                c2.metric("Tendência",        row.tendencia.split(' ')[-1] if ' ' in str(row.tendencia) else row.tendencia)
                c3.metric("Prev. Combinada", f"{row.previsao_combinada:.1f}")
                c4.metric("Melhor Método",   row.melhor_metodo)

                # Mini gráfico com eixo X em meses
                _df_mini = df_base[df_base[col_sku] == row.sku].copy()
                if col_periodo and col_ano:
                    _mord = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                             'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
                    _df_mini['_mn'] = _df_mini[col_periodo].astype(str).str.lower().str[:3].map(_mord).fillna(0)
                    _df_mini['_an'] = pd.to_numeric(_df_mini[col_ano], errors='coerce').fillna(0)
                    _df_mini = _df_mini.sort_values(['_an','_mn'])
                    x_mini = [f"{str(r[col_periodo]).lower()[:3]}/{str(int(r[col_ano]))[-2:]}"
                               for _, r in _df_mini.iterrows()]
                    s_mini = _df_mini[col_demanda].reset_index(drop=True).astype(float)
                else:
                    s_mini = _df_mini[col_demanda].reset_index(drop=True).astype(float)
                    x_mini = list(range(len(s_mini)))

                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    x=x_mini, y=s_mini.values, mode='lines+markers',
                    line=dict(color=wmape_color, width=2), marker=dict(size=5)
                ))
                fig_mini.add_hline(y=s_mini.mean(), line_dash='dash',
                                   line_color='gray', annotation_text='Média')
                fig_mini.update_layout(
                    height=180, template='plotly_white',
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title='', yaxis_title='Demanda',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9))
                )
                st.plotly_chart(fig_mini, use_container_width=True, key=f"mini_top10_{i}_{row.sku}")

                # Sugestão IA
                st.markdown(f"💡 **Sugestão IA:** {row.sugestao}")

        # Exportação
        st.divider()
        st.markdown("### 💾 Exportar Resultados")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            buf = exportar_excel(df_bt, df_ia, df_top10[['sku','wmape_pct','melhor_metodo',
                                                           'classificacao','tendencia',
                                                           'previsao_estatistica','previsao_ia',
                                                           'previsao_combinada','sugestao']])
            data_hoje = datetime.date.today().strftime('%d_%m_%Y')
            st.download_button(
                label="📥 Baixar Relatório Excel Completo",
                data=buf,
                file_name=f"forecast_v5_{data_hoje}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )

        with col_exp2:
            # CSV rápido
            csv_buf = df_ia.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Baixar Tabela IA (.csv)",
                data=csv_buf,
                file_name=f"sugestoes_ia_{data_hoje}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
