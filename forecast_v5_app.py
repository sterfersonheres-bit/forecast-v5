#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, warnings, io, datetime, unicodedata, json, re
import urllib.request, urllib.error
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

st.set_page_config(page_title="SONAR — Forecast de Demanda", page_icon="📡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #0f3460; border-radius: 12px; padding: 16px; color: white; }
    .badge-green { background:#00b894; color:white; padding:2px 8px; border-radius:12px; font-size:12px; }
    .badge-red   { background:#d63031; color:white; padding:2px 8px; border-radius:12px; font-size:12px; }
    .badge-yellow{ background:#fdcb6e; color:#2d3436; padding:2px 8px; border-radius:12px; font-size:12px; }
    .stTabs [data-baseweb="tab"] { font-size:15px; font-weight:600; }
    div[data-testid="metric-container"] { background:#f8f9fa; border-radius:10px; padding:10px; border:1px solid #e9ecef; }
    .seg-flag { background: linear-gradient(90deg,#0D9488,#14B8A6); color:white; border-radius:8px; padding:6px 12px; font-size:12px; font-weight:600; margin:4px 0; }
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
    cols_norm = {normalizar(c): c for c in df.columns}
    for palavra in palavras:
        pn = normalizar(palavra)
        for norm, orig in cols_norm.items():
            if pn in norm:
                return orig
    return None

def wmape(actual, forecast):
    a = np.array(actual, dtype=float)
    f = np.array(forecast, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return np.nan
    return np.sum(np.abs(a[mask] - f[mask])) / np.sum(a[mask])

def cor_wmape(val):
    if pd.isna(val): return "#95a5a6"
    if val < 0.20:   return "#00b894"
    if val < 0.40:   return "#f9ca24"
    if val < 0.60:   return "#e17055"
    return "#d63031"

# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════

def carregar_excel(arquivo) -> dict:
    try:
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
            'base_dados':     get_aba('basedados', 'base_dados'),          # ← NOVO: carrega Base_Dados
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
        m = ExponentialSmoothing(serie.astype(float).values, trend='add', seasonal='add', seasonal_periods=sp).fit(optimized=True, disp=False)
        return _safe_positive(m.forecast(h).tolist())
    except:
        return forecast_holt(serie, h)

def forecast_croston(serie, h=1):
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
    s = serie.dropna().reset_index(drop=True)
    n = len(s)
    resultados = []
    for step in range(1, h + 1):
        n_pred = n + step - 1
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
    'Naive': forecast_naive, 'MA-3': forecast_ma3, 'MA-6': forecast_ma6,
    'WMA-3': forecast_wma, 'SES': forecast_ses, 'Holt': forecast_holt,
    'Holt-Winters': forecast_hw, 'Croston': forecast_croston, 'TriM-Heres': forecast_trim_heres,
}

METODOS_DESC = {
    'Naive': 'Usa o último valor observado. Boa referência baseline.',
    'MA-3': 'Média simples dos 3 últimos períodos.',
    'MA-6': 'Média simples dos 6 últimos períodos. Suaviza mais.',
    'WMA-3': 'Média ponderada (maior peso nos períodos recentes).',
    'SES': 'Suavização Exponencial Simples. Alfa otimizado.',
    'Holt': 'Holt duplo. Captura tendência linear.',
    'Holt-Winters': 'Triple ES. Captura tendência e sazonalidade anual.',
    'Croston': 'Ideal para demanda intermitente com muitos zeros.',
    'TriM-Heres': 'Média do mesmo trimestre dos 2 últimos anos. Captura sazonalidade.',
}

# ══════════════════════════════════════════════════════════════
# IMEDIATO — BASE DE CONHECIMENTO E API
# ══════════════════════════════════════════════════════════════

SONAR_KNOWLEDGE = """
Você é o Imediato, o copiloto do SONAR, um assistente especialista em planejamento de demanda
integrado à ferramenta SONAR (Supply & Operations Near-real-time Analytics & Recommendation).
Seu papel é ajudar o planejador a interpretar os dados, entender as regras da ferramenta
e sugerir ações. Responda SEMPRE em português do Brasil, de forma objetiva e prática.
Quando o planejador perguntar sobre um SKU específico, use os DADOS fornecidos no contexto.
Se um dado não estiver disponível, diga isso claramente em vez de inventar.

═══ REGRAS E CONCEITOS DO SONAR ═══

WMAPE (indicador principal): WMAPE = Σ|Real − Previsto| ÷ Σ|Real|. Ponderado pela demanda real.
Faixas de qualidade: <20% Excelente | 20–35% Bom (meta do SONAR) | 35–60% Regular | >60% Crítico.
Sempre prefira a MEDIANA do WMAPE à média — SKUs de baixa demanda geram WMAPE de centenas de %
e distorcem a média.

WALK-FORWARD VALIDATION (backtesting): para cada SKU, o SONAR simula como cada método teria
se saído nos últimos N períodos, treinando só com dados anteriores (sem data leakage). É assim
que o "melhor método" de cada SKU é escolhido — o que minimiza o WMAPE no walk-forward.

OS 9 MÉTODOS ESTATÍSTICOS:
- Naive: usa o último valor. Baseline.
- MA-3 / MA-6: média móvel simples de 3 ou 6 períodos.
- WMA-3: média ponderada, mais peso no recente.
- SES: suavização exponencial simples (alfa otimizado).
- Holt: captura tendência linear.
- Holt-Winters: tendência + sazonalidade anual (exige ≥24 períodos).
- Croston: para demanda intermitente (muitos zeros).
- TriM-Heres: média do mesmo trimestre dos 2 últimos anos; captura sazonalidade sem exigir
  série longa. Recomendado quando ACF lag-12 > 0,4.

CLASSIFICAÇÃO DA DEMANDA (por CV = desvio/média e % de zeros):
- Estável (CV<0,25): previsível → MA-6 ou SES.
- Variável (CV 0,25–0,60) → SES ou Holt.
- Errática (CV>0,60): volátil → revisar dados, SES alpha alto.
- Intermitente (>50% zeros) → Croston.
- Esporádica (>25% zeros + alta variação).

MOTOR DE IA (Gradient Boosting por SKU): treinado individualmente com features de lag (últimos
6 valores, médias móveis 3 e 6, desvio padrão, mín/máx). Usa DETRENDING: treina sobre os resíduos
de uma tendência linear e re-adiciona a tendência na previsão, o que permite extrapolar tendências
de crescimento/queda. Séries com menos de 14 períodos NÃO têm IA. O WMAPE in-sample da IA é
otimista; use o OOS para avaliação honesta.

PREVISÃO COMBINADA: prev_comb = peso_IA × Prev_IA + (1 − peso_IA) × Prev_Estatística.
O peso vem do slider global OU do peso automático calibrado pelo OOS.

IA OUT-OF-SAMPLE (OOS): re-treina a IA excluindo os períodos de teste, gerando WMAPE honesto.
Compara IA vs melhor método estatístico e define o peso automático por SKU:
- IA vence (ratio<0,90) → peso IA = 70%
- desempenho similar (0,90–1,10) → peso IA = 50%
- estatístico vence (>1,10) → peso IA = 0% (100% estatístico)
Após rodar o OOS, todas as tabelas (inclusive Horizonte 3 Meses e Prev. Combinada) são
atualizadas automaticamente com os pesos calibrados. O OOS pode ser segmentado por classe
de material (filtro na sidebar).

SANEAMENTO POR SITUAÇÃO: a aba Base_Dados tem a coluna Situação. Apenas SKUs com situação
"1- NORMAL" são processados; os demais (inativos, cancelados etc.) são descartados antes de
qualquer cálculo. Há uma trava: se nenhum SKU casar com "1- NORMAL", o saneamento é ignorado
para não esvaziar a base.

FILTRO POR CLASSE: a Base_Dados tem Classe de Material e Descrição da Classe. Na sidebar, o
planejador pode segmentar quais classes entram no OOS, com campo de pesquisa.

HORIZONTE 3 MESES: previsão M+1/M+2/M+3 a partir do mês vigente, usando o modelo combinado.
Avança automaticamente no virar do mês.

TOP 10 PIORES WMAPE: dois painéis — 12 meses (problemas estruturais) e 6 meses (problemas
recentes). SKU nos dois = estrutural; só no de 6 = evento pontual. SKUs com <3 registros na
janela são excluídos.

═══ COMO ORIENTAR O PLANEJADOR ═══
- WMAPE crítico (>60%): revisar histórico, checar outliers, verificar quebra estrutural
  (novo cliente, novo projeto), considerar estoque de segurança em vez de depender da previsão.
- Mudança de classificação entre meses: sinal de mudança de comportamento — investigar.
- Para itens estratégicos (alto giro/valor), priorizar revisão manual mesmo com WMAPE bom.
- A ferramenta APOIA o julgamento do planejador; não o substitui.
"""

def _norm_sku(v):
    """Normaliza um SKU para comparação por string ('12345.0' -> '12345')."""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

_MESES_ORD_CTX = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
                  "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

def _detectar_skus_na_pergunta(pergunta, skus_validos, limite=3):
    """Encontra códigos de SKU citados no texto da pergunta.
    skus_validos: conjunto de SKUs normalizados (strings)."""
    if not pergunta:
        return []
    candidatos = re.findall(r"\d{3,}", str(pergunta))
    achados = []
    for c in candidatos:
        if c in skus_validos and c not in achados:
            achados.append(c)
        if len(achados) >= limite:
            break
    return achados

def _montar_contexto_dados(df_ia, df_bt, df_oos, skus_foco=None,
                           df_base=None, col_sku=None, col_periodo=None,
                           col_ano=None, col_demanda=None):
    """Monta um resumo compacto dos dados para enviar ao modelo.
    skus_foco: lista de SKUs (strings normalizadas) a detalhar; vazio = só panorama."""
    linhas = []
    if df_ia is None or len(df_ia) == 0:
        return "Nenhum dado disponível ainda — o planejador precisa rodar o Pipeline Completo."

    # Panorama geral
    n_total = len(df_ia)
    linhas.append(f"PANORAMA GERAL: {n_total} SKUs processados.")
    if 'classificacao' in df_ia.columns:
        dist = df_ia['classificacao'].value_counts().to_dict()
        linhas.append("Distribuição por classificação: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
    if 'melhor_metodo' in df_ia.columns:
        met = df_ia['melhor_metodo'].value_counts().to_dict()
        linhas.append("Melhor método (contagem): " + ", ".join(f"{k}={v}" for k, v in met.items()))
    if 'wmape_melhor' in df_ia.columns:
        wm = df_ia['wmape_melhor'].dropna()
        if len(wm) > 0:
            linhas.append(f"WMAPE mediano: {wm.median()*100:.1f}% | média: {wm.mean()*100:.1f}% | "
                          f"SKUs <35%: {(wm<0.35).sum()} | SKUs >60%: {(wm>0.60).sum()}")

    # Top 10 piores (sempre útil de ter em mãos)
    if 'wmape_melhor' in df_ia.columns:
        piores = df_ia.dropna(subset=['wmape_melhor']).nlargest(10, 'wmape_melhor')
        if len(piores) > 0:
            linhas.append("\nTOP 10 PIORES WMAPE:")
            for _, r in piores.iterrows():
                linhas.append(f"  SKU {r['sku']}: WMAPE={r['wmape_melhor']*100:.0f}%, "
                              f"classe={r.get('classificacao','?')}, método={r.get('melhor_metodo','?')}")

    # OOS resumido
    if df_oos is not None and len(df_oos) > 0:
        val = df_oos.dropna(subset=['wmape_ia_oos', 'wmape_stat_oos']) if 'wmape_ia_oos' in df_oos.columns else df_oos
        if len(val) > 0:
            n_ia = int((val['wmape_ia_oos'] < val['wmape_stat_oos']).sum())
            linhas.append(f"\nOOS: {len(val)} SKUs avaliados. IA vence em {n_ia}, estatístico em {len(val)-n_ia}.")

    # Detalhe dos SKUs em foco (selecionado e/ou citados na pergunta)
    for sku_foco in (skus_foco or []):
        sk = _norm_sku(sku_foco)
        try:
            _ids_norm = df_ia['sku'].apply(_norm_sku)
            row = df_ia[_ids_norm == sk]
            if len(row) > 0:
                r = row.iloc[0]
                linhas.append(f"\n═══ DETALHE DO SKU {sk} ═══")
                for col in ['classificacao','tendencia','sazonalidade','n_periodos','media_historica',
                            'cv','previsao_estatistica','previsao_ia','previsao_combinada',
                            'melhor_metodo','wmape_melhor','wmape_ia_insample','peso_ia_usado']:
                    if col in r.index:
                        val_c = r[col]
                        if isinstance(val_c, float) and 'wmape' in col and pd.notna(val_c):
                            val_c = f"{val_c*100:.1f}%"
                        linhas.append(f"  {col}: {val_c}")
                if df_bt is not None:
                    rb = df_bt[df_bt['sku'].apply(_norm_sku) == sk]
                    if len(rb) > 0:
                        wcols = {c.replace('wmape_',''): rb.iloc[0][c] for c in rb.columns if c.startswith('wmape_') and pd.notna(rb.iloc[0][c])}
                        if wcols:
                            linhas.append("  WMAPE por método (backtesting): " +
                                          ", ".join(f"{k}={v*100:.0f}%" for k, v in wcols.items()))
                # ── Histórico de demanda (últimos 24 períodos) ──
                if df_base is not None and col_sku and col_demanda:
                    _b = df_base[df_base[col_sku].apply(_norm_sku) == sk].copy()
                    if len(_b) > 0:
                        if col_ano and col_periodo:
                            _b["__an_ctx"] = pd.to_numeric(_b[col_ano], errors="coerce").fillna(0).astype(int)
                            _b["__mn_ctx"] = (_b[col_periodo].astype(str).str.lower().str[:3]
                                              .map(_MESES_ORD_CTX).fillna(0).astype(int))
                            _b = _b.sort_values(["__an_ctx", "__mn_ctx"])
                            _rotulos = (_b[col_periodo].astype(str).str[:3] + "/" +
                                        _b[col_ano].astype(str)).tolist()
                        else:
                            _rotulos = [str(i) for i in range(len(_b))]
                        _vals = pd.to_numeric(_b[col_demanda], errors="coerce").fillna(0).tolist()
                        _rotulos, _vals = _rotulos[-24:], _vals[-24:]
                        linhas.append(f"  Histórico de demanda ({len(_vals)} últimos períodos, do mais antigo ao mais recente):")
                        linhas.append("  " + ", ".join(f"{rt}={v:.0f}" for rt, v in zip(_rotulos, _vals)))
            else:
                linhas.append(f"\nSKU {sk} não encontrado nos dados processados.")
        except Exception:
            linhas.append(f"\nNão consegui localizar o SKU {sk}.")

    return "\n".join(linhas)

def chamar_claude_stream(api_key, mensagens, contexto_dados, modelo="claude-sonnet-4-6"):
    """Chama a API da Anthropic em STREAMING via urllib (sem dependência externa).
    É um gerador: vai entregando o texto em pedaços, para uso com st.write_stream.
    mensagens: lista [{'role':'user'/'assistant','content':str}]."""
    system = SONAR_KNOWLEDGE + "\n\n═══ DADOS ATUAIS DO SONAR ═══\n" + contexto_dados
    payload = {
        "model": modelo,
        "max_tokens": 2500,
        "system": system,
        "messages": mensagens,
        "stream": True,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            truncada = False
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue
                t = ev.get("type")
                if t == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
                elif t == "message_delta":
                    if ev.get("delta", {}).get("stop_reason") == "max_tokens":
                        truncada = True
                elif t == "error":
                    yield ("\n\n❌ Erro da API: "
                           + str(ev.get("error", {}).get("message", "desconhecido")))
                    return
            if truncada:
                yield ("\n\n⚠️ *A resposta foi cortada pelo limite de tamanho — "
                       "peça \"continue\" para ver o restante.*")
    except urllib.error.HTTPError as e:
        corpo = e.read().decode("utf-8", errors="ignore")
        if e.code == 401:
            yield "❌ Chave de API inválida ou não autorizada. Verifique a chave inserida."
        elif e.code == 429:
            yield "⏳ Limite de uso da API atingido (rate limit). Aguarde um momento e tente novamente."
        else:
            yield f"❌ Erro da API (HTTP {e.code}): {corpo[:300]}"
    except urllib.error.URLError as e:
        yield f"❌ Falha de conexão com a API: {e.reason}"
    except Exception as e:
        yield f"❌ Erro inesperado: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════

def backtest_sku(serie: pd.Series, fn, n_test: int = 3):
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
# CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════════════

def classificar_demanda(serie: pd.Series) -> str:
    s = serie.dropna()
    if len(s) == 0: return "Indefinida"
    zeros = (s == 0).sum() / len(s)
    mean_v = s.mean()
    cv = s.std() / mean_v if mean_v > 0 else np.inf
    if zeros > 0.50: return "Intermitente"
    if zeros > 0.25 and cv > 0.5: return "Esporádica"
    if cv < 0.25: return "Estável"
    if cv < 0.60: return "Variável"
    return "Errática"

CLASSE_ICONES = {
    "Estável": "🟢", "Variável": "🟡", "Errática": "🔴",
    "Intermitente": "⚪", "Esporádica": "🔵", "Indefinida": "⬛",
}

def detectar_tendencia(serie: pd.Series):
    s = serie.dropna()
    if len(s) < 4: return 0.0, "➡️ Sem dados"
    # Série constante (variância zero) → estável, sem regressão degenerada
    if s.nunique() <= 1 or np.isclose(s.std(), 0.0):
        return 0.0, "➡️ Estável"
    x = np.arange(len(s))
    slope, _, _, p_value, _ = stats.linregress(x, s.values)
    # p_value NaN (degenerado) ou não significativo, ou slope ~0 → estável
    if pd.isna(p_value) or p_value > 0.1 or np.isclose(slope, 0.0):
        return slope, "➡️ Estável"
    return (slope, "📈 Crescente") if slope > 0 else (slope, "📉 Decrescente")

def detectar_sazonalidade(serie: pd.Series) -> str:
    s = serie.dropna()
    if len(s) < 13: return "Dados insuficientes"
    acf12 = s.autocorr(lag=12)
    if pd.notna(acf12) and abs(acf12) > 0.4:
        return f"🔄 Sazonal (ACF lag12={acf12:.2f})"
    return "— Sem sazonalidade detectada"

# ══════════════════════════════════════════════════════════════
# IA — GRADIENT BOOSTING
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
FEAT_NAMES = ([f'lag_{i}' for i in range(1, N_LAGS+1)] +
              ['roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_min_3', 'roll_max_3'])

def _ajustar_tendencia(s_vals):
    """Ajusta tendência linear robusta. Retorna (slope, intercept).
    Para séries < 4 pontos ou constantes, slope=0 (sem detrend)."""
    n = len(s_vals)
    if n < 4:
        return 0.0, float(np.mean(s_vals)) if n > 0 else 0.0
    x = np.arange(n)
    try:
        slope, intercept = np.polyfit(x, s_vals, 1)
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return 0.0, float(np.mean(s_vals))
        return float(slope), float(intercept)
    except Exception:
        return 0.0, float(np.mean(s_vals))

def _features_de(vals_arr):
    """Monta o vetor de features a partir de um array (resíduos), usando os
    últimos pontos. Mesma definição de criar_features, mas pontual."""
    buf = np.array(vals_arr[-(N_LAGS + 3):] if len(vals_arr) >= N_LAGS + 3 else vals_arr, dtype=float)
    lags = [buf[-(i)] for i in range(1, N_LAGS + 1)]
    r3 = np.mean(buf[-3:]) if len(buf) >= 3 else np.mean(buf)
    s3 = np.std(buf[-3:]) if len(buf) >= 3 else 0.0
    r6 = np.mean(buf[-min(6, len(buf)):])
    mn3 = np.min(buf[-3:]) if len(buf) >= 3 else np.min(buf)
    mx3 = np.max(buf[-3:]) if len(buf) >= 3 else np.max(buf)
    return np.array([*lags, r3, s3, r6, mn3, mx3], dtype=float).reshape(1, -1)

def treinar_ia(serie: pd.Series):
    """Treina GBM sobre os RESÍDUOS de uma tendência linear (detrending).
    Isso permite que a IA extrapole tendências — algo que árvores puras não fazem,
    pois saturam no valor máximo visto. Retorna um dicionário-wrapper."""
    s = serie.dropna().reset_index(drop=True)
    if len(s) < 14: return None

    # Detrend: ajusta tendência linear e treina sobre os resíduos
    slope, intercept = _ajustar_tendencia(s.values.astype(float))
    x = np.arange(len(s))
    resid = s.values.astype(float) - (slope * x + intercept)
    resid_s = pd.Series(resid)

    df_f = criar_features(resid_s, N_LAGS)
    if len(df_f) < 6: return None
    X = df_f[FEAT_NAMES].values
    y = df_f['y'].values
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42, min_samples_leaf=2)
    try:
        model.fit(X, y)
        return {'gbm': model, 'slope': slope, 'intercept': intercept, 'n_train': len(s)}
    except Exception:
        return None

def _get_gbm(model):
    """Compatibilidade: extrai o GBM do wrapper (ou retorna o próprio se for modelo cru)."""
    if isinstance(model, dict):
        return model.get('gbm')
    return model

def prever_ia(model, serie: pd.Series):
    if model is None: return None
    gbm = _get_gbm(model)
    if gbm is None: return None
    s = serie.dropna().reset_index(drop=True)
    if len(s) < N_LAGS: return None
    try:
        slope     = model.get('slope', 0.0) if isinstance(model, dict) else 0.0
        intercept = model.get('intercept', 0.0) if isinstance(model, dict) else 0.0
        n_train   = model.get('n_train', len(s)) if isinstance(model, dict) else len(s)

        # Resíduos sobre a tendência ajustada no treino
        x = np.arange(len(s))
        resid = s.values.astype(float) - (slope * x + intercept)
        feats = _features_de(resid)
        resid_pred = float(gbm.predict(feats)[0])

        # Re-adiciona a tendência no índice futuro
        idx_futuro = n_train  # próximo ponto após o treino
        trend_futuro = slope * idx_futuro + intercept
        return float(max(0.0, resid_pred + trend_futuro))
    except Exception:
        return None

def prever_ia_multistep(model, serie: pd.Series, h: int = 3):
    if model is None: return [None] * h
    gbm = _get_gbm(model)
    if gbm is None: return [None] * h
    s = serie.dropna().reset_index(drop=True)
    if len(s) < N_LAGS: return [None] * h
    try:
        slope     = model.get('slope', 0.0) if isinstance(model, dict) else 0.0
        intercept = model.get('intercept', 0.0) if isinstance(model, dict) else 0.0
        n_train   = model.get('n_train', len(s)) if isinstance(model, dict) else len(s)
    except Exception:
        slope, intercept, n_train = 0.0, 0.0, len(s)

    # Buffer de resíduos (não de valores) — o GBM opera no domínio dos resíduos
    x = np.arange(len(s))
    resid_buf = list(s.values.astype(float) - (slope * x + intercept))
    preds = []
    for step in range(h):
        try:
            feats = _features_de(resid_buf)
            resid_pred = float(gbm.predict(feats)[0])
            idx_futuro = n_train + step
            trend_futuro = slope * idx_futuro + intercept
            valor = float(max(0.0, resid_pred + trend_futuro))
            preds.append(valor)
            resid_buf.append(resid_pred)   # alimenta o próximo passo com o RESÍDUO
        except Exception:
            preds.append(None)
    return preds

def wmape_ia_insample(model, serie: pd.Series):
    """WMAPE in-sample no domínio dos VALORES (reconstruídos = resíduo previsto + tendência)."""
    gbm = _get_gbm(model)
    if gbm is None: return np.nan, np.array([]), np.array([])
    s = serie.dropna().reset_index(drop=True)
    slope     = model.get('slope', 0.0) if isinstance(model, dict) else 0.0
    intercept = model.get('intercept', 0.0) if isinstance(model, dict) else 0.0

    x = np.arange(len(s))
    trend = slope * x + intercept
    resid_s = pd.Series(s.values.astype(float) - trend)

    df_f = criar_features(resid_s, N_LAGS)
    if len(df_f) < 4: return np.nan, np.array([]), np.array([])
    X = df_f[FEAT_NAMES].values
    resid_pred = gbm.predict(X)

    # Reconstruir valores: resíduo previsto + tendência no índice correspondente
    offset = len(resid_s) - len(df_f)
    idxs = np.arange(offset, len(resid_s))
    trend_vals = slope * idxs + intercept
    y_pred = np.maximum(0, resid_pred + trend_vals)
    y_real = s.values.astype(float)[offset:]
    return wmape(y_real, y_pred), y_real, y_pred

# ══════════════════════════════════════════════════════════════
# EXPORTAÇÃO EXCEL
# ══════════════════════════════════════════════════════════════

_XL = {
    'navy': '0F2B4F', 'teal': '0D9488', 'teal_lt': '14B8A6', 'amber': 'F59E0B',
    'white': 'FFFFFF', 'off_white': 'F8FAFC', 'slate': '64748B', 'green': '22C55E',
    'red': 'EF4444', 'yellow': 'EAB308', 'light_green': 'C8E6C9', 'light_yellow': 'FFF9C4',
    'light_red': 'FFCDD2', 'light_teal': 'E0F2F1', 'light_amber': 'FEF9EC',
}

def _xl_fmt(wb, **kw):
    return wb.add_format(kw)

def _escrever_aba_estilizada(writer, df, nome_aba, col_wmape=None, col_wmape_escala=1.0,
                              col_metodo=None, col_classe=None, freeze=True, col_widths=None):
    if df is None or len(df) == 0: return
    wb = writer.book
    hdr = _xl_fmt(wb, bold=True, bg_color=_XL['navy'], font_color=_XL['white'],
                  border=1, border_color='CCCCCC', align='center', valign='vcenter', font_size=10, font_name='Calibri')
    row_par   = _xl_fmt(wb, bg_color=_XL['white'],     border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', valign='vcenter')
    row_impar = _xl_fmt(wb, bg_color=_XL['off_white'], border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', valign='vcenter')
    fmt_excelente = _xl_fmt(wb, bg_color=_XL['light_green'],  border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', bold=True, font_color='1B5E20', num_format='0.0%')
    fmt_bom       = _xl_fmt(wb, bg_color=_XL['light_teal'],   border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', bold=True, font_color='004D40', num_format='0.0%')
    fmt_regular   = _xl_fmt(wb, bg_color=_XL['light_yellow'], border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', bold=True, font_color='E65100', num_format='0.0%')
    fmt_critico   = _xl_fmt(wb, bg_color=_XL['light_red'],    border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', bold=True, font_color='B71C1C', num_format='0.0%')
    fmt_metodo    = _xl_fmt(wb, bg_color=_XL['light_teal'],   border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', bold=True, font_color=_XL['teal'])
    fmt_num       = _xl_fmt(wb, bg_color=_XL['white'],        border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', num_format='#,##0.00', valign='vcenter')
    fmt_num_impar = _xl_fmt(wb, bg_color=_XL['off_white'],    border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', num_format='#,##0.00', valign='vcenter')
    fmt_pct       = _xl_fmt(wb, bg_color=_XL['white'],        border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', num_format='0.0%', valign='vcenter')
    fmt_pct_impar = _xl_fmt(wb, bg_color=_XL['off_white'],    border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', num_format='0.0%', valign='vcenter')
    titulo_fmt = _xl_fmt(wb, bold=True, bg_color=_XL['teal'], font_color=_XL['white'], font_size=11, font_name='Calibri', valign='vcenter', align='left')
    leg_hdr = _xl_fmt(wb, bold=True, bg_color=_XL['slate'], font_color=_XL['white'], font_size=8, font_name='Calibri')

    cols = list(df.columns)
    n_cols = len(cols)
    if col_wmape is None:
        _wmape_cols = []
    elif isinstance(col_wmape, (list, tuple, set)):
        _wmape_cols = list(col_wmape)
    else:
        _wmape_cols = [col_wmape]
    wmape_idx_set = {cols.index(c) for c in _wmape_cols if c in cols}
    metodo_idx = cols.index(col_metodo) if col_metodo and col_metodo in cols else None
    sku_idx_list = [i for i, c in enumerate(cols) if c.upper() in ('SKU','COD. MATERIAL','CODIGO','MATERIAL','ITEM') or 'sku' in c.lower()]

    df.to_excel(writer, sheet_name=nome_aba, index=False, startrow=2, header=False)
    ws = writer.sheets[nome_aba]
    ws.set_row(0, 22)
    ws.merge_range(0, 0, 0, max(n_cols-1, 0), f'  {nome_aba.replace("_"," ").upper()}', titulo_fmt)
    ws.set_row(1, 20)
    for ci, col in enumerate(cols):
        ws.write(1, ci, str(col), hdr)

    for ri, row in enumerate(df.itertuples(index=False)):
        ws.set_row(ri + 2, 16)
        bg_par = ri % 2 == 0
        for ci, val in enumerate(row):
            col_name = cols[ci]
            is_wmape_col  = (ci in wmape_idx_set)
            is_metodo_col = (ci == metodo_idx)
            is_num = isinstance(val, (int, float)) and not pd.isna(val)

            if ci in sku_idx_list:
                fmt_sku = _xl_fmt(wb, bg_color=_XL['white'] if bg_par else _XL['off_white'],
                                  border=1, border_color='E2E8F0', font_size=9, font_name='Calibri', valign='vcenter', num_format='@')
                ws.write_string(ri+2, ci, str(int(val)) if isinstance(val, float) and val == int(val) else str(val) if val is not None else '—', fmt_sku)
            elif is_wmape_col:
                raw = val
                if isinstance(raw, str):
                    _s = raw.replace('%', '').replace(',', '.').strip()
                    try:
                        raw = (float(_s) / 100) if raw.strip().endswith('%') else float(_s)
                    except Exception:
                        raw = np.nan
                if not isinstance(raw, (int, float)) or pd.isna(raw):
                    ws.write(ri+2, ci, '—', row_par if bg_par else row_impar)
                else:
                    v_pct = raw * col_wmape_escala if col_wmape_escala != 1.0 else raw
                    if v_pct < 0.20:   fmt_w = fmt_excelente
                    elif v_pct < 0.35: fmt_w = fmt_bom
                    elif v_pct < 0.60: fmt_w = fmt_regular
                    else:              fmt_w = fmt_critico
                    ws.write_number(ri+2, ci, float(v_pct), fmt_w)
            elif is_metodo_col:
                ws.write(ri+2, ci, str(val) if val is not None else '—', fmt_metodo)
            elif is_num and ('wmape' in col_name.lower() or 'erro' in col_name.lower() or '%' in col_name):
                ws.write(ri+2, ci, val, fmt_pct if bg_par else fmt_pct_impar)
            elif is_num:
                ws.write(ri+2, ci, val, fmt_num if bg_par else fmt_num_impar)
            else:
                fmt_use = row_par if bg_par else row_impar
                ws.write(ri+2, ci, str(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else '—', fmt_use)

    default_w = 16
    col_w_map = col_widths or {}
    for ci, col in enumerate(cols):
        ws.set_column(ci, ci, col_w_map.get(col, default_w))

    if freeze:
        ws.freeze_panes(2, 0)

    leg_row = len(df) + 4
    ws.write(leg_row, 0, 'Legenda WMAPE', leg_hdr)
    for j, (lbl, bg, fc) in enumerate([
        ('< 20%  — Excelente', _XL['light_green'],  '1B5E20'),
        ('20–35% — Bom ⭐',    _XL['light_teal'],   '004D40'),
        ('35–60% — Regular',   _XL['light_yellow'], 'E65100'),
        ('> 60%  — Crítico',   _XL['light_red'],    'B71C1C'),
    ], 1):
        ws.write(leg_row + j, 0, lbl, _xl_fmt(wb, bg_color=bg, font_color=fc,
                                               font_size=8, font_name='Calibri', bold=True,
                                               border=1, border_color='CCCCCC'))

def exportar_excel_visual(abas: dict, filename_prefix: str = 'forecast') -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        for nome_aba, cfg in abas.items():
            _escrever_aba_estilizada(writer, df=cfg.get('df'), nome_aba=nome_aba,
                col_wmape=cfg.get('col_wmape'), col_wmape_escala=cfg.get('col_wmape_escala', 1.0),
                col_metodo=cfg.get('col_metodo'), col_widths=cfg.get('col_widths'))
        wb = writer.book
        ls = wb.add_worksheet('Legenda_Geral')
        ls.set_column(0, 0, 35)
        hf = wb.add_format({'bold':True,'bg_color':_XL['navy'],'font_color':_XL['white'],'font_size':10,'font_name':'Calibri','border':1})
        ls.write(0, 0, 'Legenda Geral do Arquivo', hf)
        for i, (txt, bg, fc) in enumerate([
            ('WMAPE < 20%  — Excelente (verde escuro)', _XL['light_green'],  '1B5E20'),
            ('WMAPE 20–35% — Bom ⭐ (teal)',            _XL['light_teal'],   '004D40'),
            ('WMAPE 35–60% — Regular (amarelo)',         _XL['light_yellow'], 'E65100'),
            ('WMAPE > 60%  — Crítico (vermelho)',        _XL['light_red'],    'B71C1C'),
            ('Método Base  — Melhor método (teal)',      _XL['light_teal'],   _XL['teal']),
        ], 1):
            ls.write(i, 0, txt, wb.add_format({'bg_color':bg,'font_color':fc,'font_size':9,'font_name':'Calibri','bold':True,'border':1}))
    buf.seek(0)
    return buf

@st.cache_data(show_spinner=False)
def calcular_wmape_janela(_df_base_hash, df_base_ref, col_sku_r, col_periodo_r,
                          col_ano_r, col_demanda_r, df_ia_ref, df_bt_ref, n_meses: int):
    meses_ord_c = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
                   "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
    _df_per = df_base_ref[[col_ano_r, col_periodo_r]].copy()
    _df_per["_mn"] = _df_per[col_periodo_r].astype(str).str.lower().str[:3].map(meses_ord_c).fillna(0).astype(int)
    _df_per["_an"] = pd.to_numeric(_df_per[col_ano_r], errors="coerce").fillna(0).astype(int)
    _periodos_sorted = (_df_per[["_an","_mn"]].drop_duplicates().sort_values(["_an","_mn"]).tail(n_meses))
    _periodo_keys = set(zip(_periodos_sorted["_an"], _periodos_sorted["_mn"]))
    _df_full = df_base_ref.copy()
    _df_full["__mn"] = _df_full[col_periodo_r].astype(str).str.lower().str[:3].map(meses_ord_c).fillna(0).astype(int)
    _df_full["__an"] = pd.to_numeric(_df_full[col_ano_r], errors="coerce").fillna(0).astype(int)
    _df_full["__key"] = list(zip(_df_full["__an"], _df_full["__mn"]))
    _df_full = _df_full[_df_full["__key"].isin(_periodo_keys)]
    MIN_REG = 3
    wmape_map, nreg_map = {}, {}
    _df_full_sku = df_base_ref.copy()
    _df_full_sku["__mn2"] = _df_full_sku[col_periodo_r].astype(str).str.lower().str[:3].map(meses_ord_c).fillna(0).astype(int)
    _df_full_sku["__an2"] = pd.to_numeric(_df_full_sku[col_ano_r], errors="coerce").fillna(0).astype(int)

    for _sku in df_ia_ref["sku"].unique():
        _s = (_df_full[_df_full[col_sku_r] == _sku].sort_values(["__an","__mn"])[col_demanda_r].reset_index(drop=True).astype(float))
        n = len(_s)
        nreg_map[_sku] = n
        if n < MIN_REG:
            wmape_map[_sku] = np.nan
            continue
        _rb = df_bt_ref[df_bt_ref["sku"] == _sku] if df_bt_ref is not None else pd.DataFrame()
        _met = _rb.iloc[0]["melhor_metodo"] if len(_rb) > 0 else "MA-3"
        _fn = METODOS[_met]
        if n >= 5:
            _nt = min(2, max(1, n - 2))
            _w, _, _ = backtest_sku(_s, _fn, n_test=_nt)
            wmape_map[_sku] = _w if pd.notna(_w) else np.nan
        else:
            _s_full = (_df_full_sku[_df_full_sku[col_sku_r] == _sku].sort_values(["__an2","__mn2"])[col_demanda_r].reset_index(drop=True).astype(float))
            _n_full = len(_s_full)
            _n_antes = _n_full - n
            _s_treino = _s_full.iloc[:_n_antes] if _n_antes >= 3 else _s_full
            try:
                _pred_fixo = _fn(_s_treino, h=1)[0]
                _actual = _s.values
                _denom = np.sum(np.abs(_actual))
                if _denom > 0:
                    wmape_map[_sku] = float(np.sum(np.abs(_actual - _pred_fixo)) / _denom)
                else:
                    wmape_map[_sku] = np.nan
            except Exception:
                wmape_map[_sku] = np.nan

    return wmape_map, nreg_map, MIN_REG, _periodo_keys

# ══════════════════════════════════════════════════════════════
# IA OUT-OF-SAMPLE — AGORA COM FILTRO POR CLASSE (skus_alvo)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def calcular_wmape_ia_oos(_base_hash, df_base_ref, col_sku_r, col_periodo_r,
                          col_ano_r, col_demanda_r, df_ia_ref, n_test: int = 2,
                          skus_alvo: frozenset = None):
    """
    Walk-forward out-of-sample da IA.
    skus_alvo: frozenset de SKUs para avaliar (derivado do filtro de classe).
               None = todos os SKUs elegíveis da base.
    """
    meses_ord_oos = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
                     "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

    _df = df_base_ref.copy()
    _df["__mn"] = _df[col_periodo_r].astype(str).str.lower().str[:3].map(meses_ord_oos).fillna(0).astype(int)
    _df["__an"] = pd.to_numeric(_df[col_ano_r], errors="coerce").fillna(0).astype(int)

    wmape_oos_map = {}
    detalhe_map   = {}
    MIN_PERIODOS  = 14 + n_test

    # ── Filtrar SKUs conforme segmentação por classe ─────────
    if skus_alvo is not None and len(skus_alvo) > 0:
        _skus_alvo_int = {int(s) for s in skus_alvo}
        _skus_para_rodar = [
            s for s in df_ia_ref["sku"].unique()
            if int(s) in _skus_alvo_int
        ]
    else:
        # Sem filtro: roda todos os SKUs elegíveis
        _skus_para_rodar = list(df_ia_ref["sku"].unique())

    for sku in _skus_para_rodar:
        serie = (_df[_df[col_sku_r] == sku].sort_values(["__an","__mn"])[col_demanda_r]
                 .reset_index(drop=True).astype(float))
        n = len(serie)

        if n < MIN_PERIODOS:
            wmape_oos_map[sku] = np.nan
            detalhe_map[sku]   = {"status": f"Série curta ({n} períodos — mínimo {MIN_PERIODOS})"}
            continue

        actuals, preds_ia, preds_stat = [], [], []
        _rb  = df_ia_ref[df_ia_ref["sku"] == sku]
        _met = _rb.iloc[0]["melhor_metodo"] if len(_rb) > 0 else "MA-3"
        _fn  = METODOS[_met]

        for step in range(n_test, 0, -1):
            treino = serie.iloc[:n - step]
            _modelo_oos = treinar_ia(treino)
            _pred_ia    = prever_ia(_modelo_oos, treino) if _modelo_oos else None
            try:
                _pred_stat = float(_fn(treino, h=1)[0])
            except Exception:
                _pred_stat = float(treino.mean())
            actuals.append(float(serie.iloc[n - step]))
            preds_ia.append(float(_pred_ia) if _pred_ia is not None else _pred_stat)
            preds_stat.append(_pred_stat)

        w_oos  = wmape(actuals, preds_ia)
        w_stat = wmape(actuals, preds_stat)

        wmape_oos_map[sku] = w_oos if pd.notna(w_oos) else np.nan
        detalhe_map[sku] = {
            "status": "OK",
            "wmape_ia_oos": w_oos, "wmape_stat_oos": w_stat, "n_test": n_test,
            "actuals": actuals, "preds_ia": preds_ia, "preds_stat": preds_stat,
            "met_stat": _met,
            "recomendacao": (
                "✅ IA é mais confiável — aumente o peso no slider"
                if pd.notna(w_oos) and pd.notna(w_stat) and w_oos < w_stat
                else "⚠️ Método estatístico é mais confiável — reduza o peso da IA"
                if pd.notna(w_oos) and pd.notna(w_stat) and w_oos > w_stat * 1.10
                else "➡️ Desempenho similar — peso 50/50 é adequado"
            ),
        }

    return wmape_oos_map, detalhe_map

def exportar_excel(df_bt, df_ia, df_top10):
    abas = {
        'Backtesting_Completo': {'df': df_bt,    'col_wmape': 'melhor_wmape', 'col_metodo': 'melhor_metodo'},
        'Sugestoes_IA':         {'df': df_ia,    'col_wmape': 'wmape_melhor', 'col_metodo': 'melhor_metodo'},
        'Top10_Piores_WMAPE':   {'df': df_top10, 'col_wmape': 'wmape_pct', 'col_wmape_escala': 100.0, 'col_metodo': 'melhor_metodo'},
    }
    return exportar_excel_visual(abas)

# ══════════════════════════════════════════════════════════════
# SUGESTÃO AUTOMÁTICA
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
        st.markdown("## 📡 SONAR")
        st.markdown("*Supply & Operations Near-real-time Analytics & Recommendation*")
        st.divider()

        uploaded_file = st.file_uploader(
            "📂 Faça upload do arquivo Excel", type=["xlsx"],
            help="Selecione o arquivo .xlsx com as abas: Base_Limpa, Base_Dados, etc."
        )

        if uploaded_file is None:
            st.info("👆 Faça o upload do arquivo Excel para começar.")
            st.stop()

        arquivo = uploaded_file

        st.divider()
        st.markdown("**⚙️ Configurações**")
        _n_auto_side = len(st.session_state.get('peso_por_sku', {}))
        if _n_auto_side > 0:
            st.caption(f"🎯 Peso automático ativo: {_n_auto_side} SKUs calibrados pelo OOS")
        else:
            st.caption("Slider global aplicado a todos os SKUs")

        # ── Informativo de saneamento (situação ≠ "1- NORMAL") ────
        _n_desc = st.session_state.get('n_descartados_situacao', 0)
        _alerta_sit = st.session_state.get('alerta_situacao')
        if _alerta_sit:
            st.warning(_alerta_sit)
        elif _n_desc > 0:
            st.warning(f"⚠️ **{_n_desc} SKUs descartados** — situação ≠ \"1- NORMAL\"")

        n_test = st.slider("Períodos de backtesting (walk-forward)", 2, 6, 3)
        peso_ia = st.slider("Peso da IA na previsão combinada (%)", 0, 100, 50,
                            help="Fallback global — SKUs com OOS calibrado usam peso automático") / 100
        st.divider()
        rodar = st.button("🚀 Rodar Pipeline Completo", type="primary", use_container_width=True)
        if st.button("🗑️ Limpar Cache", use_container_width=True):
            st.cache_data.clear()
            for k in ['df_backtest', 'df_ia', 'df_oos', 'peso_por_sku', '_det_oos',
                      'n_descartados_situacao', 'df_sku_meta', 'classes_disponiveis',
                      'alerta_situacao', 'classes_selecionadas_oos', '_oos_segmentacao']:
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()
        st.markdown("**🔬 Análise Avançada**")
        n_test_oos = st.slider("Períodos de teste out-of-sample", 1, 3, 2)

        # ── FILTRO DE CLASSE PARA IA OOS ─────────────────────
        _classes_disp = st.session_state.get('classes_disponiveis', [])
        if _classes_disp:
            st.markdown("**🏷️ Segmentação por Classe**")
            _busca_classe = st.text_input(
                "🔍 Pesquisar classe", key='busca_classe_oos',
                placeholder="Ex: MOB, FERRAGEM..."
            )
            _classes_filtradas = (
                [c for c in _classes_disp if _busca_classe.lower() in c.lower()]
                if _busca_classe.strip() else _classes_disp
            )
            _classes_sel = st.multiselect(
                "Selecionar classes para OOS",
                options=_classes_filtradas,
                default=[c for c in st.session_state.get('classes_selecionadas_oos', []) if c in _classes_filtradas],
                key='classes_sel_oos',
                help="Deixe vazio para avaliar todos os SKUs elegíveis"
            )
            st.session_state['classes_selecionadas_oos'] = _classes_sel

            # Flag de segmentação ativa
            _df_meta_sb = st.session_state.get('df_sku_meta', pd.DataFrame())
            if _classes_sel and not _df_meta_sb.empty and 'classe' in _df_meta_sb.columns:
                _cls_codes_sb = [c.split(' — ')[0].strip() for c in _classes_sel]
                _n_skus_sel_sb = _df_meta_sb[_df_meta_sb['classe'].astype(str).isin(_cls_codes_sb)]['sku'].nunique()
                st.markdown(
                    f'<div class="seg-flag">🎯 Segmentação ativa<br/>'
                    f'{len(_classes_sel)} classe(s) · {_n_skus_sel_sb} SKUs</div>',
                    unsafe_allow_html=True
                )
            elif not _classes_sel:
                st.caption("Sem segmentação — todos os SKUs elegíveis")
        else:
            st.caption("ℹ️ Suba o arquivo para habilitar o filtro de classe")

        rodar_oos = st.button(
            "🔬 IA Out-of-Sample", use_container_width=True,
            help="Avalia a IA em dados que ela nunca viu. "
                 "Respeita o filtro de classe selecionado acima."
        )

    # ─── CABEÇALHO ─────────────────────────────────────────────
    st.title("📡 SONAR")
    st.caption("Supply & Operations Near-real-time Analytics & Recommendation  |  IA + Backtesting + Análise por SKU")

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
        st.error("Aba com histórico de demanda não encontrada.")
        with st.expander("Abas encontradas no arquivo"):
            st.write(dados.get('abas', []))
        st.stop()

    col_sku     = encontrar_coluna(df_base, 'sku', 'material', 'codigo', 'cod', 'produto', 'item')
    col_periodo = encontrar_coluna(df_base, 'periodo', 'mes', 'data', 'competencia', 'referencia')
    col_ano     = encontrar_coluna(df_base, 'ano', 'year')
    col_demanda = encontrar_coluna(df_base, 'demanda', 'consumo', 'quantidade', 'qtd', 'qtde', 'realizado')

    if not col_sku or not col_demanda:
        st.error(f"Colunas de SKU e/ou demanda não encontradas. Colunas: {list(df_base.columns)}")
        st.stop()

    df_base[col_demanda] = pd.to_numeric(df_base[col_demanda], errors='coerce').fillna(0)

    # ══════════════════════════════════════════════════════════
    # PROCESSAR BASE_DADOS: situação (saneamento) + meta de classe
    # ══════════════════════════════════════════════════════════
    df_dados_raw = dados.get('base_dados')

    if df_dados_raw is not None:
        _col_sku_d  = encontrar_coluna(df_dados_raw, 'sku', 'material', 'codigo', 'cod')
        _col_sit    = encontrar_coluna(df_dados_raw, 'situacao', 'situacao', 'status')
        _col_cls    = encontrar_coluna(df_dados_raw, 'classedematerial', 'classematerial', 'classe')
        _col_desc   = encontrar_coluna(df_dados_raw, 'descricaodaclasse', 'descricaoclasse', 'descclasse', 'descricao')

        if _col_sku_d:
            cols_meta = [c for c in [_col_sku_d, _col_sit, _col_cls, _col_desc] if c]
            df_sku_meta = df_dados_raw[cols_meta].drop_duplicates(subset=[_col_sku_d]).copy()
            rename_map = {_col_sku_d: 'sku'}
            if _col_sit:  rename_map[_col_sit]  = 'situacao'
            if _col_cls:  rename_map[_col_cls]  = 'classe'
            if _col_desc: rename_map[_col_desc] = 'descricao_classe'
            df_sku_meta = df_sku_meta.rename(columns=rename_map)

            # ── 1. SANEAMENTO: manter apenas SKUs com situação = "1- NORMAL" ──
            # Comparação robusta: ignora caixa, acentos e espaços ("1- NORMAL",
            # "1 - NORMAL", "1-NORMAL" e variações de capitalização são aceitas).
            if 'situacao' in df_sku_meta.columns:
                _sit_norm = (
                    df_sku_meta['situacao'].astype(str)
                    .str.strip().str.lower()
                    .str.replace(' ', '', regex=False)
                )
                _skus_normais = set(
                    df_sku_meta[_sit_norm == '1-normal']['sku'].unique()
                )
                _n_antes = df_base[col_sku].nunique()

                # TRAVA DE SEGURANÇA: se o filtro descartaria TODA (ou quase toda)
                # a base, o formato da coluna provavelmente diverge de "1- NORMAL".
                # Nesse caso, NÃO filtra e avisa — evita esvaziar a base por engano.
                _skus_base = set(df_base[col_sku].unique())
                _n_manteria = len(_skus_normais & _skus_base)

                if _n_manteria == 0 and _n_antes > 0:
                    st.session_state['n_descartados_situacao'] = 0
                    st.session_state['alerta_situacao'] = (
                        "⚠️ Nenhum SKU tem situação \"1- NORMAL\". O saneamento foi "
                        "IGNORADO para não esvaziar a base. Verifique os valores da "
                        f"coluna Situação. Valores encontrados: "
                        f"{sorted(df_sku_meta['situacao'].dropna().astype(str).unique())[:8]}"
                    )
                else:
                    df_base   = df_base[df_base[col_sku].isin(_skus_normais)].copy()
                    _n_depois = df_base[col_sku].nunique()
                    st.session_state['n_descartados_situacao'] = _n_antes - _n_depois
                    st.session_state['alerta_situacao'] = None
            else:
                st.session_state['n_descartados_situacao'] = 0
                st.session_state['alerta_situacao'] = None

            # ── 2. CLASSES: construir lista para o filtro OOS ─────────
            if 'classe' in df_sku_meta.columns and 'descricao_classe' in df_sku_meta.columns:
                _classes_opts = (
                    df_sku_meta[['classe', 'descricao_classe']]
                    .dropna()
                    .drop_duplicates()
                    .sort_values('classe')
                    .apply(lambda r: f"{r['classe']} — {r['descricao_classe']}", axis=1)
                    .tolist()
                )
                st.session_state['classes_disponiveis'] = _classes_opts
                st.session_state['df_sku_meta'] = df_sku_meta
            elif 'classe' in df_sku_meta.columns:
                _classes_opts = sorted(df_sku_meta['classe'].dropna().unique().tolist())
                st.session_state['classes_disponiveis'] = [str(c) for c in _classes_opts]
                st.session_state['df_sku_meta'] = df_sku_meta

    # ─── PERÍODO COMBINADO ─────────────────────────────────────
    if col_ano and col_periodo:
        df_base['_periodo_combined'] = df_base[col_ano].astype(str) + '_' + df_base[col_periodo].astype(str)
        col_periodo_count = '_periodo_combined'
    elif col_periodo:
        col_periodo_count = col_periodo
    else:
        col_periodo_count = None

    wmape_original = None
    if df_aval is not None:
        col_wmape_orig = encontrar_coluna(df_aval, 'wmape')
        col_sku_aval   = encontrar_coluna(df_aval, 'sku', 'material', 'codigo')
        if col_wmape_orig and col_sku_aval:
            wmape_original = df_aval.groupby(col_sku_aval)[col_wmape_orig].mean().to_dict()

    # ─── MÉTRICAS RÁPIDAS ──────────────────────────────────────
    n_skus  = df_base[col_sku].nunique()
    n_reg   = len(df_base)
    n_per   = df_base[col_periodo_count].nunique() if col_periodo_count else "—"
    med_dem = df_base[col_demanda].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏷️ SKUs únicos",        n_skus)
    c2.metric("📋 Registros",          f"{n_reg:,}")
    c3.metric("📅 Períodos",            n_per)
    c4.metric("📊 Demanda média",       f"{med_dem:.1f}")
    c5.metric("🔧 Métodos disponíveis", len(METODOS))
    st.divider()

    # ─── ESTADO DA SESSÃO ──────────────────────────────────────
    for key in ['df_backtest', 'df_ia', 'df_oos', 'peso_por_sku']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'peso_por_sku' else {}

    # ─── RODAR PIPELINE ────────────────────────────────────────
    if rodar:
        with st.status("🔄 Executando pipeline...", expanded=True) as status:
            st.write("📊 Etapa 1/3 — Backtesting walk-forward...")
            df_bt = rodar_backtesting(df_base, col_sku, col_demanda, n_test)
            st.session_state['df_backtest'] = df_bt

            st.write("🤖 Etapa 2/3 — Treinando modelos de IA por SKU...")
            ia_rows = []
            skus_all = df_base[col_sku].unique()
            barra2 = st.progress(0, text="Treinando IA...")

            for idx, sku in enumerate(skus_all):
                serie = df_base[df_base[col_sku] == sku][col_demanda].reset_index(drop=True)
                model_ia  = treinar_ia(serie)
                pred_ia   = prever_ia(model_ia, serie)
                wmape_ia  = np.nan
                row_bt = df_bt[df_bt['sku'] == sku]
                if len(row_bt) > 0:
                    melhor   = row_bt.iloc[0]['melhor_metodo']
                    melhor_w = row_bt.iloc[0]['melhor_wmape']
                else:
                    melhor, melhor_w = 'MA-3', np.nan
                fn_melhor = METODOS[melhor]
                pred_stat = fn_melhor(serie, h=1)[0]
                _pesos_oos = st.session_state.get('peso_por_sku', {})
                _peso_sku  = _pesos_oos.get(sku, _pesos_oos.get(float(sku), None))
                _peso_efetivo = _peso_sku if _peso_sku is not None else peso_ia
                if pred_ia is not None:
                    pred_comb = _peso_efetivo * pred_ia + (1 - _peso_efetivo) * pred_stat
                    if model_ia:
                        wmape_ia, _, _ = wmape_ia_insample(model_ia, serie)
                else:
                    pred_comb = pred_stat
                    _peso_efetivo = 0.0

                classe  = classificar_demanda(serie)
                _, tend = detectar_tendencia(serie)
                sazon   = detectar_sazonalidade(serie)
                row_dict = {
                    'sku': sku, 'classificacao': classe, 'tendencia': tend,
                    'sazonalidade': sazon, 'n_periodos': len(serie),
                    'media_historica': round(serie.mean(), 2),
                    'cv': round(serie.std() / serie.mean(), 3) if serie.mean() > 0 else np.nan,
                    'previsao_estatistica': round(pred_stat, 2),
                    'previsao_ia': round(pred_ia, 2) if pred_ia is not None else np.nan,
                    'previsao_combinada': round(pred_comb, 2),
                    'melhor_metodo': melhor, 'wmape_melhor': melhor_w,
                    'wmape_ia_insample': wmape_ia,
                    'peso_ia_usado': round(_peso_efetivo * 100, 0),
                }
                if wmape_original:
                    row_dict['wmape_original'] = wmape_original.get(sku, np.nan)
                ia_rows.append(row_dict)
                barra2.progress((idx+1)/len(skus_all), text=f"IA: {idx+1}/{len(skus_all)}")

            barra2.empty()
            st.session_state['df_ia'] = pd.DataFrame(ia_rows)
            st.write("✅ Etapa 3/3 — Consolidando resultados...")
            status.update(label="✅ Pipeline concluído com sucesso!", state="complete")
        st.rerun()

    # ─── GATILHO OUT-OF-SAMPLE ────────────────────────────────
    if rodar_oos:
        df_ia_loaded = st.session_state.get('df_ia')
        if df_ia_loaded is not None and len(df_ia_loaded) > 0:
            _base_hash_oos = str(len(df_base)) + str(df_base[col_demanda].sum())

            # ── Resolver SKUs pelo filtro de classe ──────────
            _classes_sel_oos = st.session_state.get('classes_selecionadas_oos', [])
            _df_meta_oos     = st.session_state.get('df_sku_meta', pd.DataFrame())
            _skus_filtrados  = None  # None = todos

            if _classes_sel_oos and not _df_meta_oos.empty and 'classe' in _df_meta_oos.columns:
                _cls_codes_oos = [c.split(' — ')[0].strip() for c in _classes_sel_oos]
                _skus_filtrados = frozenset(
                    _df_meta_oos[
                        _df_meta_oos['classe'].astype(str).isin(_cls_codes_oos)
                    ]['sku'].unique()
                )
                _label_seg = f"{len(_classes_sel_oos)} classe(s) — {len(_skus_filtrados)} SKUs"
            else:
                _label_seg = "Todos os SKUs elegíveis"

            with st.status(f"🔬 IA Out-of-Sample — {_label_seg}", expanded=True) as _oos_status:
                st.write(f"Segmentação: **{_label_seg}** | {n_test_oos} períodos de teste por SKU")
                _woos_map, _det_map = calcular_wmape_ia_oos(
                    _base_hash_oos, df_base, col_sku, col_periodo,
                    col_ano, col_demanda, df_ia_loaded, n_test=n_test_oos,
                    skus_alvo=_skus_filtrados
                )
                df_oos_result = df_ia_loaded[['sku','melhor_metodo','wmape_melhor']].copy()
                df_oos_result['wmape_ia_oos']    = df_oos_result['sku'].map(_woos_map)
                df_oos_result['wmape_stat_oos']  = df_oos_result['sku'].apply(lambda s: _det_map.get(s, {}).get('wmape_stat_oos', np.nan))
                df_oos_result['recomendacao']    = df_oos_result['sku'].apply(lambda s: _det_map.get(s, {}).get('recomendacao', '—'))
                df_oos_result['status']          = df_oos_result['sku'].apply(lambda s: _det_map.get(s, {}).get('status', '—'))

                _peso_por_sku = {}
                for _sku_oos, _det in _det_map.items():
                    if _det.get('status') != 'OK': continue
                    _w_ia   = _det.get('wmape_ia_oos',  np.nan)
                    _w_stat = _det.get('wmape_stat_oos', np.nan)
                    if pd.isna(_w_ia) or pd.isna(_w_stat) or _w_stat == 0: continue
                    ratio = _w_ia / _w_stat
                    if ratio < 0.90:   _peso_por_sku[_sku_oos] = 0.70
                    elif ratio <= 1.10: _peso_por_sku[_sku_oos] = 0.50
                    else:               _peso_por_sku[_sku_oos] = 0.00

                df_oos_result['peso_ia_automatico'] = df_oos_result['sku'].apply(lambda s: _peso_por_sku.get(s, np.nan))
                df_oos_result['peso_ia_pct']        = df_oos_result['peso_ia_automatico'].apply(
                    lambda x: f"{x*100:.0f}%" if pd.notna(x) else "slider global"
                )
                # Salvar segmentação usada para exibição
                df_oos_result['_segmentacao'] = _label_seg

                st.session_state['df_oos']       = df_oos_result
                st.session_state['_det_oos']     = _det_map
                st.session_state['peso_por_sku'] = _peso_por_sku
                st.session_state['_oos_segmentacao'] = _label_seg

                # ── RECALCULAR df_ia COM OS NOVOS PESOS ───────────────
                # Atualiza só as colunas dependentes do peso (previsao_combinada
                # e peso_ia_usado). As previsões individuais (IA e estatística)
                # já foram geradas pelo pipeline e não mudam.
                st.write("🔄 Aplicando pesos calibrados às previsões combinadas...")
                _df_ia_upd = df_ia_loaded.copy()
                _n_recalc = 0
                for _idx, _row in _df_ia_upd.iterrows():
                    _sku_r   = _row['sku']
                    _pred_st = _row.get('previsao_estatistica', 0)
                    _pred_ia = _row.get('previsao_ia', np.nan)

                    # Peso: automático (OOS) se existir, senão slider global
                    _peso_r = _peso_por_sku.get(_sku_r, _peso_por_sku.get(float(_sku_r), None))
                    _peso_efetivo = _peso_r if _peso_r is not None else peso_ia

                    if pd.notna(_pred_ia):
                        _pred_comb = _peso_efetivo * _pred_ia + (1 - _peso_efetivo) * _pred_st
                    else:
                        _pred_comb = _pred_st
                        _peso_efetivo = 0.0

                    _df_ia_upd.at[_idx, 'previsao_combinada'] = round(_pred_comb, 2)
                    _df_ia_upd.at[_idx, 'peso_ia_usado']      = round(_peso_efetivo * 100, 0)
                    if _peso_r is not None:
                        _n_recalc += 1

                st.session_state['df_ia'] = _df_ia_upd
                st.write(f"✅ {_n_recalc} SKUs com peso automático aplicado ao df_ia.")

                _oos_status.update(label="✅ Out-of-Sample concluído!", state="complete")
            st.rerun()
        else:
            st.warning("⚠️ Rode o Pipeline Completo antes de calcular o OOS.")

    df_bt = st.session_state.get('df_backtest')
    df_ia = st.session_state.get('df_ia')

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Simulação Retrospectiva", "🎯 Seletor de Método", "🔍 Análise por SKU",
        "🤖 IA + Previsão", "📋 Top 10 Piores WMAPE", "📖 Guia do Usuário",
        "🧭 Imediato",
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1: SIMULAÇÃO RETROSPECTIVA
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("📊 Comparação Retrospectiva — Walk-Forward Validation")
        st.markdown("Para cada SKU, o pipeline simula como **cada método teria se saído nos últimos N períodos**, treinando apenas com dados passados (sem *data leakage*).")

        if df_bt is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        wmape_cols = [c for c in df_bt.columns if c.startswith('wmape_')]
        metodos_nomes = [c.replace('wmape_', '') for c in wmape_cols]
        if not wmape_cols:
            st.warning("⚠️ Dados de backtesting incompletos. Clique em **Limpar Cache** e rode o pipeline novamente.")
            st.stop()
        wmape_orig_por_sku = wmape_original or {}
        med_wmape_orig = np.nanmedian(list(wmape_orig_por_sku.values())) if wmape_orig_por_sku else np.nan

        resumo = []
        for col, nome in zip(wmape_cols, metodos_nomes):
            vals       = df_bt[col].dropna()
            n_melhor   = (df_bt['melhor_metodo'] == nome).sum()
            n_abaixo35 = (vals < 0.35).sum()
            med_v5     = vals.median()
            ganho_str  = "—"
            if pd.notna(med_wmape_orig) and med_wmape_orig > 0:
                ganho = (med_wmape_orig - med_v5) / med_wmape_orig * 100
                ganho_str = f"{'▲' if ganho >= 0 else '▼'} {abs(ganho):.1f}%"
            resumo.append({
                'Método': nome, 'Descrição': METODOS_DESC.get(nome, ''),
                'SKUs — Melhor': int(n_melhor), '% SKUs': f"{n_melhor/len(df_bt)*100:.0f}%",
                'SKUs WMAPE < 35%': int(n_abaixo35), '% SKUs < 35%': f"{n_abaixo35/max(len(vals),1)*100:.0f}%",
                'Ganho vs Original': ganho_str,
            })

        df_res = pd.DataFrame(resumo)
        _col_sort = 'SKUs WMAPE < 35%'
        if not df_res.empty and _col_sort in df_res.columns:
            df_res = df_res.sort_values(_col_sort, ascending=False).reset_index(drop=True)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            melt_data = []
            for col, nome in zip(wmape_cols, metodos_nomes):
                for v in df_bt[col].dropna():
                    melt_data.append({'Método': nome, 'WMAPE (%)': v * 100})
            df_melt = pd.DataFrame(melt_data)
            if not df_melt.empty and 'Método' in df_melt.columns:
                ordem = df_melt.groupby('Método')['WMAPE (%)'].median().sort_values().index.tolist()
                fig_box = px.box(df_melt, x='Método', y='WMAPE (%)', title='Distribuição do WMAPE por Método',
                                 color='Método', category_orders={'Método': ordem}, template='plotly_white', points='outliers')
                fig_box.update_layout(showlegend=False, xaxis_tickangle=-30)
                fig_box.add_hline(y=20, line_dash='dash', line_color='green', annotation_text='Meta 20%', annotation_position='right')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Rode o pipeline para visualizar a distribuição do WMAPE.")

        with col_b:
            counts = df_bt['melhor_metodo'].value_counts().reset_index()
            counts.columns = ['Método', 'Qtd']
            fig_pie = px.pie(counts, values='Qtd', names='Método', title='Melhor Método por SKU (backtesting)', template='plotly_white', hole=0.4)
            fig_pie.update_traces(textinfo='label+percent')
            st.plotly_chart(fig_pie, use_container_width=True)

        if wmape_original and df_ia is not None:
            st.divider()
            st.markdown("### 📉 Ganho vs Arquivo Original")
            df_comp = df_ia[['sku', 'wmape_melhor']].copy()
            df_comp['wmape_original'] = df_comp['sku'].apply(lambda x: wmape_original.get(x, np.nan))
            df_comp = df_comp.dropna(subset=['wmape_original', 'wmape_melhor'])
            if len(df_comp) > 0:
                df_comp['ganho'] = df_comp['wmape_original'] - df_comp['wmape_melhor']
                df_comp['ganho_pct'] = df_comp['ganho'] / (df_comp['wmape_original'] + 1e-9) * 100
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("SKUs com WMAPE melhorado", f"{(df_comp['ganho'] > 0).sum()}/{len(df_comp)}")
                gc2.metric("Ganho mediano no WMAPE", f"{df_comp['ganho_pct'].median():.1f}%")
                gc3.metric("Total SKUs comparados", len(df_comp))
                fig_ganho = px.scatter(df_comp, x='wmape_original', y='wmape_melhor', hover_data=['sku','ganho_pct'],
                                       title='WMAPE Original vs WMAPE Novo (SONAR)', template='plotly_white', color='ganho', color_continuous_scale='RdYlGn')
                fig_ganho.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
                fig_ganho.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_ganho, use_container_width=True)

        st.divider()
        st.markdown("### 📋 Comparativo Mensal por SKU")
        if df_ia is not None and col_periodo and col_ano:
            meses_ord = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
            df_mensal = df_base[[col_sku, col_ano, col_periodo, col_demanda]].copy()
            df_mensal.columns = ['sku','ano','mes','realizado']
            df_mensal['mes_num'] = df_mensal['mes'].astype(str).str.lower().str[:3].map(meses_ord).fillna(0).astype(int)
            df_mensal['periodo_lbl'] = df_mensal['mes'].astype(str).str.lower().str[:3] + '/' + df_mensal['ano'].astype(str).str[-2:]
            periodos_disp = (df_mensal[['ano','mes_num','periodo_lbl']].drop_duplicates().sort_values(['ano','mes_num'])['periodo_lbl'].tolist())
            col_f1, col_f2 = st.columns([2, 4])
            per_sel = col_f1.selectbox("📅 Filtrar por período", options=periodos_disp, index=len(periodos_disp)-1 if periodos_disp else 0, key='tab1_per_sel')
            df_per = df_mensal[df_mensal['periodo_lbl'] == per_sel].copy()
            df_per = df_per.groupby('sku', as_index=False)['realizado'].sum()

            prev_orig_map = {}
            wmape_orig_per_map = {}
            if df_aval is not None:
                col_prev_orig = encontrar_coluna(df_aval, 'prev', 'previsao', 'forecast', 'hibrido')
                col_mes_aval  = encontrar_coluna(df_aval, 'mes', 'periodo', 'referencia')
                col_sku_aval2 = encontrar_coluna(df_aval, 'sku', 'material', 'codigo')
                col_ano_aval  = encontrar_coluna(df_aval, 'ano', 'year')
                if col_prev_orig and col_mes_aval and col_sku_aval2:
                    df_aval_f = df_aval.copy()
                    df_aval_f['_mes_abr'] = df_aval_f[col_mes_aval].astype(str).str.lower().str[:3]
                    mes_sel_abr = per_sel[:3].lower()
                    ano_sel = per_sel[-2:]
                    if col_ano_aval:
                        df_aval_f['_ano_abr'] = df_aval_f[col_ano_aval].astype(str).str[-2:]
                        df_aval_per = df_aval_f[(df_aval_f['_mes_abr'] == mes_sel_abr) & (df_aval_f['_ano_abr'] == ano_sel)]
                        if len(df_aval_per) == 0:
                            df_aval_per = df_aval_f[df_aval_f['_mes_abr'] == mes_sel_abr]
                    else:
                        df_aval_per = df_aval_f[df_aval_f['_mes_abr'] == mes_sel_abr]
                    prev_orig_map = df_aval_per.groupby(col_sku_aval2)[col_prev_orig].mean().to_dict()

            melhor_prev_map = df_ia.set_index('sku')['previsao_combinada'].to_dict() if df_ia is not None else {}
            wmape_v5_map    = df_ia.set_index('sku')['wmape_melhor'].to_dict() if df_ia is not None else {}
            df_per['prev_original']  = df_per['sku'].apply(lambda x: prev_orig_map.get(x, np.nan))
            df_per['prev_v5']        = df_per['sku'].apply(lambda x: melhor_prev_map.get(x, np.nan))
            df_per['wmape_v5']       = df_per['sku'].apply(lambda x: wmape_v5_map.get(x, np.nan))
            df_per['wmape_original'] = df_per['sku'].apply(lambda x: wmape_orig_per_map.get(x, np.nan))
            df_per['erro_prev_orig'] = np.where(df_per['realizado'] > 0, abs(df_per['prev_original'] - df_per['realizado']) / df_per['realizado'] * 100, np.nan)
            df_per['erro_prev_v5']   = np.where(df_per['realizado'] > 0, abs(df_per['prev_v5'] - df_per['realizado']) / df_per['realizado'] * 100, np.nan)

            df_per_show = df_per.rename(columns={'sku':'SKU','realizado':'Realizado','prev_original':'Prev. Original (arquivo)',
                                                  'prev_v5':'Melhor Prev. SONAR','wmape_v5':'WMAPE SONAR (backtesting)',
                                                  'wmape_original':'WMAPE Original (período)','erro_prev_orig':'Erro % Prev. Original','erro_prev_v5':'Erro % Prev. SONAR'}).copy()
            for col_fmt in ['WMAPE SONAR (backtesting)','WMAPE Original (período)']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            for col_fmt in ['Erro % Prev. Original','Erro % Prev. SONAR']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            for col_fmt in ['Prev. Original (arquivo)','Melhor Prev. SONAR','Realizado']:
                if col_fmt in df_per_show.columns:
                    df_per_show[col_fmt] = df_per_show[col_fmt].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "—")

            st.caption(f"Período: **{per_sel}** — {len(df_per_show)} SKUs")
            st.dataframe(df_per_show, use_container_width=True, hide_index=True)

            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            total_real = df_per['realizado'].sum()
            total_v5   = df_per['prev_v5'].sum()
            r_col1.metric("Total Realizado", f"{total_real:,.0f}")
            r_col2.metric("Total Prev. SONAR", f"{total_v5:,.0f}")
            r_col3.metric("Diferença", f"{total_v5-total_real:,.0f}", delta=f"{(total_v5-total_real)/max(total_real,1)*100:.1f}%")

            buf_mensal = exportar_excel_visual({f'Mensal_{per_sel.replace("/","_")}': {
                'df': df_per.rename(columns={'sku':'SKU','realizado':'Realizado','prev_original':'Prev. Original','prev_v5':'Melhor Prev. SONAR','wmape_v5':'WMAPE SONAR','wmape_original':'WMAPE Original','erro_prev_orig':'Erro % Orig','erro_prev_v5':'Erro % SONAR'}),
                'col_wmape': 'WMAPE SONAR', 'col_widths': {'SKU':12,'Realizado':14,'Prev. Original':20,'Melhor Prev. SONAR':18,'WMAPE SONAR':18,'WMAPE Original':18,'Erro % Orig':16,'Erro % SONAR':16},
            }})
            r_col4.download_button("📥 Exportar Excel", data=buf_mensal, file_name=f"mensal_{per_sel.replace('/','_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Rode o pipeline completo para visualizar a tabela mensal.")

    # ─────────────────────────────────────────────────────────
    # TAB 2: SELETOR DE MÉTODO
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🎯 Melhor Método por SKU")
        if df_bt is None or df_ia is None:
            st.info("👆 Clique em **🚀 Rodar Pipeline Completo** na barra lateral para iniciar.")
            st.stop()

        df_sel = df_bt[['sku','melhor_metodo','melhor_wmape']].merge(df_ia[['sku','classificacao','tendencia','cv','n_periodos']], on='sku', how='left')
        fc1, fc2, fc3 = st.columns(3)
        f_metodo    = fc1.multiselect("Filtrar método", df_sel['melhor_metodo'].dropna().unique())
        f_classe    = fc2.multiselect("Filtrar classificação", df_sel['classificacao'].dropna().unique())
        f_wmape_max = fc3.slider("WMAPE máximo (%)", 0, 200, 100)
        df_f = df_sel.copy()
        if f_metodo: df_f = df_f[df_f['melhor_metodo'].isin(f_metodo)]
        if f_classe: df_f = df_f[df_f['classificacao'].isin(f_classe)]
        df_f = df_f[df_f['melhor_wmape'].fillna(999) <= f_wmape_max / 100]
        df_display = df_f.copy()
        df_display['melhor_wmape'] = (df_display['melhor_wmape'] * 100).round(1).astype(str) + '%'
        df_display['cv'] = df_display['cv'].round(3)
        st.dataframe(df_display.rename(columns={'sku':'SKU','melhor_metodo':'Melhor Método','melhor_wmape':'WMAPE','classificacao':'Classificação','tendencia':'Tendência','cv':'CV','n_periodos':'N Períodos'}), use_container_width=True, hide_index=True)
        st.caption(f"{len(df_f)} SKUs exibidos")

        buf_sel = exportar_excel_visual({'Melhor_Metodo_por_SKU': {'df': df_f.rename(columns={'sku':'SKU','melhor_metodo':'Melhor Método','melhor_wmape':'WMAPE','classificacao':'Classificação','tendencia':'Tendência','cv':'CV','n_periodos':'N Períodos'}), 'col_wmape':'WMAPE','col_metodo':'Melhor Método','col_widths':{'SKU':14,'Melhor Método':18,'WMAPE':14,'Classificação':16,'Tendência':20,'CV':10,'N Períodos':14}}})
        st.download_button("📥 Exportar Melhor Método por SKU (.xlsx)", data=buf_sel, file_name=f"melhor_metodo_sku_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        col_l, col_r = st.columns(2)
        with col_l:
            cross = pd.crosstab(df_sel['classificacao'].fillna('N/A'), df_sel['melhor_metodo'].fillna('N/A'))
            fig_heat = px.imshow(cross, text_auto=True, title='Classificação da Demanda × Melhor Método', color_continuous_scale='Blues', template='plotly_white', aspect='auto')
            fig_heat.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_heat, use_container_width=True)
        with col_r:
            df_cv_wmape = df_sel.copy()
            df_cv_wmape['wmape_pct'] = df_cv_wmape['melhor_wmape'] * 100
            fig_cls = px.box(df_cv_wmape.dropna(subset=['wmape_pct','classificacao']), x='classificacao', y='wmape_pct', color='classificacao', title='WMAPE por Classificação da Demanda', template='plotly_white', labels={'classificacao':'Classificação','wmape_pct':'WMAPE (%)'})
            fig_cls.update_layout(showlegend=False)
            st.plotly_chart(fig_cls, use_container_width=True)

        st.divider()
        st.markdown("### 💡 Regras de Recomendação Automática")
        regras = [
            ("🔵 Intermitente","Croston ou SBA","> 50% de zeros na série"),
            ("🔴 Errática","SES (alpha alto)","CV > 0.6"),
            ("🟡 Variável","SES ou Holt","CV entre 0.25 e 0.6"),
            ("🟢 Estável","MA-6 ou SES","CV < 0.25"),
            ("📈 Tendência ↑","Holt ou Holt-Winters","Slope significativo (p<0.1)"),
            ("📉 Tendência ↓","Holt","Slope negativo significativo"),
            ("🔄 Sazonal","Holt-Winters","ACF lag-12 > 0.4"),
        ]
        st.dataframe(pd.DataFrame(regras, columns=['Perfil','Método Recomendado','Critério']), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    # TAB 3: ANÁLISE POR SKU
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("🔍 Análise Detalhada por SKU")
        skus_lista = sorted(df_base[col_sku].dropna().unique())
        sku_sel = st.selectbox("Selecione o SKU", options=skus_lista, key='sku_sel_tab3')
        serie_sku = df_base[df_base[col_sku] == sku_sel][col_demanda].reset_index(drop=True).astype(float)
        x_labels = df_base[df_base[col_sku] == sku_sel][col_periodo].reset_index(drop=True).astype(str).tolist() if col_periodo else list(range(len(serie_sku)))

        cls_sku = classificar_demanda(serie_sku)
        slope_sku, tend_sku = detectar_tendencia(serie_sku)
        sazon_sku = detectar_sazonalidade(serie_sku)
        cv_sku = serie_sku.std() / serie_sku.mean() if serie_sku.mean() > 0 else 0
        zeros_pct = (serie_sku == 0).mean() * 100

        ms1, ms2, ms3, ms4, ms5, ms6 = st.columns(6)
        ms1.metric("Média", f"{serie_sku.mean():.1f}")
        ms2.metric("Desvio Padrão", f"{serie_sku.std():.1f}")
        ms3.metric("CV", f"{cv_sku:.3f}")
        ms4.metric("Zeros", f"{zeros_pct:.0f}%")
        ms5.metric("Classificação", f"{CLASSE_ICONES.get(cls_sku,'')} {cls_sku}")
        ms6.metric("Tendência", tend_sku)

        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=x_labels, y=serie_sku.values, mode='lines+markers', name='Histórico', line=dict(color='#2c3e50', width=2), marker=dict(size=5), fill='tozeroy', fillcolor='rgba(44,62,80,0.07)'))
        if len(serie_sku) >= 4:
            x_arr = np.arange(len(serie_sku))
            s_, i_ = np.polyfit(x_arr, serie_sku.values, 1)
            fig_main.add_trace(go.Scatter(x=x_labels, y=s_*x_arr+i_, mode='lines', name='Tendência', line=dict(color='#e74c3c', width=1.5, dash='dash')))
        fig_main.add_hline(y=serie_sku.mean(), line_dash='dot', line_color='#27ae60', annotation_text='Média')
        cores = px.colors.qualitative.Pastel
        for i, (nome, fn) in enumerate(METODOS.items()):
            try:
                pred_v = fn(serie_sku, h=1)[0]
                fig_main.add_trace(go.Scatter(x=["Próx."], y=[pred_v], mode='markers', name=f'{nome}: {pred_v:.1f}', marker=dict(size=13, symbol='diamond', color=cores[i%len(cores)], line=dict(width=1, color='#333'))))
            except: pass
        fig_main.update_layout(title=f'Histórico + Previsões — SKU: {sku_sel}', xaxis_title='Período', yaxis_title='Demanda', template='plotly_white', hovermode='x unified', height=420, legend=dict(orientation='h', yanchor='bottom', y=-0.4))
        st.plotly_chart(fig_main, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            rows_prev = []
            best_method = None
            if df_bt is not None:
                row_bt = df_bt[df_bt['sku'] == sku_sel]
                if len(row_bt) > 0: best_method = row_bt.iloc[0]['melhor_metodo']
            for nome, fn in METODOS.items():
                try: pred_v = fn(serie_sku, h=1)[0]
                except: pred_v = np.nan
                w_bt = np.nan
                if df_bt is not None:
                    rb = df_bt[df_bt['sku'] == sku_sel]
                    if len(rb) > 0: w_bt = rb.iloc[0].get(f'wmape_{nome}', np.nan)
                rows_prev.append({'Método': nome, 'Previsão': round(pred_v, 2) if pd.notna(pred_v) else '—', 'WMAPE BT': f"{w_bt*100:.1f}%" if pd.notna(w_bt) else '—', 'Recomendado': '⭐ Melhor' if nome == best_method else ''})
            st.markdown("**Previsões próximo período**")
            st.dataframe(pd.DataFrame(rows_prev), use_container_width=True, hide_index=True)

        with col_b:
            if len(serie_sku) >= 8:
                max_lag = min(12, len(serie_sku)//2)
                acf_vals = [serie_sku.autocorr(lag=i) for i in range(1, max_lag+1)]
                fig_acf = go.Figure(go.Bar(x=[f'L{i}' for i in range(1, max_lag+1)], y=acf_vals, marker_color=['#27ae60' if v > 0 else '#e74c3c' for v in acf_vals]))
                conf_95 = 1.96 / np.sqrt(len(serie_sku))
                fig_acf.add_hline(y=conf_95, line_dash='dot', line_color='gray')
                fig_acf.add_hline(y=-conf_95, line_dash='dot', line_color='gray')
                fig_acf.update_layout(title='Autocorrelação (ACF)', yaxis_title='ACF', template='plotly_white', height=300)
                st.plotly_chart(fig_acf, use_container_width=True)
            else:
                st.info("Série curta — autocorrelação indisponível.")

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
**Arquitetura:** Gradient Boosting Regressor (scikit-learn) — 200 árvores sequenciais por SKU

**Features de entrada (11 variáveis):** Lags 1–6, Média móvel 3/6, Desvio padrão 3, Mínimo e Máximo 3 períodos

**Previsão combinada:**
`prev_comb = {peso_ia*100:.0f}% × IA + {(1-peso_ia)*100:.0f}% × melhor método estatístico`

Ajuste o peso no slider da barra lateral. Após rodar o **🔬 IA Out-of-Sample**, o peso é calibrado automaticamente por SKU.
""")

        skus_ia = sorted(df_ia['sku'].dropna().unique())
        sku_ia = st.selectbox("Selecione o SKU para análise da IA", options=skus_ia, key='sku_ia')
        row_ia_df = df_ia[df_ia['sku'] == sku_ia]
        if len(row_ia_df) == 0:
            st.warning("SKU não encontrado na tabela de IA.")
            st.stop()
        row_ia = row_ia_df.iloc[0].to_dict()
        serie_ia_full = df_base[df_base[col_sku] == sku_ia][col_demanda].reset_index(drop=True).astype(float)

        mi1, mi2, mi3, mi4 = st.columns(4)
        mi1.metric("Prev. Estatística",   f"{row_ia['previsao_estatistica']:.1f}")
        mi2.metric("Prev. IA",            f"{row_ia['previsao_ia']:.1f}" if pd.notna(row_ia.get('previsao_ia')) else "N/A (série curta)")
        mi3.metric("Prev. Combinada",     f"{row_ia['previsao_combinada']:.1f}")
        mi4.metric("WMAPE Melhor Método", f"{row_ia['wmape_melhor']*100:.1f}%" if pd.notna(row_ia.get('wmape_melhor')) else "—")

        col_vis1, col_vis2 = st.columns([3, 2])
        with col_vis1:
            model_vis = treinar_ia(serie_ia_full)
            if model_vis is not None:
                wmape_v, y_real, y_pred_ia = wmape_ia_insample(model_vis, serie_ia_full)
                offset = len(serie_ia_full) - len(y_real)
                fn_melhor = METODOS[row_ia['melhor_metodo']]
                preds_stat_is = []
                for j in range(len(y_real)):
                    idx_t = offset + j
                    s_train = serie_ia_full.iloc[:idx_t]
                    preds_stat_is.append(fn_melhor(s_train, h=1)[0] if len(s_train) >= 2 else serie_ia_full.mean())
                x_plot = list(range(offset, len(serie_ia_full)))
                if col_periodo and col_ano:
                    _df_sku_ia = df_base[df_base[col_sku] == sku_ia][[col_ano, col_periodo]].reset_index(drop=True)
                    _mord2 = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
                    _df_sku_ia['_mn'] = _df_sku_ia[col_periodo].astype(str).str.lower().str[:3].map(_mord2).fillna(0)
                    _df_sku_ia['_an'] = pd.to_numeric(_df_sku_ia[col_ano], errors='coerce').fillna(0)
                    _df_sku_ia = _df_sku_ia.sort_values(['_an','_mn']).reset_index(drop=True)
                    x_labels_ia_all = [f"{str(r[col_periodo]).lower()[:3]}/{str(int(r[col_ano]))[-2:]}" for _, r in _df_sku_ia.iterrows()]
                    _prox_m = (int(_df_sku_ia['_mn'].iloc[-1]) % 12) + 1
                    _prox_a = int(_df_sku_ia['_an'].iloc[-1]) + (1 if _df_sku_ia['_mn'].iloc[-1] == 12 else 0)
                    _mn2 = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}
                    x_labels_ia_prox = [f"{_mn2[_prox_m]}/{str(_prox_a)[-2:]}"]
                else:
                    x_labels_ia_all = [str(i) for i in range(len(serie_ia_full))]
                    x_labels_ia_prox = ["Próx."]
                x_labels_ia_plot = [x_labels_ia_all[i] for i in x_plot] if x_plot else []
                fig_ia = go.Figure()
                fig_ia.add_trace(go.Scatter(x=x_labels_ia_all, y=serie_ia_full.values, mode='lines+markers', name='Histórico Real', line=dict(color='#2c3e50', width=2), marker=dict(size=4)))
                fig_ia.add_trace(go.Scatter(x=x_labels_ia_plot, y=y_pred_ia, mode='lines', name=f'IA (WMAPE={wmape_v*100:.1f}%)', line=dict(color='#e67e22', width=2, dash='dot')))
                fig_ia.add_trace(go.Scatter(x=x_labels_ia_plot, y=preds_stat_is, mode='lines', name=f'Método {row_ia["melhor_metodo"]}', line=dict(color='#3498db', width=1.5, dash='dashdot')))
                fig_ia.add_trace(go.Scatter(x=x_labels_ia_prox, y=[row_ia['previsao_combinada']], mode='markers', name='Prev. Combinada (próx.)', marker=dict(size=16, symbol='star', color='#e74c3c', line=dict(width=1.5, color='white'))))
                fig_ia.update_layout(title=f'IA vs Modelo Estatístico — SKU {sku_ia}', xaxis_title='Período', yaxis_title='Demanda', template='plotly_white', height=400, legend=dict(orientation='h', y=-0.3))
                st.plotly_chart(fig_ia, use_container_width=True)
            else:
                st.warning(f"⚠️ Série do SKU {sku_ia} tem menos de 14 períodos — IA não pode ser treinada.")

        with col_vis2:
            if model_vis is not None:
                _gbm_vis = _get_gbm(model_vis)
                imp_df = pd.DataFrame({'Feature': FEAT_NAMES, 'Importância': _gbm_vis.feature_importances_}).sort_values('Importância', ascending=True)
                fig_imp = px.bar(imp_df, x='Importância', y='Feature', orientation='h', title='Importância das Features (IA)', template='plotly_white', color='Importância', color_continuous_scale='Blues')
                fig_imp.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
                st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("**💡 Diagnóstico e Sugestão:**")
            st.markdown(gerar_sugestao(row_ia, row_ia.get('wmape_melhor')))

        st.divider()
        st.markdown("### 📊 Resumo IA — Todos os SKUs")
        df_ia_show = df_ia.copy()
        df_ia_show['wmape_melhor']      = (df_ia_show['wmape_melhor'] * 100).round(1).astype(str) + '%'
        df_ia_show['wmape_ia_insample'] = df_ia_show['wmape_ia_insample'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        st.dataframe(df_ia_show.rename(columns={'sku':'SKU','classificacao':'Classificação','tendencia':'Tendência','media_historica':'Média Hist.','cv':'CV','previsao_estatistica':'Prev. Estatística','previsao_ia':'Prev. IA','previsao_combinada':'Prev. Combinada','melhor_metodo':'Melhor Método','wmape_melhor':'WMAPE','wmape_ia_insample':'WMAPE IA (in-sample)'}), use_container_width=True, hide_index=True)

        buf_ia = exportar_excel_visual({'Resumo_IA_Todos_SKUs': {'df': df_ia.rename(columns={'sku':'SKU','classificacao':'Classificação','tendencia':'Tendência','media_historica':'Média Hist.','cv':'CV','previsao_estatistica':'Prev. Estatística','previsao_ia':'Prev. IA','previsao_combinada':'Prev. Combinada','melhor_metodo':'Melhor Método','wmape_melhor':'WMAPE','wmape_ia_insample':'WMAPE IA (in-sample)'}), 'col_wmape':'WMAPE','col_metodo':'Melhor Método','col_widths':{'SKU':14,'Classificação':16,'Tendência':18,'Média Hist.':14,'CV':10,'Prev. Estatística':18,'Prev. IA':14,'Prev. Combinada':16,'Melhor Método':18,'WMAPE':14,'WMAPE IA (in-sample)':22}}})
        st.download_button("📥 Exportar Resumo IA (.xlsx)", data=buf_ia, file_name=f"resumo_ia_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── PAINEL IA OUT-OF-SAMPLE ──────────────────────────
        st.divider()
        st.markdown("### 🔬 IA Out-of-Sample — Calibração do Peso por SKU")

        df_oos = st.session_state.get('df_oos')
        _oos_seg_label = st.session_state.get('_oos_segmentacao', None)

        if df_oos is None:
            # ── Flag de segmentação configurada (ainda não rodou) ──
            _classes_cfg = st.session_state.get('classes_selecionadas_oos', [])
            _df_meta_tab = st.session_state.get('df_sku_meta', pd.DataFrame())
            if _classes_cfg and not _df_meta_tab.empty and 'classe' in _df_meta_tab.columns:
                _cls_codes_tab = [c.split(' — ')[0].strip() for c in _classes_cfg]
                _n_skus_tab = _df_meta_tab[_df_meta_tab['classe'].astype(str).isin(_cls_codes_tab)]['sku'].nunique()
                st.markdown(
                    f'<div class="seg-flag">🎯 Segmentação configurada: '
                    f'{len(_classes_cfg)} classe(s) → {_n_skus_tab} SKUs serão avaliados</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="seg-flag" style="background:linear-gradient(90deg,#64748B,#94A3B8);">'
                    '🔍 Sem segmentação — todos os SKUs elegíveis serão avaliados</div>',
                    unsafe_allow_html=True
                )
            st.info("Clique em **🔬 IA Out-of-Sample** na barra lateral para iniciar. Tempo varia conforme o número de SKUs.")
            with st.expander("Por que isso importa?"):
                st.markdown(
                    "**O problema do in-sample:** o modelo treinou nos mesmos dados que avaliou — resultado otimista.\n\n"
                    "**O out-of-sample:** re-treina a IA excluindo os períodos de teste — WMAPE honesto e comparável ao backtesting estatístico.\n\n"
                    "**O filtro de classe:** selecione as classes de material na sidebar para restringir a avaliação aos SKUs relevantes, reduzindo o tempo de processamento.\n\n"
                    "**Resultado prático:** peso automático calibrado por SKU (0%, 50% ou 70%) em vez de um valor global arbitrário."
                )
        else:
            # ── Flag de segmentação usada ──────────────────────
            if _oos_seg_label:
                st.markdown(
                    f'<div class="seg-flag">🎯 OOS calculado com segmentação: <b>{_oos_seg_label}</b></div>',
                    unsafe_allow_html=True
                )

            _det_oos = st.session_state.get('_det_oos', {})
            df_oos_val = df_oos.dropna(subset=['wmape_ia_oos','wmape_stat_oos'])
            n_ia_wins   = int((df_oos_val['wmape_ia_oos'] < df_oos_val['wmape_stat_oos']).sum())
            n_stat_wins = int(len(df_oos_val) - n_ia_wins)
            n_sem_dados = int(df_oos['wmape_ia_oos'].isna().sum())

            _pesos_ativos = st.session_state.get('peso_por_sku', {})
            _n_auto = len(_pesos_ativos)
            if _n_auto > 0:
                st.success(f"✅ **Peso automático ativo** para **{_n_auto} SKUs**. Previsão Combinada já usa pesos calibrados pelo OOS.")

            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("SKUs avaliados", len(df_oos_val))
            oc2.metric("IA vence estatístico", n_ia_wins, help="Peso IA = 70%")
            oc3.metric("Estatístico vence IA", n_stat_wins, help="Peso IA = 0%")
            oc4.metric("Série curta (excluídos)", n_sem_dados)

            _f_rec = st.selectbox("Filtrar por recomendação", options=["Todos","IA é mais confiável","Estatístico é mais confiável","Desempenho similar"], key="oos_filter_rec")
            df_oos_show = df_oos_val.copy()
            if _f_rec == "IA é mais confiável":
                df_oos_show = df_oos_show[df_oos_show['wmape_ia_oos'] < df_oos_show['wmape_stat_oos']]
            elif _f_rec == "Estatístico é mais confiável":
                df_oos_show = df_oos_show[df_oos_show['wmape_ia_oos'] >= df_oos_show['wmape_stat_oos']]
            elif _f_rec == "Desempenho similar":
                df_oos_show = df_oos_show[df_oos_show['recomendacao'].str.contains("similar", na=False)]

            df_oos_display = df_oos_show.rename(columns={'sku':'SKU','melhor_metodo':'Melhor Método','wmape_melhor':'WMAPE BT','wmape_ia_oos':'WMAPE IA (OOS)','wmape_stat_oos':'WMAPE Estat. (OOS)','recomendacao':'Recomendação','peso_ia_pct':'Peso IA Automático'}).copy()
            for cp in ['WMAPE BT','WMAPE IA (OOS)','WMAPE Estat. (OOS)']:
                if cp in df_oos_display.columns:
                    df_oos_display[cp] = df_oos_display[cp].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            st.dataframe(df_oos_display, use_container_width=True, hide_index=True)
            st.caption(f"{len(df_oos_show)} SKUs exibidos")

            if len(df_oos_val) > 0:
                _p95 = float(np.percentile(pd.concat([df_oos_val['wmape_stat_oos'],df_oos_val['wmape_ia_oos']]).dropna(), 95))
                _axis_max = round(_p95 * 1.15, 2)
                df_oos_plot = df_oos_val.copy()
                df_oos_plot['_x'] = df_oos_plot['wmape_stat_oos'].clip(upper=_axis_max)
                df_oos_plot['_y'] = df_oos_plot['wmape_ia_oos'].clip(upper=_axis_max)
                df_oos_plot['_outlier'] = ((df_oos_val['wmape_stat_oos'] > _axis_max) | (df_oos_val['wmape_ia_oos'] > _axis_max))
                color_map  = {'✅ IA é mais confiável — aumente o peso no slider':'#22C55E','⚠️ Método estatístico é mais confiável — reduza o peso da IA':'#EF4444','➡️ Desempenho similar — peso 50/50 é adequado':'#F59E0B'}
                label_map  = {'✅ IA é mais confiável — aumente o peso no slider':'✅ IA melhor','⚠️ Método estatístico é mais confiável — reduza o peso da IA':'⚠️ Estatístico melhor','➡️ Desempenho similar — peso 50/50 é adequado':'➡️ Similar'}
                df_oos_plot['_label'] = df_oos_plot['recomendacao'].map(label_map).fillna('—')
                fig_oos = px.scatter(df_oos_plot, x='_x', y='_y', color='_label', color_discrete_map={v:color_map.get(k,'#94A3B8') for k,v in label_map.items()},
                                     hover_data={'sku':True,'melhor_metodo':True,'wmape_stat_oos':':.1%','wmape_ia_oos':':.1%','_x':False,'_y':False,'_label':False,'_outlier':False,'recomendacao':False},
                                     title='WMAPE Out-of-Sample: IA vs Melhor Método Estatístico', labels={'_x':'WMAPE Estatístico OOS','_y':'WMAPE IA OOS','_label':'Recomendação'}, template='plotly_white', height=420)
                fig_oos.add_shape(type='line', x0=0, y0=0, x1=_axis_max, y1=_axis_max, line=dict(dash='dash', color='#94A3B8', width=1.5))
                fig_oos.update_xaxes(tickformat='.0%', range=[0,_axis_max])
                fig_oos.update_yaxes(tickformat='.0%', range=[0,_axis_max])
                fig_oos.add_annotation(x=_axis_max*0.25, y=_axis_max*0.75, text="IA pior<br>(acima da linha)", showarrow=False, font=dict(size=10,color='#EF4444'), bgcolor='rgba(255,255,255,0.7)')
                fig_oos.add_annotation(x=_axis_max*0.75, y=_axis_max*0.22, text="IA melhor<br>(abaixo da linha)", showarrow=False, font=dict(size=10,color='#22C55E'), bgcolor='rgba(255,255,255,0.7)')
                n_out = int(df_oos_plot['_outlier'].sum())
                if n_out > 0:
                    fig_oos.add_annotation(xref='paper',yref='paper',x=1,y=1, text=f"{n_out} SKU(s) fora da escala (>{_axis_max:.0%})", showarrow=False, font=dict(size=9,color='#94A3B8'), xanchor='right', yanchor='top')
                fig_oos.update_layout(legend=dict(title='',orientation='h',yanchor='bottom',y=-0.25,xanchor='center',x=0.5), margin=dict(l=60,r=20,t=50,b=80))
                st.plotly_chart(fig_oos, use_container_width=True, key="scatter_oos")

            buf_oos = exportar_excel_visual({'IA_OutofSample': {'df': df_oos_display.reset_index(drop=True), 'col_wmape':['WMAPE BT','WMAPE IA (OOS)','WMAPE Estat. (OOS)'],'col_metodo':'Melhor Método','col_widths':{'SKU':14,'Melhor Método':18,'WMAPE BT':14,'WMAPE IA (OOS)':18,'WMAPE Estat. (OOS)':20,'Recomendação':52,'status':10,'peso_ia_automatico':18,'Peso IA Automático':18,'_segmentacao':22}}})
            st.download_button("📥 Exportar IA Out-of-Sample (.xlsx)", data=buf_oos, file_name=f"ia_oos_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ── HORIZONTE 3 MESES ────────────────────────────────
        st.divider()
        st.markdown("### 📅 Previsão Horizonte 3 Meses — Todos os SKUs")
        if df_ia is not None and col_periodo and col_ano:
            meses_ord2 = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
            meses_nome = {1:'jan',2:'fev',3:'mar',4:'abr',5:'mai',6:'jun',7:'jul',8:'ago',9:'set',10:'out',11:'nov',12:'dez'}
            _hoje = datetime.date.today()
            mes_atual_real = _hoje.month
            ano_atual_real = _hoje.year
            df_ult = df_base[[col_ano, col_periodo]].copy()
            df_ult['_mes_num'] = df_ult[col_periodo].astype(str).str.lower().str[:3].map(meses_ord2).fillna(0).astype(int)
            df_ult['_ano_int'] = pd.to_numeric(df_ult[col_ano], errors='coerce').fillna(0).astype(int)
            df_ult = df_ult.sort_values(['_ano_int','_mes_num'])
            ultimo_ano = int(df_ult['_ano_int'].max())
            ultimo_mes = int(df_ult[df_ult['_ano_int'] == ultimo_ano]['_mes_num'].max())
            proximos = []
            for h_step in range(1, 4):
                m = (mes_atual_real - 1 + h_step) % 12 + 1
                a = ano_atual_real + ((mes_atual_real - 1 + h_step) // 12)
                proximos.append((a, meses_nome[m], f"{meses_nome[m]}/{str(a)[-2:]}"))
            st.info(f"📌 Mês vigente: **{meses_nome[mes_atual_real]}/{ano_atual_real}** | Último período fechado na base: **{meses_nome[ultimo_mes]}/{ultimo_ano}** → Previsões: **{' · '.join(p[2] for p in proximos)}**")

            rows_h = []
            skus_h = df_base[col_sku].unique()
            prog_h = st.progress(0, text="Calculando horizonte...")
            for idx_h, sku in enumerate(skus_h):
                serie_h = df_base[df_base[col_sku] == sku][col_demanda].reset_index(drop=True).astype(float)
                model_h = treinar_ia(serie_h)
                row_h = {'SKU': sku}
                if df_bt is not None:
                    rb_h = df_bt[df_bt['sku'] == sku]
                    melhor_h = rb_h.iloc[0]['melhor_metodo'] if len(rb_h) > 0 else 'MA-3'
                else:
                    melhor_h = 'MA-3'
                fn_h = METODOS[melhor_h]
                try:   preds_stat_h = fn_h(serie_h, h=3)
                except: preds_stat_h = [float(serie_h.mean())] * 3
                preds_ia_h = prever_ia_multistep(model_h, serie_h, h=3)
                for step, (ano_h, mes_h, lbl_h) in enumerate(proximos):
                    pred_stat_step = preds_stat_h[step] if step < len(preds_stat_h) else preds_stat_h[-1]
                    pred_ia_step   = preds_ia_h[step] if (preds_ia_h[step] is not None) else pred_stat_step
                    _pesos_oos_h   = st.session_state.get('peso_por_sku', {})
                    _peso_h        = _pesos_oos_h.get(sku, _pesos_oos_h.get(float(sku), None))
                    _peso_ef_h     = _peso_h if _peso_h is not None else peso_ia
                    row_h[f'Prev {lbl_h}'] = round(max(0.0, _peso_ef_h * pred_ia_step + (1 - _peso_ef_h) * pred_stat_step), 1)
                row_h['Método Base'] = melhor_h
                rows_h.append(row_h)
                prog_h.progress((idx_h+1)/len(skus_h))

            prog_h.empty()
            df_horizonte = pd.DataFrame(rows_h)
            skus_busca = st.text_input("🔍 Filtrar SKU", key='horizonte_sku_filter')
            df_horizonte_show = df_horizonte[df_horizonte['SKU'].astype(str).str.contains(skus_busca.strip())] if skus_busca.strip() else df_horizonte
            st.dataframe(df_horizonte_show, use_container_width=True, hide_index=True)
            n_skus_filtro = len(df_horizonte_show)
            tot_cols = st.columns(3)
            for i, (_, _, lbl_h) in enumerate(proximos):
                col_name = f'Prev {lbl_h}'
                if col_name in df_horizonte_show.columns:
                    tot_cols[i].metric(lbl_h, f"{df_horizonte_show[col_name].sum():,.0f}")
            buf_hz = exportar_excel_visual({'Horizonte_3_Meses': {'df': df_horizonte_show, 'col_metodo':'Método Base', 'col_widths': {'SKU':14,'Método Base':18, **{f'Prev {p[2]}':16 for p in proximos}}}})
            st.download_button("📥 Exportar Horizonte 3 Meses (.xlsx)", data=buf_hz, file_name=f"horizonte_3meses_{meses_nome[mes_atual_real]}_{ano_atual_real}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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

        _base_hash = str(len(df_base)) + str(df_base[col_demanda].sum()) if col_periodo and col_ano else ""

        def _build_top10(n_meses, col_prefix):
            if col_periodo and col_ano:
                _wmap, _nreg, _MIN_R, _ = calcular_wmape_janela(_base_hash, df_base, col_sku, col_periodo, col_ano, col_demanda, df_ia, df_bt, n_meses)
                _df = df_ia.copy()
                _df["wmape_janela"] = _df["sku"].map(_wmap)
                _df = _df.dropna(subset=["wmape_janela"]).sort_values("wmape_janela", ascending=False).head(10).reset_index(drop=True)
                _df["wmape_pct"] = (_df["wmape_janela"] * 100).round(1)
            else:
                _df = df_ia.dropna(subset=["wmape_melhor"]).sort_values("wmape_melhor", ascending=False).head(10).reset_index(drop=True).copy()
                _df["wmape_pct"] = (_df["wmape_melhor"] * 100).round(1)
            _df = _df.dropna(subset=["wmape_pct"]).reset_index(drop=True)
            _df["sku_str"] = "SKU " + _df["sku"].astype(str)
            _df["sugestao"] = _df.apply(lambda r: gerar_sugestao(r.to_dict(), r["wmape_janela"] if "wmape_janela" in r.index and pd.notna(r.get("wmape_janela")) else r.get("wmape_melhor", np.nan)), axis=1)
            return _df

        def _render_panel(df_t, janela_label, col_prefix, col_container):
            with col_container:
                st.markdown(f"#### 📋 Top 10 — Últimos **{janela_label}**")
                st.caption(f"{len(df_t)} SKUs exibidos")
                if len(df_t) == 0:
                    st.info("Nenhum SKU com dados suficientes nesta janela.")
                    return
                df_plot = df_t.sort_values("wmape_pct").copy()
                fig = px.bar(df_plot, x="wmape_pct", y="sku_str", orientation="h", color="wmape_pct",
                             color_continuous_scale="RdYlGn_r", template="plotly_white",
                             labels={"wmape_pct":"WMAPE (%)","sku_str":"SKU"}, text="wmape_pct",
                             category_orders={"sku_str": df_plot["sku_str"].tolist()})
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                fig.update_layout(showlegend=False, coloraxis_showscale=False, height=360, yaxis=dict(type="category"), margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"bar_{col_prefix}")
                st.markdown("**Detalhamento por SKU:**")
                for i, row in enumerate(df_t.itertuples()):
                    wmape_color = cor_wmape(row.wmape_melhor)
                    badge = f'<span style="background:{wmape_color};color:white;padding:2px 8px;border-radius:8px;font-size:12px">{row.wmape_pct:.1f}%</span>'
                    with st.expander(f"#{i+1} SKU: {row.sku} | {row.wmape_pct:.1f}% | {row.classificacao}", expanded=(i < 2)):
                        st.markdown(badge, unsafe_allow_html=True)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Classificação", row.classificacao)
                        c2.metric("Tendência", row.tendencia.split(" ")[-1] if " " in str(row.tendencia) else row.tendencia)
                        c3.metric("Prev. Combinada", f"{row.previsao_combinada:.1f}")
                        c4.metric("Melhor Método", row.melhor_metodo)
                        _df_mini = df_base[df_base[col_sku] == row.sku].copy()
                        if col_periodo and col_ano:
                            _mord2 = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
                            _df_mini["_mn"] = _df_mini[col_periodo].astype(str).str.lower().str[:3].map(_mord2).fillna(0)
                            _df_mini["_an"] = pd.to_numeric(_df_mini[col_ano], errors="coerce").fillna(0)
                            _df_mini = _df_mini.sort_values(["_an","_mn"])
                            x_mini = [f"{str(r2[col_periodo]).lower()[:3]}/{str(int(r2[col_ano]))[-2:]}" for _, r2 in _df_mini.iterrows()]
                            s_mini = _df_mini[col_demanda].reset_index(drop=True).astype(float)
                        else:
                            s_mini = _df_mini[col_demanda].reset_index(drop=True).astype(float)
                            x_mini = list(range(len(s_mini)))
                        fig_m = go.Figure()
                        fig_m.add_trace(go.Scatter(x=x_mini, y=s_mini.values, mode="lines+markers", line=dict(color=wmape_color, width=2), marker=dict(size=4)))
                        fig_m.add_hline(y=s_mini.mean(), line_dash="dash", line_color="gray", annotation_text="Média")
                        fig_m.update_layout(height=160, template="plotly_white", showlegend=False, margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickangle=-45, tickfont=dict(size=8)))
                        st.plotly_chart(fig_m, use_container_width=True, key=f"mini_{col_prefix}_{i}_{row.sku}")
                        st.markdown(f"💡 **Sugestão IA:** {row.sugestao}")

        with st.spinner("Calculando Top 10..."):
            df_12 = _build_top10(12, "12m")
            df_6  = _build_top10(6,  "6m")

        col_12, col_6 = st.columns(2)
        _render_panel(df_12, "12 Meses", "12m", col_12)
        _render_panel(df_6,  "6 Meses",  "6m",  col_6)

        st.divider()
        st.markdown("### 💾 Exportar Resultados")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            buf = exportar_excel(df_bt, df_ia, df_12[["sku","wmape_pct","melhor_metodo","classificacao","tendencia","previsao_estatistica","previsao_ia","previsao_combinada","sugestao"]])
            st.download_button(label="📥 Baixar Relatório Excel Completo", data=buf, file_name=f"forecast_sonar_{datetime.date.today().strftime('%d_%m_%Y')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
        with col_exp2:
            st.download_button(label="📥 Baixar Tabela IA (.csv)", data=df_ia.to_csv(index=False).encode("utf-8-sig"), file_name=f"sugestoes_ia_{datetime.date.today().strftime('%d_%m_%Y')}.csv", mime="text/csv", use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # TAB 6: GUIA DO USUÁRIO
    # ─────────────────────────────────────────────────────────
    with tab6:
        st.markdown("""
<style>
.guia-card{background:#f0f9f8;border-left:4px solid #0D9488;border-radius:8px;padding:14px 18px;margin-bottom:12px;}
.guia-card h4{color:#0F2B4F;margin:0 0 6px 0;}
.guia-card p,.guia-card li{color:#334155;font-size:14px;margin:2px 0;}
.guia-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:white;}
.bg-green{background:#22C55E;}.bg-teal{background:#0D9488;}.bg-yellow{background:#EAB308;color:#1e1e1e;}.bg-red{background:#EF4444;}
</style>""", unsafe_allow_html=True)

        st.markdown("## 📖 Guia do Usuário — SONAR")
        st.markdown("*Manual de referência para analistas de demanda e planejamento*")
        st.divider()

        st.markdown("### 🗂️ Estrutura do Arquivo Excel Esperado")
        st.markdown("""
O SONAR lê as seguintes abas do arquivo Excel:

| Aba | Obrigatória | Descrição |
|-----|-------------|-----------|
| **Base_Limpa** | ✅ Sim | Histórico de demanda: SKU | Ano | Mês | Demanda | Consumo |
| **Base_Dados** | ✅ Sim | Cadastro de SKUs com as 3 novas colunas de controle |
| Estatistica_SKU | Opcional | Estatísticas por SKU do arquivo original |
| Avaliacao_Modelo | Opcional | Avaliações originais para comparação de ganho |

**Colunas da aba Base_Dados:**

| Coluna | Função |
|--------|--------|
| `Classe de Material` | Código da classe — usado como filtro para a IA Out-of-Sample |
| `Descrição da Classe` | Texto descritivo — exibido junto com o código no filtro |
| `Situação` | Saneamento de SKU — apenas SKUs com valor **"1- NORMAL"** são processados pelo SONAR |

> ⚠️ SKUs com `Situação ≠ "1- NORMAL"` (ex: "2- INATIVO", "3- CANCELADO", "4- SUSPENSO") são **descartados automaticamente** antes de qualquer cálculo. O número de SKUs descartados aparece na barra lateral.
""")
        st.divider()

        st.markdown("### 🔬 Filtro por Classe no IA Out-of-Sample")
        st.markdown("""
O filtro de classe permite segmentar quais SKUs serão avaliados pelo processo Out-of-Sample da IA:

1. **Pesquise** a classe no campo de texto (filtra por código ou descrição)
2. **Selecione** uma ou mais classes no multiselect
3. O **badge de segmentação** exibe quantas classes e SKUs serão avaliados
4. Clique em **🔬 IA Out-of-Sample**

Sem seleção → todos os SKUs elegíveis são avaliados.
A segmentação usada fica registrada no painel OOS da Tab IA + Previsão.
""")
        st.divider()

        for nome_aba, objetivo, passos in [
            ("📊 Simulação Retrospectiva","Diagnóstico e benchmarking de 9 métodos",[
                "Analise a tabela resumo: veja quantos SKUs cada método consegue abaixo de 35% de WMAPE.",
                "Use o box plot para entender a dispersão — métodos com mediana baixa mas máximo alto têm SKUs problemáticos específicos.",
                "Na tabela mensal: selecione o mês fechado mais recente e compare Realizado × Prev. Original × Melhor Prev. SONAR por SKU.",
            ]),
            ("🔬 IA Out-of-Sample (sidebar)","Calibração do peso da IA por segmento",[
                "Selecione a(s) classe(s) de material desejadas no filtro da barra lateral.",
                "Observe o badge de segmentação — ele confirma quantos SKUs serão avaliados.",
                "Clique no botão e aguarde. Após concluir, rode o Pipeline Completo novamente para aplicar os pesos calibrados.",
                "O badge na tab IA + Previsão registra qual segmentação foi usada no último OOS.",
            ]),
            ("🤖 IA + Previsão","Previsão M+1/M+2/M+3 e análise OOS",[
                "A tabela de Horizonte 3 Meses usa os pesos automáticos calibrados pelo OOS.",
                "No painel OOS, um badge de segmentação exibe o filtro de classe usado na última execução.",
                "O OOS aplica automaticamente os pesos calibrados a todas as tabelas — não é mais necessário rodar o pipeline novamente.",
            ]),
            ("📋 Top 10 Piores WMAPE","Priorização nos SKUs críticos",[
                "Dois painéis: 12 meses (problemas estruturais) e 6 meses (problemas recentes).",
                "SKU aparece nos dois → problema estrutural. Só no de 6 meses → evento pontual recente.",
                "Apenas SKUs com situação Normal aparecem aqui (os demais foram descartados na carga).",
            ]),
        ]:
            with st.expander(f"{nome_aba} — {objetivo}", expanded=False):
                st.markdown(f"**Objetivo:** {objetivo}")
                for p in passos:
                    st.markdown(f"- {p}")

        st.divider()
        col_lim1, col_lim2 = st.columns(2)
        with col_lim1:
            st.warning("**Limitações do pipeline:**\n\n"
                "- SKUs com situação ≠ \"1- NORMAL\" são descartados automaticamente\n"
                "- SKUs com menos de 14 períodos não têm modelo de IA\n"
                "- Holt-Winters exige mínimo de 24 períodos\n"
                "- TriM-Heres requer 2 anos de histórico para resultado ideal\n"
                "- WMAPE in-sample da IA é otimista — use o OOS para calibração\n"
                "- O peso automático é definido em 3 faixas fixas (0/50/70%)")
        with col_lim2:
            st.success("**Boas práticas recomendadas:**\n\n"
                "- Mantenha Situação atualizada na Base_Dados mensalmente\n"
                "- Use o filtro de classe para segmentar o OOS por categoria estratégica\n"
                "- Após o OOS, todas as tabelas refletem os pesos calibrados automaticamente\n"
                "- Verifique o badge na sidebar: '🎯 Peso automático ativo: N SKUs'\n"
                "- Compare os painéis de 6 e 12 meses no Top 10 para classificar problemas\n"
                "- Combine a previsão do SONAR com o conhecimento de mercado da equipe")

    # ─────────────────────────────────────────────────────────
    # TAB 7: IMEDIATO
    # ─────────────────────────────────────────────────────────
    with tab7:
        st.subheader("🧭 Imediato — Copiloto do Planejador")
        st.caption("Tire dúvidas sobre SKUs e sobre as regras do SONAR. O assistente conhece "
                   "a metodologia da ferramenta e enxerga os dados já processados.")

        # ── Configuração da chave de API ─────────────────────
        with st.expander("⚙️ Configuração da chave de API", expanded=False):
            st.markdown(
                "A chave **não fica salva no código** — vale apenas para esta sessão do navegador. "
                "Para uso fixo, configure em *Settings → Secrets* do Streamlit como `ANTHROPIC_API_KEY`."
            )
            _key_secret = None
            try:
                _key_secret = st.secrets.get("ANTHROPIC_API_KEY", None)
            except Exception:
                _key_secret = None
            _key_input = st.text_input(
                "Chave de API da Anthropic", type="password",
                value="", placeholder="sk-ant-...",
                help="Começa com sk-ant-. Use a sua chave existente."
            )
            api_key = _key_input.strip() or _key_secret
            if _key_secret and not _key_input.strip():
                st.caption("🔑 Usando a chave configurada nos Secrets.")
            _modelo_cop = st.selectbox(
                "Modelo",
                options=["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
                index=0,
                help="Sonnet 4.6: melhor custo-benefício (recomendado). "
                     "Opus 4.8: máxima capacidade. Haiku 4.5: mais rápido e barato."
            )

        if 'imediato_msgs' not in st.session_state:
            st.session_state['imediato_msgs'] = []

        _df_ia_cop  = st.session_state.get('df_ia')
        _df_bt_cop  = st.session_state.get('df_backtest')
        _df_oos_cop = st.session_state.get('df_oos')

        if _df_ia_cop is None or len(_df_ia_cop) == 0:
            st.info("👆 Rode o **Pipeline Completo** primeiro — o Imediato precisa dos dados "
                    "processados para responder sobre os SKUs.")
        else:
            # ── Foco opcional em um SKU ───────────────────────
            col_cf1, col_cf2 = st.columns([1, 3])
            _skus_disp = ["(panorama geral)"] + [str(s) for s in sorted(_df_ia_cop['sku'].dropna().unique())]
            _sku_foco_sel = col_cf1.selectbox("SKU em foco", options=_skus_disp, key='imediato_sku_foco')
            _sku_foco = None if _sku_foco_sel == "(panorama geral)" else _sku_foco_sel

            col_cf2.markdown("**Perguntas rápidas:**")
            _qr = col_cf2.columns(3)
            _pergunta_rapida = None
            if _qr[0].button("Por que esse WMAPE?", use_container_width=True, disabled=(_sku_foco is None)):
                _pergunta_rapida = f"Explique por que o SKU {_sku_foco} tem esse WMAPE e o que fazer para melhorar."
            if _qr[1].button("Que ação tomar?", use_container_width=True):
                _pergunta_rapida = (f"Quais ações você recomenda para o SKU {_sku_foco}?"
                                    if _sku_foco else "Quais os SKUs mais críticos e que ações recomenda?")
            if _qr[2].button("IA ou estatístico?", use_container_width=True, disabled=(_sku_foco is None)):
                _pergunta_rapida = f"Para o SKU {_sku_foco}, devo confiar mais na IA ou no método estatístico? Por quê?"

            # ── Resumo executivo da rodada (briefing automático) ──
            if st.button("📋 Gerar resumo executivo da rodada", use_container_width=True,
                         help="Parecer pronto para compartilhar: qualidade geral, riscos, "
                              "IA vs. estatístico e ações prioritárias do mês."):
                _pergunta_rapida = (
                    "Gere um RESUMO EXECUTIVO desta rodada do SONAR, em formato pronto para "
                    "compartilhar com a liderança, cobrindo: "
                    "1) Qualidade geral das previsões (WMAPE mediano e leitura das faixas); "
                    "2) Principais riscos — os SKUs mais críticos e o provável motivo de cada um; "
                    "3) Balanço IA vs. estatístico com base no OOS; "
                    "4) De 3 a 5 ações prioritárias para o planejador neste mês. "
                    "Seja direto, use apenas os dados do contexto e destaque números-chave."
                )

            # ── Histórico do chat ─────────────────────────────
            for _m in st.session_state['imediato_msgs']:
                with st.chat_message(_m['role']):
                    st.markdown(_m['content'])

            _entrada = st.chat_input("Pergunte sobre um SKU ou sobre as regras do SONAR...")
            _pergunta = _entrada or _pergunta_rapida

            if _pergunta:
                if not api_key:
                    st.warning("⚠️ Insira a chave de API em **Configuração da chave de API** acima para conversar.")
                else:
                    st.session_state['imediato_msgs'].append({'role': 'user', 'content': _pergunta})
                    with st.chat_message("user"):
                        st.markdown(_pergunta)

                    # SKUs em foco: o selecionado no seletor + os citados na pergunta
                    _skus_validos = set(_df_ia_cop['sku'].apply(_norm_sku))
                    _skus_foco = []
                    if _sku_foco:
                        _skus_foco.append(_norm_sku(_sku_foco))
                    for _sk_det in _detectar_skus_na_pergunta(_pergunta, _skus_validos):
                        if _sk_det not in _skus_foco:
                            _skus_foco.append(_sk_det)

                    with st.chat_message("assistant"):
                        _ctx = _montar_contexto_dados(
                            _df_ia_cop, _df_bt_cop, _df_oos_cop, skus_foco=_skus_foco,
                            df_base=df_base, col_sku=col_sku, col_periodo=col_periodo,
                            col_ano=col_ano, col_demanda=col_demanda,
                        )
                        # envia só as últimas 8 mensagens, garantindo que comece em 'user'
                        _hist = st.session_state['imediato_msgs'][-8:]
                        while _hist and _hist[0]['role'] != 'user':
                            _hist = _hist[1:]
                        _resp = st.write_stream(
                            chamar_claude_stream(api_key, _hist, _ctx, modelo=_modelo_cop)
                        )
                    if not isinstance(_resp, str):
                        _resp = "".join(_resp)
                    st.session_state['imediato_msgs'].append({'role': 'assistant', 'content': _resp})

            if st.session_state['imediato_msgs']:
                if st.button("🗑️ Limpar conversa"):
                    st.session_state['imediato_msgs'] = []
                    st.rerun()

if __name__ == "__main__":
    main()
