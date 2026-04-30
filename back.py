import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import re
import warnings
from scipy.optimize import minimize
from brfunds import getFundsEarnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURAÇÃO DA PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Asset Allocator Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ── Paleta Corporativa Consistente ───────────────────────────────────────────
CORP = {
    "primary":    "#1A56DB",   # azul institucional
    "success":    "#0E9F6E",   # verde
    "warning":    "#E3A008",   # âmbar
    "danger":     "#E02424",   # vermelho
    "purple":     "#7E3AF2",   # roxo
    "neutral":    "#6B7280",   # cinza médio
    "bg_card":    "#FFFFFF",
    "bg_page":    "#F8FAFC",
    "border":     "#E2E8F0",
    "text_main":  "#0F172A",
    "text_sub":   "#64748B",
}

# ── CSS Premium estilo Bloomberg / XP Investimentos ──────────────────────────
st.markdown(f"""
<style>
  /* ── Fundo geral ──────────────────────────────── */
  .stApp {{ background-color: {CORP['bg_page']}; }}

  /* ── Metric Card ──────────────────────────────── */
  .metric-card {{
    background: {CORP['bg_card']};
    border: 1px solid {CORP['border']};
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    text-align: center;
    transition: box-shadow 0.2s ease;
  }}
  .metric-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.10); }}
  .metric-value {{
    font-size: 26px;
    font-weight: 700;
    color: {CORP['text_main']};
    letter-spacing: -0.5px;
    line-height: 1.2;
  }}
  .metric-value.positive {{ color: {CORP['success']}; }}
  .metric-value.negative {{ color: {CORP['danger']}; }}
  .metric-label {{
    font-size: 11px;
    font-weight: 600;
    color: {CORP['text_sub']};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 4px;
  }}
  .metric-accent {{
    height: 3px;
    border-radius: 2px;
    margin: 10px auto 0;
    width: 40px;
  }}

  /* ── Section header ───────────────────────────── */
  .section-header {{
    border-left: 4px solid {CORP['primary']};
    padding-left: 10px;
    margin: 24px 0 12px;
    font-size: 16px;
    font-weight: 700;
    color: {CORP['text_main']};
  }}

  /* ── Status badge ─────────────────────────────── */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
  }}
  .badge-ok  {{ background:#D1FAE5; color:#065F46; }}
  .badge-err {{ background:#FEE2E2; color:#991B1B; }}
  .badge-warn{{ background:#FEF3C7; color:#92400E; }}

  /* ── CNPJ fund tag ────────────────────────────── */
  .fund-tag {{
    display: inline-flex; align-items: center; gap: 6px;
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 6px; padding: 4px 10px;
    font-size: 12px; color: {CORP['primary']};
    margin: 2px;
  }}

  /* ── Tabs premium ─────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {CORP['bg_card']};
    border-radius: 10px 10px 0 0;
    padding: 6px 6px 0;
    border-bottom: 2px solid {CORP['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    color: {CORP['text_sub']};
    background: transparent;
    border: none;
    transition: color 0.15s;
  }}
  .stTabs [aria-selected="true"] {{
    color: {CORP['primary']};
    background: #EFF6FF !important;
    border-bottom: 3px solid {CORP['primary']} !important;
  }}

  /* ── Sidebar refinements ──────────────────────── */
  [data-testid="stSidebar"] {{
    background: {CORP['bg_card']};
    border-right: 1px solid {CORP['border']};
  }}
  [data-testid="stSidebar"] .stMarkdown h2 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {CORP['text_sub']};
    margin: 18px 0 6px;
  }}

  /* ── Weight total indicator ───────────────────── */
  .weight-total {{
    font-size: 20px; font-weight: 700; text-align: center;
    padding: 10px; border-radius: 8px; margin-top: 8px;
  }}
  .weight-ok  {{ background:#D1FAE5; color:#065F46; }}
  .weight-err {{ background:#FEE2E2; color:#991B1B; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 0b. UTILITÁRIOS DE PLOTLY — layout corporativo padronizado
# ══════════════════════════════════════════════════════════════════════════════
def corp_layout(**overrides) -> dict:
    """Retorna um dict de layout Plotly com padrão corporativo premium."""
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=CORP["text_main"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=CORP["border"],
            font_size=12,
            font_color=CORP["text_main"],
        ),
        legend=dict(
            orientation="h", y=1.06, x=0.5, xanchor="center",
            font_size=11, bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        margin=dict(t=70, b=40, l=50, r=20),
        xaxis=dict(
            showgrid=False,
            linecolor=CORP["border"],
            tickcolor=CORP["border"],
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor=CORP["border"],
            zeroline=False,
        ),
    )
    base.update(overrides)
    return base


def metric_card(label: str, value: str, accent_color: str = None, value_class: str = "") -> str:
    color = accent_color or CORP["primary"]
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-value {value_class}'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-accent' style='background:{color}'></div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNÇÕES DE DADOS (CVM / YFINANCE / BCB)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_cnpj(raw: str) -> str:
    """Remove formatação e retorna somente dígitos do CNPJ."""
    return re.sub(r"\D", "", str(raw).strip())


def _fmt_cnpj(digits: str) -> str:
    """Formata 14 dígitos como XX.XXX.XXX/XXXX-XX."""
    d = digits.zfill(14)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"


@st.cache_data(show_spinner=False, ttl=3600)
def resolve_fund_name(cnpj_digits: str) -> str:
    """
    Tenta obter o nome curto do fundo via brfunds.
    Retorna o CNPJ formatado como fallback se não conseguir.
    """
    try:
        fmt = _fmt_cnpj(cnpj_digits)
        sample_start = "01/01/24"
        sample_end   = datetime.today().strftime("%d/%m/%y")
        df = getFundsEarnings(fmt, start=sample_start, end=sample_end)
        if df is not None and not df.empty:
            col = df.columns[0]
            name = str(col).strip()
            # Remove formatação de CNPJ do nome retornado pela API
            name = re.sub(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", "", name).strip(" -_|")
            return name[:30] if name else _fmt_cnpj(cnpj_digits)
    except Exception:
        pass
    return _fmt_cnpj(cnpj_digits)


@st.cache_data(show_spinner=False, ttl=86400)
def get_fundos_cvm(cnpj_dict, start_date, end_date):
    """
    Versão otimizada para evitar o erro 'Nenhuma coluna numérica encontrada'.
    """
    try:
        # 1. Sanitização: brfunds exige CNPJs apenas com números (sem pontos ou barras)
        clean_cnpjs = [re.sub(r'\D', '', cnpj) for cnpj in cnpj_dict.values()]
        
        # Datas no formato que o brfunds espera (DD/MM/YY)
        start_str = start_date.strftime('%d/%m/%y')
        end_str = end_date.strftime('%d/%m/%y')
        
        # 2. Chamada da API
        df_cvm = getFundsEarnings(*clean_cnpjs, start=start_str, end=end_str)
        
        if df_cvm is None or df_cvm.empty:
            return pd.DataFrame()

        # 3. Tratamento do erro 'Nenhuma coluna numérica encontrada'
        # Forçamos a seleção apenas de colunas que contêm dados financeiros
        df_cvm = df_cvm.select_dtypes(include=[np.number])
        
        if df_cvm.empty:
            return pd.DataFrame()

        # 4. Mapeamento Inteligente
        # A CVM pode retornar o CNPJ ou o Nome Empresarial Completo. 
        # Esta lógica faz o "de-para" independente do formato retornado.
        new_cols = {}
        for col in df_cvm.columns:
            col_str = str(col).upper()
            # Tenta encontrar qual fundo do seu dicionário pertence a esta coluna
            for name, cnpj in cnpj_dict.items():
                cnpj_only_numbers = re.sub(r'\D', '', cnpj)
                if cnpj_only_numbers in col_str or name.upper() in col_str:
                    new_cols[col] = name
                    break
        
        df_cvm = df_cvm.rename(columns=new_cols)
        
        # Mantém apenas as colunas que conseguimos renomear com sucesso
        valid_cols = [c for c in new_cols.values() if c in df_cvm.columns]
        df_cvm = df_cvm[valid_cols]
        
        # 5. Transformação em Retorno Mensal
        df_cvm.index = pd.to_datetime(df_cvm.index)
        # O brfunds retorna (1 + r), então subtraímos 1 para ter o retorno líquido
        df_ret = (df_cvm).resample('ME').last().pct_change().dropna(how='all')
        
        return df_ret

    except Exception as e:
        st.error(f"Erro ao processar dados da CVM: {e}")
        return pd.DataFrame()

    # ── Lógica Real Investor (mantida intacta) ────────────────────────────────
    legacy_ri = {
        "2012-06": 0.0035, "2012-07": 0.0483, "2012-08": 0.0247, "2012-09": 0.0385,
        "2012-10": 0.0401, "2012-11": 0.0210, "2012-12": 0.0463, "2013-01": 0.0270,
        "2013-02": -0.0150, "2013-03": -0.0190, "2013-04": 0.0194, "2013-05": 0.0232,
        "2013-06": -0.0898, "2013-07": 0.0076, "2013-08": 0.0116, "2013-09": 0.0426,
        "2013-10": 0.0346, "2013-11": -0.0135, "2013-12": -0.0125, "2014-01": -0.0384,
        "2014-02": 0.0122, "2014-03": 0.0610, "2014-04": 0.0315, "2014-05": 0.0132,
        "2014-06": 0.0378, "2014-07": 0.0203, "2014-08": 0.0760, "2014-09": -0.0543,
        "2014-10": 0.0306, "2014-11": 0.0253, "2014-12": -0.0354, "2015-01": -0.0575,
        "2015-02": 0.0631, "2015-03": -0.0163, "2015-04": 0.0768, "2015-05": -0.0441,
        "2015-06": 0.0044, "2015-07": -0.0243, "2015-08": -0.0531, "2015-09": -0.0185,
        "2015-10": 0.0366, "2015-11": -0.0224, "2015-12": -0.0128, "2016-01": -0.0427,
        "2016-02": 0.0573, "2016-03": 0.1190, "2016-04": 0.0747, "2016-05": -0.0118,
        "2016-06": 0.0541, "2016-07": 0.0863, "2016-08": 0.0205, "2016-09": 0.0076,
        "2016-10": 0.0754, "2016-11": -0.0454, "2016-12": 0.0152, "2017-01": 0.0607,
        "2017-02": 0.0487, "2017-03": 0.0016, "2017-04": 0.0154, "2017-05": -0.0229,
        "2017-06": 0.0118, "2017-07": 0.0558, "2017-08": 0.0620, "2017-09": 0.0519,
        "2017-10": 0.0119, "2017-11": -0.0250, "2017-12": 0.0494,
    }
    s_legacy = pd.Series(legacy_ri, name="Real Investor")
    s_legacy.index = pd.to_datetime(s_legacy.index).to_period("M").to_timestamp("M")

    # Injeta dados legados do Real Investor se ele for um dos fundos
    for col_name in df_out.columns:
        if "real investor" in col_name.lower() or "real_investor" in col_name.lower():
            df_out[col_name] = df_out[col_name].combine_first(s_legacy)

    return df_out


@st.cache_data
def get_cdi_data(start_date, end_date):
    """Busca o CDI mensal (Série 4391) na API do BCB."""
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df.set_index("data", inplace=True)
        df["valor"] = df["valor"].astype(float) / 100.0
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        cdi_series = df.loc[mask, "valor"].resample("ME").last()
        cdi_series.name = "CDI"
        return cdi_series
    except Exception as e:
        st.error(f"Erro ao baixar dados do CDI (BCB): {e}")
        return pd.Series(dtype="float64")


@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    processed = []
    for t in tickers:
        t = t.strip().upper()
        processed.append(f"{t}.SA" if "." not in t and any(c.isdigit() for c in t) else t)
    try:
        data = yf.download(processed, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            try:
                prices = data.xs("Adj Close", level=0, axis=1)
            except KeyError:
                prices = data.xs("Close", level=0, axis=1)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=processed[0])
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(-1)
        monthly_prices = prices.resample("ME").last()
        returns = monthly_prices.pct_change()
        returns.columns = [str(c).replace(".SA", "") for c in returns.columns]
        return returns
    except Exception as e:
        st.error(f"Erro no download (YFinance): {e}")
        return pd.DataFrame()


@st.cache_data
def get_benchmark_data(start_date, end_date):
    try:
        ibov = yf.download("BOVA11.SA", start=start_date, end=end_date, progress=False)
        if ibov.empty:
            return pd.Series(dtype="float64")
        prices = ibov["Adj Close"] if "Adj Close" in ibov.columns else ibov.iloc[:, 0]
        returns = prices.resample("ME").last().pct_change().dropna()
        returns.name = "Ibovespa"
        return returns
    except Exception as e:
        st.error(f"Erro no benchmark: {e}")
        return pd.Series(dtype="float64")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LÓGICA DE CÁLCULO — NÃO ALTERADA
# ══════════════════════════════════════════════════════════════════════════════
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    if not available_assets:
        return None, None, None
    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum()
    portfolio_pure_idx = [100.0]
    monthly_returns = []
    portfolio_wealth = [initial_cap]
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    for i in range(len(dates)):
        r_t = asset_returns_np[i]
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        new_idx = portfolio_pure_idx[-1] * (1 + port_ret)
        portfolio_pure_idx.append(new_idx)
        new_wealth = (portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution
        portfolio_wealth.append(new_wealth)
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        is_rebalance_time = (rebalance_freq == "Mensal") or \
                            (rebalance_freq == "Anual" and dates[i].month == 12)
        if is_rebalance_time:
            current_weights = active_weights.copy()
    portfolio_pure_series  = pd.Series(portfolio_pure_idx[1:], index=dates)
    portfolio_wealth_series = pd.Series(portfolio_wealth[1:], index=dates)
    monthly_returns_series  = pd.Series(monthly_returns, index=dates)
    monthly_returns_series.name = "Portfólio"
    return portfolio_pure_series, portfolio_wealth_series, monthly_returns_series


def create_monthly_heatmap(returns_series):
    df_ret = returns_series.to_frame(name="Retorno")
    df_ret["Ano"] = df_ret.index.year
    df_ret["Mes"] = df_ret.index.month
    pivot = df_ret.pivot(index="Ano", columns="Mes", values="Retorno")
    pivot["YTD"] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    month_map = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                 7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    pivot.rename(columns=month_map, inplace=True)
    return pivot


def run_walkforward_optimization(returns_df, rf_monthly_avg, window_months=6):
    n_rows, n_assets = returns_df.shape
    rf_ann = rf_monthly_avg * 12
    weights_list, window_info = [], []
    for start_idx in range(0, n_rows - window_months + 1, window_months):
        end_idx     = start_idx + window_months
        window_data = returns_df.iloc[start_idx:end_idx]
        if len(window_data) < window_months:
            continue
        mu_w    = window_data.mean().values * 12
        Sigma_w = window_data.cov().values   * 12

        def _port_vol_wf(w, S=Sigma_w):
            return float(np.sqrt(np.maximum(w @ S @ w, 0.0)))

        def _neg_sharpe_wf(w, mu=mu_w, S=Sigma_w, rf=rf_ann):
            r = float(np.dot(w, mu))
            v = float(np.sqrt(np.maximum(w @ S @ w, 0.0)))
            return -(r - rf) / v if v > 1e-9 else 0.0

        w0     = np.full(n_assets, 1.0 / n_assets)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        try:
            res = minimize(
                _neg_sharpe_wf, w0, method="SLSQP",
                bounds=bounds, constraints=[eq_sum],
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if res.success and np.isfinite(res.x).all():
                weights_list.append(np.clip(res.x, 0, 1))
                window_info.append((window_data.index[0], window_data.index[-1]))
        except Exception:
            continue
    return weights_list, window_info


def build_scenario_portfolio(weights_list, asset_names, method="median"):
    if not weights_list:
        return np.array([]), pd.DataFrame()
    weights_matrix = np.vstack(weights_list)
    df_windows     = pd.DataFrame(weights_matrix, columns=asset_names)
    w_raw    = np.median(weights_matrix, axis=0) if method == "median" \
               else np.mean(weights_matrix, axis=0)
    total    = w_raw.sum()
    w_cenarios = w_raw / total if total > 1e-9 else w_raw
    return w_cenarios, df_windows


def compute_scenario_metrics(returns_df, weights, rf_monthly_series):
    port_monthly = pd.Series(returns_df.values @ weights, index=returns_df.index)
    cdi_aligned  = rf_monthly_series.reindex(returns_df.index).fillna(0)
    excess   = port_monthly - cdi_aligned
    std_m    = port_monthly.std()
    vol_ann  = std_m * np.sqrt(12)
    ret_ann  = port_monthly.mean() * 12
    sharpe   = (excess.mean() / std_m) * np.sqrt(12) if std_m > 1e-9 else 0.0
    return sharpe, vol_ann, ret_ann


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR — PARÂMETROS E COMPOSIÇÃO DA CARTEIRA
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo / Título ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:12px 0 4px'>
      <span style='font-size:28px'>📊</span><br>
      <span style='font-size:17px; font-weight:800; color:#0F172A; letter-spacing:-0.3px'>
        Asset Allocator Pro
      </span><br>
      <span style='font-size:10px; color:#94A3B8; text-transform:uppercase; letter-spacing:1px'>
        Portfolio Analytics Engine
      </span>
    </div>
    <hr style='border:none; border-top:1px solid #E2E8F0; margin:12px 0'>
    """, unsafe_allow_html=True)

    # ── Período ───────────────────────────────────────────────────────────────
    st.markdown("## 📅 PERÍODO")
    min_date = datetime(2012, 1, 1)
    max_date = datetime.today()
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início",  datetime(2018, 1, 1), min_value=min_date, max_value=max_date)
    end_date   = col_d2.date_input("Fim", max_date, min_value=min_date, max_value=max_date)

    # ── Parâmetros Gerais ─────────────────────────────────────────────────────
    st.markdown("## ⚙️ CONFIGURAÇÕES")
    rebalance_freq      = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=100_000.0, step=1_000.0, format="%.2f")
    aporte_mensal        = st.number_input("Aporte Mensal (R$)",        value=1_000.0,   step=100.0,   format="%.2f")

    # ── Seleção de Ativos de Mercado ──────────────────────────────────────────
    st.markdown("## 🏦 ATIVOS DE MERCADO")
    with st.expander("Editar tickers", expanded=False):
        default_stocks = "EGIE3, ITUB3, PSSA3, WEGE3, CXSE3, SBSP3, TAEE3, VIVT3, CPFE3, SAPR3, BBAS3, PRIO3, TOTS3, BPAC3, ALUP3, BMOB3"
        default_fiis   = "ALZR11, BRCO11, BTLG11, HGLG11, HGRE11, HGRU11, KNCR11, KNRI11, LVBI11, MXRF11, PMLL11, XPLG11, XPML11"
        default_etfs   = "IVVB11"
        stocks_input = st.text_area("Ações BR", default_stocks, height=80)
        fiis_input   = st.text_area("FIIs",     default_fiis,   height=80)
        etfs_input   = st.text_area("ETFs",     default_etfs,   height=40)

    st.markdown("## ⚖️ PESOS DOS ATIVOS (%)")

    # Sliders para classes de ativos
    w_stocks = st.slider("📈 Ações",  0, 100, 35, key="w_stocks")
    w_fiis   = st.slider("🏢 FIIs",   0, 100,  0, key="w_fiis")
    w_etfs   = st.slider("🌍 ETFs",   0, 100, 20, key="w_etfs")
    w_cdi    = st.slider("💰 CDI",    0, 100,  0, key="w_cdi")

    # ── Seleção Dinâmica de Fundos CVM ────────────────────────────────────────
    st.markdown("""
    <hr style='border:none; border-top:1px solid #E2E8F0; margin:10px 0'>
    <div style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:4px'>
      🏛️ FUNDOS CVM — SELEÇÃO DINÂMICA
    </div>
    <div style='font-size:11px; color:#64748B; margin-bottom:8px'>
      Insira um CNPJ por linha (apenas números ou formatado).
    </div>
    """, unsafe_allow_html=True)

    DEFAULT_CNPJS = (
        "22.232.927/0001-90\n"   # Tarpon GT
        "32.073.525/0001-43\n"   # Absolute Pace
        "15.334.585/0001-53\n"   # SPX Patriot
        "10.500.884/0001-05\n"   # Real Investor
        "17.400.251/0001-66"     # Organon FIC FIA
    )

    cnpj_raw_input = st.text_area(
        "CNPJs dos Fundos",
        value=DEFAULT_CNPJS,
        height=130,
        help="Um CNPJ por linha. Pode ser formatado (XX.XXX.XXX/XXXX-XX) ou somente dígitos.",
        placeholder="22.232.927/0001-90\n32.073.525/0001-43",
    )

    # ── Parse e Validação dos CNPJs ───────────────────────────────────────────
    raw_lines   = [l.strip() for l in cnpj_raw_input.strip().splitlines() if l.strip()]
    valid_cnpjs = []
    invalid_entries = []

    for line in raw_lines:
        digits = _clean_cnpj(line)
        if len(digits) == 14:
            valid_cnpjs.append(digits)
        else:
            invalid_entries.append(line)

    if invalid_entries:
        st.warning(f"⚠️ Entradas ignoradas (CNPJ inválido): `{'`, `'.join(invalid_entries)}`")

    # ── Resolve nomes dos fundos (com cache) ──────────────────────────────────
    fund_labels = []
    if valid_cnpjs:
        with st.spinner("Identificando fundos..."):
            for d in valid_cnpjs:
                fund_labels.append(resolve_fund_name(d))

    # ── Inputs de Peso por Fundo (dinâmico) ───────────────────────────────────
    fund_weights = {}
    DEFAULT_FUND_WEIGHTS = [10, 25, 0, 0, 10]   # pesos padrão para os CNPJs padrão

    if valid_cnpjs:
        st.markdown(
            "<div style='font-size:11px;color:#64748B;margin:6px 0 4px'>Pesos dos fundos (%):</div>",
            unsafe_allow_html=True,
        )
        for i, (digits, label) in enumerate(zip(valid_cnpjs, fund_labels)):
            short = label[:22] + "…" if len(label) > 22 else label
            default_w = DEFAULT_FUND_WEIGHTS[i] if i < len(DEFAULT_FUND_WEIGHTS) else 0
            w = st.number_input(
                f"{short}",
                min_value=0, max_value=100,
                value=default_w,
                step=1,
                key=f"fund_w_{digits}",
                help=f"CNPJ: {_fmt_cnpj(digits)}",
            )
            fund_weights[label] = w

    # ── Totalizador de Pesos ──────────────────────────────────────────────────
    fund_total = sum(fund_weights.values())
    total_w    = w_stocks + w_fiis + w_etfs + w_cdi + fund_total

    color_cls = "weight-ok" if total_w == 100 else "weight-err"
    icon      = "✅" if total_w == 100 else "⚠️"
    st.markdown(
        f"<div class='weight-total {color_cls}'>{icon} Total: {total_w}%</div>",
        unsafe_allow_html=True,
    )
    if total_w != 100:
        st.caption("Os pesos serão normalizados automaticamente para somar 100%.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(
        "<h1 style='font-size:28px;font-weight:800;color:#0F172A;margin-bottom:0'>"
        "📊 Asset Allocator Pro</h1>"
        "<p style='color:#64748B;margin-top:2px;font-size:13px'>"
        f"Período: <b>{start_date.strftime('%d/%m/%Y')}</b> → <b>{end_date.strftime('%d/%m/%Y')}</b> "
        f"| Rebalanceamento: <b>{rebalance_freq}</b></p>",
        unsafe_allow_html=True,
    )
with header_col2:
    # Status dos fundos selecionados
    if valid_cnpjs:
        st.markdown(
            f"<div class='badge badge-ok' style='margin-top:20px'>"
            f"🏛️ {len(valid_cnpjs)} fundo(s) CVM</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='badge badge-warn' style='margin-top:20px'>Nenhum fundo CVM</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<hr style='border:none;border-top:1px solid #E2E8F0;margin:4px 0 16px'>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSOLIDAÇÃO DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
stock_list = [x.strip() for x in stocks_input.split(",") if x.strip()]
fii_list   = [x.strip() for x in fiis_input.split(",")   if x.strip()]
etf_list   = [x.strip() for x in etfs_input.split(",")   if x.strip()]

with st.spinner("📡 Carregando dados de mercado, taxas e fundos CVM..."):
    api_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=45)

    df_stocks = get_market_data(stock_list, api_start_date, end_date)
    df_fiis   = get_market_data(fii_list,   api_start_date, end_date)
    df_etfs   = get_market_data(etf_list,   api_start_date, end_date)
    ibov_ret  = get_benchmark_data(api_start_date, end_date)
    cdi_ret   = get_cdi_data(api_start_date, end_date)

    # Busca fundos CVM de forma dinâmica
    df_funds = pd.DataFrame()
    if valid_cnpjs:
        df_funds = get_fundos_cvm(
            tuple(valid_cnpjs),
            tuple(fund_labels),
            api_start_date,
            end_date,
        )

# ── Montar Master DataFrame ───────────────────────────────────────────────────
all_indices = []
for d in [df_stocks, df_fiis, df_etfs, cdi_ret, df_funds, ibov_ret]:
    if d is not None and not (isinstance(d, pd.DataFrame) and d.empty) \
       and not (isinstance(d, pd.Series) and d.empty):
        all_indices.append(d.index)

if not all_indices:
    st.error("❌ Nenhum dado retornado das APIs. Verifique sua conexão ou os ativos selecionados.")
    st.stop()

all_dates = all_indices[0]
for idx in all_indices[1:]:
    all_dates = all_dates.union(idx)
all_dates = pd.to_datetime(all_dates).sort_values()

master_df = pd.DataFrame(index=all_dates)

if not df_stocks.empty: master_df["Ações Consolidadas"] = df_stocks.mean(axis=1).reindex(master_df.index)
if not df_fiis.empty:   master_df["FIIs Consolidados"]  = df_fiis.mean(axis=1).reindex(master_df.index)
if not df_etfs.empty:   master_df["ETFs Consolidados"]  = df_etfs.mean(axis=1).reindex(master_df.index)
master_df["CDI"] = cdi_ret.reindex(master_df.index)

# Integração segura dos fundos CVM (lista dinâmica)
for label in fund_labels:
    if not df_funds.empty and label in df_funds.columns:
        col_data = df_funds[label]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        master_df[label] = col_data.reindex(master_df.index)
    else:
        if label in fund_weights and fund_weights[label] > 0:
            st.warning(f"⚠️ Fundo **{label}** não pôde ser integrado ao portfólio. Será ignorado.")

# Filtro temporal e limpeza
mask      = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
master_df = master_df.loc[mask].fillna(0)
ibov_ret       = ibov_ret.reindex(master_df.index).fillna(0)
cdi_ret_series = cdi_ret.reindex(master_df.index).fillna(0)

# ── Dicionário de Pesos ───────────────────────────────────────────────────────
weights = {
    "Ações Consolidadas": w_stocks / 100,
    "FIIs Consolidados":  w_fiis   / 100,
    "ETFs Consolidados":  w_etfs   / 100,
    "CDI":                w_cdi    / 100,
}
for label, w in fund_weights.items():
    if label in master_df.columns:
        weights[label] = w / 100

# ══════════════════════════════════════════════════════════════════════════════
# 6. BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is None:
    st.warning("⚠️ Nenhum ativo com peso > 0 foi encontrado nos dados. Ajuste a carteira.")
    st.stop()

# ── Métricas Principais ───────────────────────────────────────────────────────
cdi_ret_series = cdi_ret.reindex(port_ret.index).fillna(0)
cdi_accum  = (1 + cdi_ret_series).cumprod() * 100
ibov_accum = (1 + ibov_ret).cumprod() * 100

total_ret = (port_pure.iloc[-1] / 100) - 1
years     = len(port_ret) / 12
cagr      = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
vol       = port_ret.std() * np.sqrt(12)

excess_returns = port_ret - cdi_ret_series
sharpe = (excess_returns.mean() / port_ret.std()) * np.sqrt(12) if port_ret.std() > 0 else 0

cum_ret  = (1 + port_ret).cumprod()
peak     = cum_ret.cummax()
dd_series = (cum_ret - peak) / peak
max_dd   = dd_series.min()

# ══════════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD PRINCIPAL — METRIC CARDS + TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='section-header'>📐 Performance Overview</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, "Retorno Total",     f"{total_ret:.1%}",  CORP["primary"],  "positive" if total_ret >= 0 else "negative"),
    (c2, "CAGR (a.a.)",       f"{cagr:.1%}",        CORP["success"],  "positive" if cagr >= 0 else "negative"),
    (c3, "Volatilidade",      f"{vol:.1%}",          CORP["warning"],  ""),
    (c4, "Sharpe vs CDI",     f"{sharpe:.2f}",       CORP["primary"],  "positive" if sharpe >= 1 else ("negative" if sharpe < 0 else "")),
    (c5, "Max Drawdown",      f"{max_dd:.1%}",       CORP["danger"],   "negative"),
    (c6, "Período (meses)",   f"{len(port_ret)}",    CORP["neutral"],  ""),
]
for col, label, value, accent, cls in cards:
    col.markdown(metric_card(label, value, accent, cls), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_perf, tab_risk, tab_month, tab_patr, tab_proj, tab_ef = st.tabs([
    "📈 Rentabilidade",
    "🛡️ Risco",
    "📅 Retornos Mensais",
    "💰 Patrimônio",
    "🔮 Projeções",
    "🎯 Fronteira Eficiente",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RENTABILIDADE COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("<div class='section-header'>Evolução Acumulada (Base 100)</div>", unsafe_allow_html=True)

    df_chart = pd.DataFrame({
        "Portfólio":     port_pure,
        "Ibovespa":      ibov_accum,
        "CDI Real (BCB)": cdi_accum,
    })

    fig = go.Figure()
    traces = [
        ("Portfólio",      port_pure.values,    CORP["primary"],  3.0, "solid"),
        ("Ibovespa",       ibov_accum.values,   CORP["warning"],  2.0, "dot"),
        ("CDI Real (BCB)", cdi_accum.values,    CORP["success"],  1.8, "dash"),
    ]
    for name, y, color, width, dash in traces:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=y, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b/%Y}}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(**corp_layout(
        title=dict(text="Comparativo de Rentabilidade Acumulada", font_size=14),
        yaxis=dict(title="Índice (Base 100)", gridcolor="#F1F5F9", zeroline=False),
        height=420,
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Tabela comparativa de rentabilidade
    st.markdown("<div class='section-header'>Rentabilidade por Período</div>", unsafe_allow_html=True)
    periods = {"12M": 12, "24M": 24, "36M": 36, "48M": 48, "Início": len(port_ret)}
    rows = {}
    for label_p, n in periods.items():
        if len(port_ret) >= n:
            p_ret   = (1 + port_ret.tail(n)).prod() - 1
            i_ret   = (1 + ibov_ret.tail(n)).prod()  - 1
            c_ret   = (1 + cdi_ret_series.tail(n)).prod() - 1
            rows[label_p] = {
                "Portfólio": f"{p_ret:.2%}",
                "Ibovespa":  f"{i_ret:.2%}",
                "CDI":       f"{c_ret:.2%}",
                "Alpha vs CDI": f"{p_ret - c_ret:+.2%}",
            }
    if rows:
        st.dataframe(pd.DataFrame(rows).T.style
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
            use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISE DE RISCO
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("<div class='section-header'>Drawdown Submarino</div>", unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values * 100,
            mode="lines", fill="tozeroy", name="Drawdown",
            line=dict(color=CORP["danger"], width=1.5),
            fillcolor="rgba(224,36,36,0.12)",
            hovertemplate="%{x|%b/%Y}: %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(**corp_layout(
            yaxis=dict(title="Drawdown (%)", gridcolor="#F1F5F9", zeroline=False),
            height=300, margin=dict(t=40, b=30, l=50, r=10),
            legend=dict(orientation="h", y=1.1),
        ))
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_r2:
        st.markdown("<div class='section-header'>Volatilidade Móvel (12M)</div>", unsafe_allow_html=True)
        rolling_vol = port_ret.rolling(12).std() * np.sqrt(12) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values,
            mode="lines", name="Vol. 12M",
            line=dict(color=CORP["warning"], width=2),
            hovertemplate="%{x|%b/%Y}: %{y:.2f}%<extra></extra>",
        ))
        fig_vol.update_layout(**corp_layout(
            yaxis=dict(title="Volatilidade (%)", gridcolor="#F1F5F9", zeroline=False),
            height=300, margin=dict(t=40, b=30, l=50, r=10),
            legend=dict(orientation="h", y=1.1),
        ))
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<div class='section-header'>Estatísticas de Risco Detalhadas</div>", unsafe_allow_html=True)

    months_pos  = (port_ret > 0).sum()
    months_neg  = (port_ret < 0).sum()
    best_month  = port_ret.max()
    worst_month = port_ret.min()
    calmar      = cagr / abs(max_dd) if max_dd != 0 else 0
    sortino_exc = port_ret[port_ret < 0].std() * np.sqrt(12)
    sortino     = (cagr - cdi_ret_series.mean() * 12) / sortino_exc if sortino_exc > 0 else 0

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    stat_cards = [
        (s1, "Meses Positivos", f"{months_pos} ({months_pos/len(port_ret):.0%})", CORP["success"], ""),
        (s2, "Meses Negativos", f"{months_neg} ({months_neg/len(port_ret):.0%})", CORP["danger"],  ""),
        (s3, "Melhor Mês",      f"{best_month:.2%}",  CORP["success"], "positive"),
        (s4, "Pior Mês",        f"{worst_month:.2%}", CORP["danger"],  "negative"),
        (s5, "Ratio de Calmar", f"{calmar:.2f}",       CORP["primary"], ""),
        (s6, "Ratio de Sortino",f"{sortino:.2f}",      CORP["purple"],  ""),
    ]
    for col, label, value, accent, cls in stat_cards:
        col.markdown(metric_card(label, value, accent, cls), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RETORNOS MENSAIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_month:
    st.markdown("<div class='section-header'>Tabela de Rentabilidade — Heatmap</div>", unsafe_allow_html=True)
    heatmap_data = create_monthly_heatmap(port_ret)
    st.dataframe(
        heatmap_data.style
            .format("{:.2%}")
            .background_gradient(cmap="RdYlGn", vmin=-0.05, vmax=0.05, axis=None)
            .highlight_null(color="white"),
        use_container_width=True, height=400,
    )

    st.markdown("<div class='section-header'>Índice de Sharpe por Janela</div>", unsafe_allow_html=True)
    sharpe_periods = {"12M": 12, "24M": 24, "48M": 48, "60M": 60, "Início": len(port_ret)}
    sharpe_results = {}
    for label_s, months in sharpe_periods.items():
        if len(port_ret) >= months:
            sub_p = port_ret.tail(months)
            sub_c = cdi_ret_series.tail(months)
            v     = sub_p.std()
            sharpe_results[label_s] = ((sub_p - sub_c).mean() / v) * np.sqrt(12) if v > 0 else 0.0
        else:
            sharpe_results[label_s] = None

    df_sharpe_table = pd.DataFrame([sharpe_results], index=["Índice de Sharpe"])
    st.dataframe(
        df_sharpe_table.style.format("{:.2f}", na_rep="-")
            .background_gradient(cmap="Blues", axis=1, vmin=0, vmax=2),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVOLUÇÃO PATRIMONIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_patr:
    st.markdown("<div class='section-header'>Crescimento Patrimonial</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([3, 1])

    with col_p1:
        fig_wealth = go.Figure()
        fig_wealth.add_trace(go.Scatter(
            x=port_wealth.index, y=port_wealth.values,
            mode="lines", name="Patrimônio",
            fill="tozeroy",
            line=dict(color=CORP["success"], width=2.5),
            fillcolor="rgba(14,159,110,0.10)",
            hovertemplate="%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))
        fig_wealth.update_layout(**corp_layout(
            title=dict(text="Crescimento Patrimonial (Cotas + Aportes)", font_size=14),
            yaxis=dict(title="Saldo (R$)", tickformat=",.0f", gridcolor="#F1F5F9"),
            height=380,
        ))
        st.plotly_chart(fig_wealth, use_container_width=True)

    with col_p2:
        final_val      = port_wealth.iloc[-1]
        total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
        profit_loss    = final_val - total_invested
        roi_pct        = (final_val / total_invested - 1)

        for label_m, value_m, delta_m, inv_d in [
            ("Saldo Final",      f"R$ {final_val:,.0f}",      None, False),
            ("Total Investido",  f"R$ {total_invested:,.0f}", None, False),
            ("Lucro / Prejuízo", f"R$ {profit_loss:,.0f}",    f"{roi_pct:.1%}", False),
        ]:
            if delta_m:
                st.metric(label_m, value_m, delta=delta_m, delta_color="normal")
            else:
                st.metric(label_m, value_m)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PROJEÇÕES (MONTE CARLO)
# ══════════════════════════════════════════════════════════════════════════════
with tab_proj:
    st.markdown("<div class='section-header'>Projeção de Cenários — Próximos 36 Meses</div>", unsafe_allow_html=True)

    mu       = port_ret.mean()
    sigma    = port_ret.std()
    N_MONTHS = 36
    N_SIM    = 20_000
    saldo_t0 = port_wealth.iloc[-1]
    last_date = port_wealth.index[-1]

    np.random.seed(42)
    paths = np.empty((N_SIM, N_MONTHS + 1))
    paths[:, 0] = saldo_t0
    rand_returns = np.random.normal(mu, sigma, size=(N_SIM, N_MONTHS))
    for t in range(1, N_MONTHS + 1):
        paths[:, t] = paths[:, t - 1] * (1 + rand_returns[:, t - 1]) + aporte_mensal

    p_otimista   = np.percentile(paths, 95, axis=0)
    p_neutro     = np.percentile(paths, 50, axis=0)
    p_pessimista = np.percentile(paths, 5,  axis=0)

    hist_tail    = port_wealth.tail(12)
    future_dates = pd.date_range(start=last_date, periods=N_MONTHS + 1, freq="ME")[1:]
    proj_dates   = [last_date] + list(future_dates)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=list(proj_dates) + list(reversed(proj_dates)),
        y=list(p_otimista) + list(reversed(p_pessimista)),
        fill="toself", fillcolor="rgba(26,86,219,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="Intervalo P5–P95", hoverinfo="skip",
    ))
    fig_proj.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values, mode="lines", name="Histórico Real",
        line=dict(color=CORP["text_main"], width=3),
        hovertemplate="<b>Histórico</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
    ))
    fig_proj.add_trace(go.Scatter(
        x=proj_dates, y=p_otimista, mode="lines", name="Otimista (P95)",
        line=dict(color=CORP["success"], width=2.2, dash="dash"),
        hovertemplate="<b>Otimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
    ))
    fig_proj.add_trace(go.Scatter(
        x=proj_dates, y=p_neutro, mode="lines", name="Neutro (P50)",
        line=dict(color=CORP["primary"], width=2.2, dash="dot"),
        hovertemplate="<b>Neutro</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
    ))
    fig_proj.add_trace(go.Scatter(
        x=proj_dates, y=p_pessimista, mode="lines", name="Pessimista (P5)",
        line=dict(color=CORP["danger"], width=2.2, dash="dash"),
        hovertemplate="<b>Pessimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
    ))
    fig_proj.add_vline(x=last_date, line_width=1.2, line_dash="dot", line_color=CORP["neutral"])

    fig_proj.update_layout(**corp_layout(
        title=dict(
            text=f"Monte Carlo — {N_SIM:,} simulações | µ={mu:.2%}/mês | σ={sigma:.2%}/mês | Aporte R$ {aporte_mensal:,.0f}/mês",
            font_size=12,
        ),
        yaxis=dict(title="Saldo (R$)", tickformat=",.0f", gridcolor="#F1F5F9"),
        height=430, margin=dict(t=80, b=40),
    ))
    st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("<div class='section-header'>Saldo Final Projetado em 36 Meses</div>", unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    pc1.markdown(metric_card("🟢 Otimista (P95)", f"R$ {p_otimista[-1]:,.0f}", CORP["success"]), unsafe_allow_html=True)
    pc2.markdown(metric_card("🔵 Neutro (P50)",   f"R$ {p_neutro[-1]:,.0f}",   CORP["primary"]), unsafe_allow_html=True)
    pc3.markdown(metric_card("🔴 Pessimista (P5)",f"R$ {p_pessimista[-1]:,.0f}",CORP["danger"]),  unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FRONTEIRA EFICIENTE + WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_ef:
    st.markdown("<div class='section-header'>🎯 Fronteira Eficiente de Markowitz</div>", unsafe_allow_html=True)

    active_assets = [a for a, w in weights.items() if w > 0 and a in master_df.columns]

    if len(active_assets) < 2:
        st.warning("⚠️ A Fronteira Eficiente requer pelo menos **2 ativos com peso > 0**.")
    else:
        returns_ef = master_df[active_assets].replace(0, np.nan).dropna(how="all").fillna(0)
        mu_vec   = returns_ef.mean() * 12
        Sigma    = returns_ef.cov() * 12
        rf_rate  = cdi_ret_series.mean() * 12
        n_assets = len(active_assets)
        Sigma_np = Sigma.values
        mu_np    = mu_vec.values

        def port_return(w): return float(np.dot(w, mu_np))
        def port_vol(w):    return float(np.sqrt(w @ Sigma_np @ w))
        def neg_sharpe(w):
            r, v = port_return(w), port_vol(w)
            return -(r - rf_rate) / v if v > 1e-9 else 0.0

        w0     = np.full(n_assets, 1.0 / n_assets)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res_minvol = minimize(port_vol,    w0, method="SLSQP", bounds=bounds, constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 1000})
        res_maxsh  = minimize(neg_sharpe,  w0, method="SLSQP", bounds=bounds, constraints=[eq_sum], options={"ftol": 1e-12, "maxiter": 1000})
        w_minvol, w_maxsh = res_minvol.x, res_maxsh.x

        ret_minvol, vol_minvol = port_return(w_minvol), port_vol(w_minvol)
        ret_maxsh,  vol_maxsh  = port_return(w_maxsh),  port_vol(w_maxsh)
        shrp_minvol = (ret_minvol - rf_rate) / vol_minvol if vol_minvol > 1e-9 else 0.0
        shrp_maxsh  = (ret_maxsh  - rf_rate) / vol_maxsh  if vol_maxsh  > 1e-9 else 0.0

        raw_w_cur = np.array([weights[a] for a in active_assets], dtype=float)
        w_cur     = raw_w_cur / raw_w_cur.sum()
        ret_cur, vol_cur = port_return(w_cur), port_vol(w_cur)
        shrp_cur  = (ret_cur - rf_rate) / vol_cur if vol_cur > 1e-9 else 0.0

        # Fronteira Eficiente
        target_rets = np.linspace(ret_minvol, mu_np.max() * 1.05, 120)
        frontier_vols, frontier_rets = [], []
        for tgt in target_rets:
            cons  = [eq_sum, {"type": "eq", "fun": lambda w, t=tgt: port_return(w) - t}]
            res_f = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 800})
            if res_f.success and res_f.fun < 2.0:
                frontier_vols.append(res_f.fun)
                frontier_rets.append(tgt)

        indiv_vols = [float(np.sqrt(Sigma_np[i, i])) for i in range(n_assets)]
        indiv_rets = [float(mu_np[i])                 for i in range(n_assets)]

        # Gráfico Fronteira Eficiente
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_rets, mode="lines", name="Fronteira Eficiente",
            line=dict(color=CORP["primary"], width=3),
            hovertemplate="<b>Fronteira</b><br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))
        cml_x = [0, vol_maxsh * 1.6]
        cml_y = [rf_rate, rf_rate + shrp_maxsh * vol_maxsh * 1.6]
        fig_ef.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode="lines", name="Capital Market Line",
            line=dict(color=CORP["warning"], width=1.8, dash="dash"),
            hoverinfo="skip",
        ))
        fig_ef.add_trace(go.Scatter(
            x=indiv_vols, y=indiv_rets, mode="markers+text",
            name="Ativos Individuais", text=active_assets, textposition="top center",
            textfont=dict(size=9, color=CORP["neutral"]),
            marker=dict(size=8, color="#CBD5E1", line=dict(color=CORP["neutral"], width=1)),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))
        for name_pt, ww, col_pt, sym, sz, text_pt in [
            (f"Atual  (Sharpe {shrp_cur:.2f})",    w_cur,    CORP["warning"], "star",    18, "Atual"),
            (f"Máx. Sharpe ({shrp_maxsh:.2f})",    w_maxsh,  CORP["success"], "star",    18, "Máx. Sharpe"),
            (f"Mín. Vol. (Sharpe {shrp_minvol:.2f})", w_minvol, CORP["purple"], "diamond", 16, "Mín. Vol."),
        ]:
            rv, vv = port_return(ww), port_vol(ww)
            fig_ef.add_trace(go.Scatter(
                x=[vv], y=[rv], mode="markers+text",
                name=name_pt, text=[text_pt], textposition="top right",
                marker=dict(size=sz, color=col_pt, symbol=sym, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>{text_pt}</b><br>Ret: %{{y:.2%}}<br>Vol: %{{x:.2%}}<extra></extra>",
            ))
        fig_ef.add_trace(go.Scatter(
            x=[0], y=[rf_rate], mode="markers+text", name=f"CDI ({rf_rate:.2%} a.a.)",
            text=["CDI"], textposition="bottom right",
            marker=dict(size=10, color=CORP["danger"], symbol="circle", line=dict(color="white", width=1)),
            hovertemplate=f"<b>CDI</b><br>Ret: {rf_rate:.2%}<br>Vol: 0%<extra></extra>",
        ))
        fig_ef.update_layout(**corp_layout(
            title=dict(text="Fronteira Eficiente de Markowitz — Universo de Ativos Ativos", font_size=14),
            xaxis=dict(title="Volatilidade Anualizada (%)", tickformat=".1%", rangemode="tozero", showgrid=False, linecolor=CORP["border"]),
            yaxis=dict(title="Retorno Esperado Anualizado (%)", tickformat=".1%", gridcolor="#F1F5F9", zeroline=False),
            legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", font_size=10),
            height=560, margin=dict(t=60, b=130),
        ))
        st.plotly_chart(fig_ef, use_container_width=True)

        # ── Comparativo de Alocação ───────────────────────────────────────────
        st.markdown("<div class='section-header'>Comparativo de Alocação</div>", unsafe_allow_html=True)

        df_alloc = pd.DataFrame({
            "⭐ Atual":        np.round(w_cur    * 100, 2),
            "🟢 Máx. Sharpe": np.round(w_maxsh  * 100, 2),
            "🟣 Mín. Vol.":   np.round(w_minvol * 100, 2),
        }, index=active_assets)
        df_alloc.index.name = "Ativo"

        df_metrics = pd.DataFrame({
            "Retorno Esperado (a.a.)": [f"{ret_cur:.2%}", f"{ret_maxsh:.2%}", f"{ret_minvol:.2%}"],
            "Volatilidade (a.a.)":     [f"{vol_cur:.2%}", f"{vol_maxsh:.2%}", f"{vol_minvol:.2%}"],
            "Índice de Sharpe":        [f"{shrp_cur:.2f}", f"{shrp_maxsh:.2f}", f"{shrp_minvol:.2f}"],
        }, index=["⭐ Atual", "🟢 Máx. Sharpe", "🟣 Mín. Vol."])

        col_ef1, col_ef2 = st.columns([3, 2])
        with col_ef1:
            st.markdown("**Alocação por Ativo (%)**")
            st.dataframe(
                df_alloc.style.format("{:.1f}%")
                    .background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100)
                    .highlight_null(color="white"),
                use_container_width=True, height=min(400, 50 + 35 * n_assets),
            )
        with col_ef2:
            st.markdown("**Métricas Resumidas**")
            st.dataframe(
                df_metrics.style
                    .set_properties(**{"text-align": "center"})
                    .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
                use_container_width=True,
            )

        delta_ret = ret_maxsh - ret_cur
        delta_vol = vol_maxsh - vol_cur
        st.info(
            f"**Potencial de Melhoria → Máx. Sharpe** | "
            f"Retorno: {'▲' if delta_ret >= 0 else '▼'} {abs(delta_ret):.2%} a.a.  |  "
            f"Volatilidade: {'▲' if delta_vol >= 0 else '▼'} {abs(delta_vol):.2%} a.a."
        )

        # ── Walk-Forward Optimization ─────────────────────────────────────────
        st.markdown("""
        <hr style='border:none;border-top:1px solid #E2E8F0;margin:20px 0'>
        <div class='section-header'>🔄 Otimização Walk-Forward — Carteira Cenários</div>
        """, unsafe_allow_html=True)
        st.caption(
            "Reotimiza a carteira a cada **6 meses** usando somente dados históricos disponíveis "
            "naquele momento. A **Carteira Cenários** é a mediana dos pesos ótimos de cada semestre."
        )

        with st.spinner("⚙️ Executando Walk-Forward Optimization semestral…"):
            rf_monthly_avg = cdi_ret_series.mean()
            wf_weights_list, wf_window_info = run_walkforward_optimization(
                returns_ef, rf_monthly_avg, window_months=6,
            )

        if len(wf_weights_list) < 2:
            st.warning(
                f"⚠️ Dados insuficientes para Walk-Forward "
                f"({len(wf_weights_list)} janela(s) — mínimo: 2). "
                "Amplie o período no sidebar para pelo menos 12 meses."
            )
        else:
            w_cenarios, df_wf_windows = build_scenario_portfolio(wf_weights_list, active_assets, method="median")
            shrp_cen,   vol_cen,  ret_cen  = compute_scenario_metrics(returns_ef, w_cenarios, cdi_ret_series)
            shrp_cur_wf, vol_cur_wf, ret_cur_wf = compute_scenario_metrics(returns_ef, w_cur, cdi_ret_series)

            st.markdown("<div class='section-header'>Alocação Comparativa — 4 Estratégias</div>", unsafe_allow_html=True)

            df_comp = pd.DataFrame({
                "⭐ Atual":          np.round(w_cur      * 100, 1),
                "🟢 Máx. Sharpe":   np.round(w_maxsh    * 100, 1),
                "🟣 Mín. Vol.":     np.round(w_minvol   * 100, 1),
                "🔵 Cenários (WF)": np.round(w_cenarios * 100, 1),
            }, index=active_assets)
            df_comp.index.name = "Ativo"

            st.dataframe(
                df_comp.style.format("{:.1f}%")
                    .background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100)
                    .highlight_null(color="white"),
                use_container_width=True,
                height=min(500, 60 + 35 * n_assets),
            )

            # Gráfico barras agrupadas
            _bar_colors = {
                "⭐ Atual":          CORP["warning"],
                "🟢 Máx. Sharpe":   CORP["success"],
                "🟣 Mín. Vol.":     CORP["purple"],
                "🔵 Cenários (WF)": CORP["primary"],
            }
            fig_wf = go.Figure()
            for col_label, color in _bar_colors.items():
                fig_wf.add_trace(go.Bar(
                    name=col_label, x=active_assets, y=df_comp[col_label],
                    marker_color=color, opacity=0.85,
                    hovertemplate=f"<b>{col_label}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
                ))
            fig_wf.update_layout(**corp_layout(
                barmode="group",
                title=dict(text=f"Comparativo de Alocação — {len(wf_weights_list)} janelas semestrais", font_size=13),
                xaxis=dict(title="Ativo", tickangle=-30, showgrid=False, linecolor=CORP["border"]),
                yaxis=dict(title="Peso (%)", ticksuffix="%", gridcolor="#F1F5F9"),
                legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center", font_size=11),
                height=440, margin=dict(t=90, b=80),
            ))
            st.plotly_chart(fig_wf, use_container_width=True)

            # Métricas comparativas
            st.markdown("<div class='section-header'>Validação — Carteira Cenários vs. Atual</div>", unsafe_allow_html=True)
            delta_sh = shrp_cen - shrp_cur_wf
            delta_rt = ret_cen  - ret_cur_wf
            delta_vl = vol_cen  - vol_cur_wf

            df_valid = pd.DataFrame({
                "Retorno (a.a.)":      [f"{ret_cur_wf:.2%}", f"{ret_cen:.2%}"],
                "Volatilidade (a.a.)": [f"{vol_cur_wf:.2%}", f"{vol_cen:.2%}"],
                "Índice de Sharpe":    [f"{shrp_cur_wf:.2f}", f"{shrp_cen:.2f}"],
            }, index=["⭐ Carteira Atual", "🔵 Cenários (WF)"])

            col_v1, col_v2 = st.columns([5, 4])
            with col_v1:
                st.dataframe(
                    df_valid.style
                        .set_properties(**{"text-align": "center"})
                        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
                    use_container_width=True,
                )
            with col_v2:
                _ic = lambda v: "▲" if v >= 0 else "▼"
                st.info(
                    f"**Ganho Cenários vs. Atual**\n\n"
                    f"Sharpe:       {_ic(delta_sh)} {abs(delta_sh):.2f}\n\n"
                    f"Retorno:      {_ic(delta_rt)} {abs(delta_rt):.2%} a.a.\n\n"
                    f"Volatilidade: {_ic(delta_vl)} {abs(delta_vl):.2%} a.a."
                )

            with st.expander(f"🔍 Detalhe por semestre ({len(wf_weights_list)} janelas — método: mediana)"):
                if wf_window_info:
                    df_wf_disp = df_wf_windows.copy()
                    df_wf_disp.index = [
                        f"S{i+1}: {s.strftime('%b/%Y')} → {e.strftime('%b/%Y')}"
                        for i, (s, e) in enumerate(wf_window_info)
                    ]
                    df_wf_disp.index.name = "Semestre"
                    st.dataframe(
                        (df_wf_disp * 100).style
                            .format("{:.1f}%")
                            .background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100),
                        use_container_width=True,
                    )
                st.caption("💡 A **Carteira Cenários** é a mediana coluna-a-coluna dos pesos acima, normalizada para 100%.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border:none;border-top:1px solid #E2E8F0;margin:32px 0 12px'>
<div style='text-align:center;color:#94A3B8;font-size:11px'>
  Asset Allocator Pro · Dados: yFinance, BCB (SGS 4391), brfunds/CVM ·
  Resultados históricos não garantem retornos futuros.
</div>
""", unsafe_allow_html=True)
