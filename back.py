import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import warnings
from brfunds import getFundsEarnings

# ==========================================
# 0. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Asset Allocator Pro - Style Mais Retorno",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# CSS Customizado estilo "Financial Dashboard"
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. FUNÇÕES DE DADOS (CVM / YFINANCE / BCB)
# ==========================================

@st.cache_data(show_spinner=False, ttl=86400)
def get_fundos_cvm(cnpj_dict, start_date, end_date):
    """
    Versão Sênior: Resolve problemas de colunas duplicadas e índice de tempo.
    """
    try:
        start_str = start_date.strftime('%d/%m/%y')
        end_str = end_date.strftime('%d/%m/%y')
        cnpjs = list(cnpj_dict.values())
        
        # Coleta os dados
        df_cvm = getFundsEarnings(*cnpjs, start=start_str, end=end_str)
        
        if df_cvm is None or df_cvm.empty:
            return pd.DataFrame()

        # --- CORREÇÃO 1: GARANTIR DATETIMEINDEX ---
        if 'Date' in df_cvm.columns:
            df_cvm['Date'] = pd.to_datetime(df_cvm['Date'])
            df_cvm.set_index('Date', inplace=True)
        df_cvm.index = pd.to_datetime(df_cvm.index)
        df_cvm = df_cvm.sort_index()

        # --- CORREÇÃO 2: RENOMEAÇÃO E DESDUPLICAÇÃO ---
        new_cols = []
        for col in df_cvm.columns:
            matched_name = "DROP"
            for short_name, cnpj in cnpj_dict.items():
                # Compara CNPJ limpo ou nome curto no retorno da API
                if cnpj.replace('.','').replace('/','') in str(col).replace('.','').replace('/','') or \
                   short_name.upper() in str(col).upper():
                    matched_name = short_name
                    break
            new_cols.append(matched_name)
        
        df_cvm.columns = new_cols
        if "DROP" in df_cvm.columns:
            df_cvm = df_cvm.drop(columns=["DROP"])

        # Se a API retornou 2 colunas para o mesmo fundo, pegamos apenas a primeira
        # Isso evita o erro "Cannot set a DataFrame to single column"
        df_cvm = df_cvm.groupby(level=0, axis=1).first()

        # Conversão para retorno mensal
        df_retornos = (df_cvm + 1.0).resample('ME').last().pct_change().dropna(how='all')

        # --- LÓGICA HÍBRIDA (REAL INVESTOR LEGACY) ---
        legacy_ri = {
            '2012-06': 0.0035, '2012-07': 0.0483, '2012-08': 0.0247, '2012-09': 0.0385, '2012-10': 0.0401, '2012-11': 0.0210, '2012-12': 0.0463,
            '2013-01': 0.0270, '2013-02': -0.0150, '2013-03': -0.0190, '2013-04': 0.0194, '2013-05': 0.0232, '2013-06': -0.0898, '2013-07': 0.0076, '2013-08': 0.0116, '2013-09': 0.0426, '2013-10': 0.0346, '2013-11': -0.0135, '2013-12': -0.0125,
            '2014-01': -0.0384, '2014-02': 0.0122, '2014-03': 0.0610, '2014-04': 0.0315, '2014-05': 0.0132, '2014-06': 0.0378, '2014-07': 0.0203, '2014-08': 0.0760, '2014-09': -0.0543, '2014-10': 0.0306, '2014-11': 0.0253, '2014-12': -0.0354,
            '2015-01': -0.0575, '2015-02': 0.0631, '2015-03': -0.0163, '2015-04': 0.0768, '2015-05': -0.0441, '2015-06': 0.0044, '2015-07': -0.0243, '2015-08': -0.0531, '2015-09': -0.0185, '2015-10': 0.0366, '2015-11': -0.0224, '2015-12': -0.0128,
            '2016-01': -0.0427, '2016-02': 0.0573, '2016-03': 0.1190, '2016-04': 0.0747, '2016-05': -0.0118, '2016-06': 0.0541, '2016-07': 0.0863, '2016-08': 0.0205, '2016-09': 0.0076, '2016-10': 0.0754, '2016-11': -0.0454, '2016-12': 0.0152,
            '2017-01': 0.0607, '2017-02': 0.0487, '2017-03': 0.0016, '2017-04': 0.0154, '2017-05': -0.0229, '2017-06': 0.0118, '2017-07': 0.0558, '2017-08': 0.0620, '2017-09': 0.0519, '2017-10': 0.0119, '2017-11': -0.0250, '2017-12': 0.0494
        }
        s_legacy = pd.Series(legacy_ri, name='Real Investor')
        s_legacy.index = pd.to_datetime(s_legacy.index).to_period('M').to_timestamp('M')
        
        if 'Real Investor' in df_retornos.columns:
            df_retornos['Real Investor'] = df_retornos['Real Investor'].combine_first(s_legacy)
        else:
            df_retornos['Real Investor'] = s_legacy
            
        return df_retornos

    except Exception as e:
        st.warning(f"⚠️ Erro na CVM: {e}")
        return pd.DataFrame()

@st.cache_data
def get_cdi_data(start_date, end_date):
    """Busca o CDI mensal (Série 4391) na API do BCB."""
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df.set_index('data', inplace=True)
        df['valor'] = df['valor'].astype(float) / 100.0
        
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        cdi_series = df.loc[mask, 'valor']
        cdi_series = cdi_series.resample('ME').last()
        cdi_series.name = 'CDI'
        
        return cdi_series
    except Exception as e:
        st.error(f"Erro ao baixar dados do CDI (BCB): {e}")
        return pd.Series(dtype='float64')

@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    
    processed_tickers = []
    for t in tickers:
        t = t.strip().upper()
        if "." not in t and any(char.isdigit() for char in t): 
            processed_tickers.append(f"{t}.SA")
        else:
            processed_tickers.append(t)
            
    try:
        data = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()

        if 'Adj Close' in data.columns: prices = data['Adj Close']
        elif 'Close' in data.columns: prices = data['Close']
        else:
            try: prices = data.xs('Adj Close', level=0, axis=1)
            except KeyError: prices = data.xs('Close', level=0, axis=1)
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=processed_tickers[0])
            
        if isinstance(prices.columns, pd.MultiIndex):
             prices.columns = prices.columns.get_level_values(-1)

        monthly_prices = prices.resample('ME').last() 
        returns = monthly_prices.pct_change()
        returns.columns = [str(c).replace('.SA', '') for c in returns.columns]
        
        return returns
    except Exception as e:
        st.error(f"Erro no download (YFinance): {e}")
        return pd.DataFrame()

@st.cache_data
def get_benchmark_data(start_date, end_date):
    try:
        ibov = yf.download("BOVA11.SA", start=start_date, end=end_date, progress=False)
        if ibov.empty: return pd.Series(dtype='float64')

        if 'Adj Close' in ibov.columns: prices = ibov['Adj Close']
        else: prices = ibov.iloc[:, 0]

        returns = prices.resample('ME').last().pct_change().dropna()
        returns.name = "Ibovespa"
        return returns
    except Exception as e:
        st.error(f"Erro no benchmark: {e}")
        return pd.Series(dtype='float64')

# ==========================================
# 2. LÓGICA DE CÁLCULO
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    
    if not available_assets: return None, None, None

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
        
        is_rebalance_time = (rebalance_freq == 'Mensal') or \
                            (rebalance_freq == 'Anual' and dates[i].month == 12)
        if is_rebalance_time:
            current_weights = active_weights.copy()
            
    portfolio_pure_series = pd.Series(portfolio_pure_idx[1:], index=dates)
    portfolio_wealth_series = pd.Series(portfolio_wealth[1:], index=dates)
    monthly_returns_series = pd.Series(monthly_returns, index=dates)
    monthly_returns_series.name = "Portfólio"
    
    return portfolio_pure_series, portfolio_wealth_series, monthly_returns_series

def create_monthly_heatmap(returns_series):
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'] = df_ret.index.year
    df_ret['Mes'] = df_ret.index.month
    
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    pivot.rename(columns=month_map, inplace=True)
    return pivot

# ==========================================
# 3. INTERFACE E EXECUÇÃO
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    min_date = datetime(2012, 1, 1)
    max_date = datetime.today()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", datetime(2018, 1, 1), min_value=min_date, max_value=max_date)
    end_date = col_d2.date_input("Fim", max_date, min_value=min_date, max_value=max_date)
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])

    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=1000.0, step=100.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=100000.0, step=1000.0)

    st.markdown("---")
    st.subheader("📦 Composição da Carteira")
    
    with st.expander("Selecionar Ativos", expanded=False):
        default_stocks = "EGIE3, ITUB3, PSSA3, WEGE3, CXSE3, SBSP3, TAEE3, VIVT3, CPFE3, SAPR3, BBAS3, PRIO3, TOTS3, BPAC3, ALUP3, BMOB3"
        default_fiis = "ALZR11, BRCO11, BTLG11, HGLG11, HGRE11, HGRU11, KNCR11, KNRI11, LVBI11, MXRF11, PMAL11, XPLG11, XPML11"
        default_etfs = "IVVB11"
        stocks_input = st.text_area("Ações BR", default_stocks)
        fiis_input = st.text_area("FIIs", default_fiis)
        etfs_input = st.text_area("ETFs", default_etfs)
    
    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações", 0, 100, 35)
    w_fiis = st.slider("FIIs", 0, 100, 0)
    w_etfs = st.slider("ETFs", 0, 100, 20)
    
    st.markdown("**Fundos Ativos & Caixa**")
    col_f1, col_f2 = st.columns(2)
    w_tarpon = col_f1.number_input("Tarpon GT", 0, 100, 10)
    w_absolute = col_f2.number_input("Absolute Pace", 0, 100, 25)
    w_cdi = col_f1.number_input("CDI", 0, 100, 0)
    w_spx = col_f2.number_input("SPX Patriot", 0, 100, 0)
    w_real = col_f1.number_input("Real Investor", 0, 100, 0)
    w_organon = col_f2.number_input("Organon FIC", 0, 100, 10)
    
    total_w = w_stocks + w_fiis + w_etfs + w_tarpon + w_absolute + w_cdi + w_spx + w_real + w_organon
    if total_w != 100:
        st.warning(f"Total: {total_w}%. Será normalizado.")

stock_list = [x.strip() for x in stocks_input.split(',') if x.strip()]
fii_list = [x.strip() for x in fiis_input.split(',') if x.strip()]
etf_list = [x.strip() for x in etfs_input.split(',') if x.strip()]

# Dicionário de CNPJs exatos dos fundos acompanhados
dict_fundos_cnpjs = {
    'Tarpon GT': '22.232.927/0001-90',

    'Absolute Pace': '32.073.525/0001-43',

    'SPX Patriot': '15.334.585/0001-53', # Favor validar este CNPJ

    'Real Investor': '10.500.884/0001-05',

    'Organon FIC FIA': '17.400.251/0001-66'
}

with st.spinner('Consolidando dados de mercado, taxas e fundos CVM...'):
    df_stocks = get_market_data(stock_list, start_date, end_date)
    df_fiis = get_market_data(fii_list, start_date, end_date)
    df_etfs = get_market_data(etf_list, start_date, end_date)
    ibov_ret = get_benchmark_data(start_date, end_date)
    cdi_ret = get_cdi_data(start_date, end_date)
    
    # Nova extração automatizada
    df_funds = get_fundos_cvm(dict_fundos_cnpjs, start_date, end_date)

# Agrupando todos os índices de data possíveis
all_dates = pd.DatetimeIndex([])
if not df_stocks.empty: all_dates = all_dates.union(df_stocks.index)
if not df_fiis.empty: all_dates = all_dates.union(df_fiis.index)
if not df_etfs.empty: all_dates = all_dates.union(df_etfs.index)
if not cdi_ret.empty: all_dates = all_dates.union(cdi_ret.index)
if not df_funds.empty: all_dates = all_dates.union(df_funds.index)
if not ibov_ret.empty: all_dates = all_dates.union(ibov_ret.index)

all_dates = all_dates.sort_values()
master_df = pd.DataFrame(index=all_dates)

if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)

master_df['CDI'] = cdi_ret.reindex(master_df.index)

# Integração segura dos fundos da CVM no Master Dataframe
for fundo in dict_fundos_cnpjs.keys():
    if not df_funds.empty and fundo in df_funds.columns:
        master_df[fundo] = df_funds[fundo].reindex(master_df.index)
    else:
        master_df[fundo] = 0.0  # Asserção de fallback caso a CVM não retorne

# Filtro final de datas
mask = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
master_df = master_df.loc[mask].dropna(how='all').fillna(0)
ibov_ret = ibov_ret.reindex(master_df.index).fillna(0)
cdi_ret = cdi_ret.reindex(master_df.index).fillna(0)

weights = {
    'Ações Consolidadas': w_stocks,
    'FIIs Consolidados': w_fiis,
    'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon,
    'Absolute Pace': w_absolute,
    'CDI': w_cdi,
    'SPX Patriot': w_spx,
    'Real Investor': w_real,
    'Organon FIC FIA': w_organon
}

port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is not None:
    cdi_ret_series = cdi_ret.reindex(port_ret.index).fillna(0)
    cdi_accum = (1 + cdi_ret_series).cumprod() * 100
    ibov_accum = (1 + ibov_ret).cumprod() * 100
    
    total_ret = (port_pure.iloc[-1] / 100) - 1
    years = len(port_ret) / 12
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol = port_ret.std() * np.sqrt(12)
    
    excess_returns = port_ret - cdi_ret_series
    sharpe = (excess_returns.mean() / port_ret.std()) * np.sqrt(12) if port_ret.std() > 0 else 0
    
    cum_ret = (1 + port_ret).cumprod()
    peak = cum_ret.cummax()
    dd_series = (cum_ret - peak) / peak
    max_dd = dd_series.min()

    st.title("📊 Relatório de Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='metric-card'><div class='metric-value'>{total_ret:.1%}</div><div class='metric-label'>Retorno Total</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-value'>{cagr:.1%}</div><div class='metric-label'>CAGR (a.a.)</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-value'>{vol:.1%}</div><div class='metric-label'>Volatilidade</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-label'>Sharpe (vs. CDI)</div><div class='metric-value'>{sharpe:.2f}</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-card'><div class='metric-value' style='color:red'>{max_dd:.1%}</div><div class='metric-label'>Max Drawdown</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    tab_perf, tab_risk, tab_month, tab_patr, tab_proj = st.tabs([
        "📈 Rentabilidade Comparativa",
        "🛡️ Análise de Risco",
        "📅 Retornos Mensais",
        "💰 Evolução Patrimonial",
        "🔮 Projeções (3 Anos)"
    ])
    
    with tab_perf:
        st.subheader("Evolução (Base 100)")
        df_chart = pd.DataFrame({
            'Seu Portfólio': port_pure,
            'Ibovespa': ibov_accum,
            'CDI Real (BCB)': cdi_accum
        })
        
        fig = px.line(df_chart, title="Comparativo de Rentabilidade Acumulada")
        fig.update_layout(
            template="plotly_white", 
            xaxis_title="", 
            yaxis_title="Índice (Base 100)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")

    with tab_risk:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Drawdown Submarino**")
            fig_dd = px.area(dd_series, title="")
            fig_dd.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
            fig_dd.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_dd, width="stretch")
            
        with col_r2:
            st.markdown("**Volatilidade Móvel (12 Meses)**")
            rolling_vol = port_ret.rolling(12).std() * np.sqrt(12)
            fig_vol = px.line(rolling_vol, title="")
            fig_vol.update_traces(line_color='#FF9800')
            fig_vol.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_vol, width="stretch")

        st.markdown("### Estatísticas Detalhadas")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        months_pos = (port_ret > 0).sum()
        months_neg = (port_ret < 0).sum()
        best_month = port_ret.max()
        worst_month = port_ret.min()
        
        stat_col1.metric("Meses Positivos", f"{months_pos} ({months_pos/len(port_ret):.0%})")
        stat_col2.metric("Meses Negativos", f"{months_neg} ({months_neg/len(port_ret):.0%})")
        stat_col3.metric("Melhor Mês", f"{best_month:.2%}")
        stat_col4.metric("Pior Mês", f"{worst_month:.2%}", delta_color="inverse")

    with tab_month:
        st.subheader("Tabela de Rentabilidade (Heatmap)")
        heatmap_data = create_monthly_heatmap(port_ret)
        
        st.dataframe(
            heatmap_data.style.format("{:.2%}")
            .background_gradient(cmap='RdYlGn', vmin=-0.05, vmax=0.05, axis=None)
            .highlight_null(color='white'),
            width="stretch",
            height=400
        )

        st.markdown("---")
        st.subheader("💎 Sharpe Ratio (Janelas Móveis)")
        
        sharpe_periods = {
            "12 Meses": 12, "24 Meses": 24, "48 Meses": 48, 
            "60 Meses": 60, "Desde o Início": len(port_ret)
        }
        
        sharpe_results = {}
        for label, months in sharpe_periods.items():
            if len(port_ret) >= months:
                subset_port = port_ret.tail(months)
                subset_cdi = cdi_ret_series.tail(months)
                vol_subset = subset_port.std()
                if vol_subset > 0:
                    excess_subset = subset_port - subset_cdi
                    sharpe_val = (excess_subset.mean() / vol_subset) * np.sqrt(12)
                    sharpe_results[label] = sharpe_val
                else:
                    sharpe_results[label] = 0.0
            else:
                sharpe_results[label] = None

        df_sharpe_table = pd.DataFrame([sharpe_results], index=["Índice de Sharpe"])
        
        st.dataframe(
            df_sharpe_table.style.format("{:.2f}", na_rep="-")
            .background_gradient(cmap='Blues', axis=1, vmin=0, vmax=2),
            width="stretch"
        )

    with tab_patr:
        st.subheader("Evolução do Saldo em Conta")
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            fig_wealth = px.area(port_wealth, title="Crescimento Patrimonial (Cotas + Aportes)")
            fig_wealth.update_traces(fillcolor='rgba(76, 175, 80, 0.3)', line_color='#4CAF50')
            fig_wealth.update_layout(template="plotly_white", yaxis_title="Saldo (R$)")
            st.plotly_chart(fig_wealth, width="stretch")
        
        with col_p2:
            final_val = port_wealth.iloc[-1]
            total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
            profit_loss = final_val - total_invested
            
            st.metric("Saldo Final", f"R$ {final_val:,.2f}")
            st.metric("Total Investido", f"R$ {total_invested:,.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_loss:,.2f}", 
                      delta=f"{(final_val/total_invested - 1):.1%}")

    with tab_proj:
        st.subheader("🔮 Projeção de Cenários — Próximos 36 Meses")

        mu    = port_ret.mean()
        sigma = port_ret.std()
        N_MONTHS  = 36
        N_SIM     = 20_000
        saldo_t0  = port_wealth.iloc[-1]
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
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.09)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="Intervalo P5–P95",
            hoverinfo="skip",
        ))

        fig_proj.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values, mode="lines", name="Histórico Real",
            line=dict(color="#2c3e50", width=3),
            hovertemplate="<b>Histórico</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates, y=p_otimista, mode="lines", name="Otimista (P95)",
            line=dict(color="#27ae60", width=2.5, dash="dash"),
            hovertemplate="<b>Otimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates, y=p_neutro, mode="lines", name="Neutro (P50)",
            line=dict(color="#2980b9", width=2.5, dash="dot"),
            hovertemplate="<b>Neutro</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates, y=p_pessimista, mode="lines", name="Pessimista (P5)",
            line=dict(color="#e74c3c", width=2.5, dash="dash"),
            hovertemplate="<b>Pessimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_vline(x=last_date, line_width=1.5, line_dash="dot", line_color="gray")
        
        fig_proj.update_layout(
            template="plotly_white",
            title=dict(
                text=f"Monte Carlo — {N_SIM:,} sim. | µ={mu:.2%}/mês | σ={sigma:.2%}/mês | Aporte R$ {aporte_mensal:,.0f}/mês",
                font_size=13,
            ),
            yaxis=dict(title="Saldo (R$)", tickformat=",.0f"),
            xaxis_title="",
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
            hovermode="x unified",
            margin=dict(t=80),
        )

        st.plotly_chart(fig_proj, width="stretch")

        st.markdown("### 📊 Saldo Final Projetado em 36 Meses")
        col_p1, col_p2, col_p3 = st.columns(3)

        col_p1.metric("🟢 Otimista (P95)", f"R$ {p_otimista[-1]:,.2f}", f"+R$ {p_otimista[-1] - saldo_t0:,.0f} vs. hoje")
        col_p2.metric("🔵 Neutro (P50)", f"R$ {p_neutro[-1]:,.2f}", f"+R$ {p_neutro[-1] - saldo_t0:,.0f} vs. hoje")
        col_p3.metric("🔴 Pessimista (P5)", f"R$ {p_pessimista[-1]:,.2f}", f"R$ {p_pessimista[-1] - saldo_t0:,.0f} vs. hoje", delta_color="inverse")

else:
    st.info("👈 Configure os parâmetros na barra lateral e aguarde o processamento.")
