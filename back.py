import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import requests
import io

# ==========================================
# 0. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Asset Allocator Pro - CVM Automático",
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. FUNÇÕES DE DADOS CVM (AUTOMÁTICO)
# ==========================================
@st.cache_data(ttl=86400) # Cache de 24 horas
def get_cvm_funds_data(cnpj_dict, start_date, end_date):
    """
    Busca dados de cotas no repositório de dados abertos da CVM.
    cnpj_dict: { 'Nome amigável': 'CNPJ_SÓ_NÚMEROS' }
    """
    all_returns = pd.DataFrame()
    
    # Gerar lista de meses/anos para baixar
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    progress_text = st.empty()
    
    for label, cnpj in cnpj_dict.items():
        cnpj_clean = cnpj.replace('.', '').replace('/', '').replace('-', '')
        fund_series = []
        
        for dt in date_range:
            year = dt.year
            month = dt.month
            url = f"https://dados.cvm.gov.br/dados/FIE/MED/INFORME_DIARIO/DADOS/inf_diario_fie_{year}{month:02d}.zip"
            
            try:
                # O CVM disponibiliza em ZIP. O pandas lê direto se for CSV, 
                # mas aqui precisamos tratar o download do ZIP ou CSV direto dependendo do ano
                # Historicamente a CVM muda o padrão. Usaremos o CSV direto para anos recentes:
                url_csv = f"https://dados.cvm.gov.br/dados/FIE/MED/INFORME_DIARIO/DADOS/inf_diario_fie_{year}{month:02d}.csv"
                df_month = pd.read_csv(url_csv, sep=';', encoding='ISO-8859-1', usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'])
                
                # Filtrar pelo CNPJ
                df_fund = df_month[df_month['CNPJ_FUNDO'] == f"{cnpj_clean[:2]}.{cnpj_clean[2:5]}.{cnpj_clean[5:8]}/{cnpj_clean[8:12]}-{cnpj_clean[12:]}"]
                if df_fund.empty:
                    # Tentar sem formatação caso o CSV venha limpo
                    df_fund = df_month[df_month['CNPJ_FUNDO'] == cnpj_clean]
                
                if not df_fund.empty:
                    df_fund['DT_COMPTC'] = pd.to_datetime(df_fund['DT_COMPTC'])
                    df_fund = df_fund.set_index('DT_COMPTC').sort_index()
                    fund_series.append(df_fund['VL_QUOTA'])
            except:
                continue
        
        if fund_series:
            full_series = pd.concat(fund_series)
            # Resample para mensal (última cota do mês) e calcula retorno
            monthly_ret = full_series.resample('ME').last().pct_change()
            all_returns[label] = monthly_ret

    return all_returns

# ==========================================
# 2. FUNÇÕES DE MERCADO (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    processed_tickers = [f"{t.strip().upper()}.SA" if "." not in t and any(c.isdigit() for c in t) else t.strip().upper() for t in tickers]
    try:
        data = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        if isinstance(prices, pd.Series): prices = prices.to_frame(name=processed_tickers[0])
        if isinstance(prices.columns, pd.MultiIndex): prices.columns = prices.columns.get_level_values(-1)
        returns = prices.resample('ME').last().pct_change()
        returns.columns = [str(c).replace('.SA', '') for c in returns.columns]
        return returns
    except: return pd.DataFrame()

@st.cache_data
def get_benchmark_data(start_date, end_date):
    try:
        ibov = yf.download("BOVA11", start=start_date, end=end_date, progress=False)['Adj Close']
        return ibov.resample('ME').last().pct_change()
    except: return pd.Series()

# ==========================================
# 3. LÓGICA DE CÁLCULO
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    if not available_assets: return None, None, None

    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() 
    
    portfolio_pure_idx = [100.0]
    portfolio_wealth = [initial_cap]
    monthly_returns = []
    
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        r_t = asset_returns_np[i]
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        
        portfolio_pure_idx.append(portfolio_pure_idx[-1] * (1 + port_ret))
        portfolio_wealth.append((portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution)
        
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        if (rebalance_freq == 'Mensal') or (rebalance_freq == 'Anual' and dates[i].month == 12):
            current_weights = active_weights.copy()
            
    return pd.Series(portfolio_pure_idx[1:], index=dates), \
           pd.Series(portfolio_wealth[1:], index=dates), \
           pd.Series(monthly_returns, index=dates, name="Portfólio")

def create_monthly_heatmap(returns_series):
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'], df_ret['Mes'] = df_ret.index.year, df_ret.index.month
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    return pivot.rename(columns=month_map)

# ==========================================
# 4. INTERFACE SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    col_d1, col_d2 = st.columns(2)
    # CVM data costuma ter delay de alguns meses para dados abertos completos, 
    # ajustado para 2020 para ser mais rápido no exemplo
    start_date = col_d1.date_input("Início", date(2020, 1, 1))
    end_date = col_d2.date_input("Fim", datetime.today())
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate_annual = st.number_input("CDI Esperado (% a.a.)", value=10.5)
    rf_rate_monthly = (1 + rf_rate_annual/100)**(1/12) - 1

    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=50000.0)

    st.markdown("---")
    st.subheader("🏦 Configuração dos Fundos (CVM)")
    cnpj_tarpon = st.text_input("CNPJ Tarpon GT", "08.941.130/0001-41")
    cnpj_absolute = st.text_input("CNPJ Absolute Pace", "22.016.963/0001-90")
    cnpj_sparta = st.text_input("CNPJ Sparta Infra", "32.548.784/0001-09")

    with st.expander("Ativos de Mercado (Tickers)", expanded=False):
        stocks_input = st.text_area("Ações BR", "AGRO3, B3SA3, BBAS3, ITUB3, WEGE3")
        fiis_input = st.text_area("FIIs", "HGLG11, KNRI11, MXRF11")
        etfs_input = st.text_area("ETFs", "IVVB11")
    
    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações", 0, 100, 20)
    w_fiis = st.slider("FIIs", 0, 100, 10)
    w_etfs = st.slider("ETFs", 0, 100, 20)
    w_tarpon = st.number_input("Peso Tarpon", 0, 100, 20)
    w_absolute = st.number_input("Peso Absolute", 0, 100, 20)
    w_sparta = st.number_input("Peso Sparta", 0, 100, 10)

# --- PROCESSAMENTO ---
with st.spinner('Buscando dados na CVM e Yahoo Finance...'):
    # 1. Dados CVM
    cnpjs = {"Tarpon GT": cnpj_tarpon, "Absolute Pace": cnpj_absolute, "Sparta Infra": cnpj_sparta}
    df_cvm = get_cvm_funds_data(cnpjs, start_date, end_date)
    
    # 2. Dados Mercado
    df_stocks = get_market_data(stocks_input.split(','), start_date, end_date)
    df_fiis = get_market_data(fiis_input.split(','), start_date, end_date)
    df_etfs = get_market_data(etfs_input.split(','), start_date, end_date)
    ibov_ret = get_benchmark_data(start_date, end_date)

    # Consolidação
    master_df = pd.DataFrame(index=df_cvm.index)
    if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
    if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
    if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)
    
    for col in df_cvm.columns:
        master_df[col] = df_cvm[col]

    master_df = master_df.fillna(0)
    weights = {
        'Ações Consolidadas': w_stocks, 'FIIs Consolidados': w_fiis, 'ETFs Consolidados': w_etfs,
        'Tarpon GT': w_tarpon, 'Absolute Pace': w_absolute, 'Sparta Infra': w_sparta
    }

    port_pure, port_wealth, port_ret = calculate_portfolio_performance(
        master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
    )

# --- DASHBOARD ---
if port_ret is not None:
    # Métricas
    total_ret = (port_pure.iloc[-1] / 100) - 1
    vol = port_ret.std() * np.sqrt(12)
    sharpe = (port_ret.mean() - rf_rate_monthly) / port_ret.std() * np.sqrt(12) if port_ret.std() != 0 else 0
    
    st.title("📊 Asset Allocator Pro (CVM Real-Time)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{total_ret:.1%}</div><div class='metric-label'>Retorno Total</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{vol:.1%}</div><div class='metric-label'>Volatilidade (a.a.)</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{sharpe:.2f}</div><div class='metric-label'>Sharpe</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-value'>R$ {port_wealth.iloc[-1]:,.0f}</div><div class='metric-label'>Patrimônio Final</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Rentabilidade", "📅 Mensal", "💰 Patrimônio"])
    
    with tab1:
        ibov_acc = (1 + ibov_ret.reindex(port_pure.index).fillna(0)).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_pure.index, y=port_pure, name="Sua Carteira", line=dict(color='#2ecc71', width=3)))
        fig.add_trace(go.Scatter(x=ibov_acc.index, y=ibov_acc, name="Ibovespa", line=dict(color='#95a5a6', dash='dot')))
        fig.update_layout(template="plotly_white", title="Evolução de R$ 100", xaxis_title="Data")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        heatmap = create_monthly_heatmap(port_ret)
        st.dataframe(heatmap.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

    with tab3:
        fig_p = px.area(port_wealth, title="Evolução Patrimonial (Aportes + Rentabilidade)")
        fig_p.update_layout(yaxis_title="Reais (R$)", template="plotly_white")
        st.plotly_chart(fig_p, use_container_width=True)
else:
    st.error("Dados insuficientes para gerar o relatório. Verifique os CNPJs e o intervalo de datas.")
