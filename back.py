import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime

# ==========================================
# 0. CONFIGURAÇÃO DA PÁGINA E CONSTANTES
# ==========================================
st.set_page_config(
    page_title="Asset Allocator Pro - Style Mais Retorno",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÃO MAIS RETORNO ---
# Substituído BRAPI_TOKEN por MAISRETORNO_TOKEN
MAISRETORNO_TOKEN = "SEU_TOKEN_AQUI" 

# Mapeamento Nome no Dashboard -> CNPJ
FUND_CNPJS = {
    'Tarpon GT': '22.232.927/0001-90',
    'Absolute Pace': '32.073.525/0001-43',
    'SPX Patriot': '15.334.585/0001-53',
    'Sparta Infra': '30.877.528/0001-04'
}

# CSS Customizado
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
# 1. DADOS DOS FUNDOS (ALTERADO PARA MAIS RETORNO)
# ==========================================

@st.cache_data(ttl=3600)
def get_maisretorno_fund_data(cnpjs_dict, token):
    """
    Busca dados de fundos na API Mais Retorno via CNPJ e calcula retorno mensal.
    """
    api_returns = pd.DataFrame()

    for name, cnpj_raw in cnpjs_dict.items():
        # Limpar CNPJ (apenas números)
        cnpj_clean = "".join(filter(str.isdigit, cnpj_raw))
        
        # Endpoint atualizado para Mais Retorno
        url = f"https://api.maisretorno.com/v1/fundos/performance/{cnpj_clean}"
        params = {
            'token': token, # Caso sua chave seja via parâmetro
            'periodo': 'max' # Busca o máximo de histórico disponível
        }
        
        try:
            # Headers comuns para APIs profissionais
            headers = {'Accept': 'application/json'}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # A estrutura da Mais Retorno costuma vir em 'performance' ou 'data'
                history = data.get('performance', []) or data.get('data', [])
                
                if history:
                    df_temp = pd.DataFrame(history)
                    
                    # Mapeamento de colunas comum: 'd' para data e 'v' para valor da cota/performance
                    if 'd' in df_temp.columns and 'v' in df_temp.columns:
                        df_temp['date'] = pd.to_datetime(df_temp['d'])
                        df_temp.set_index('date', inplace=True)
                        
                        # Resample para Mensal e cálculo de rentabilidade
                        # Se 'v' for cota, usamos pct_change. Se for performance acumulada, tratamos a variação.
                        monthly_data = df_temp['v'].resample('ME').last()
                        monthly_ret = monthly_data.pct_change()
                        monthly_ret.name = name
                        
                        if api_returns.empty:
                            api_returns = monthly_ret.to_frame()
                        else:
                            api_returns = api_returns.join(monthly_ret, how='outer')
            else:
                st.warning(f"Erro API Mais Retorno para {name}: {response.status_code}")
                
        except Exception as e:
            st.error(f"Falha ao buscar {name} na Mais Retorno: {e}")
            continue
            
    return api_returns

def get_combined_funds_data():
    # --- 1. Dados Hardcoded (Seu histórico manual permanece o mesmo) ---
    # (Omitido aqui por brevidade, mas deve conter seus dicionários tarpon_returns, absolute_returns, etc.)
    tarpon_returns = { '2018-01': 0.0518, '2025-12': 0.0112 } # Exemplo
    absolute_returns = { '2018-12': 0.0262 } # Exemplo
    spx_returns = { '2018-01': 0.0203 } # Exemplo
    sparta_returns = { '2018-06': 0.0075 } # Exemplo

    df_manual = pd.DataFrame({
        'Tarpon GT': pd.Series(tarpon_returns),
        'Absolute Pace': pd.Series(absolute_returns),
        'SPX Patriot': pd.Series(spx_returns),
        'Sparta Infra': pd.Series(sparta_returns)
    })
    df_manual.index = pd.to_datetime(df_manual.index).to_period('M').to_timestamp('M')

    # --- 2. Dados da API (Mais Retorno) ---
    df_api = get_maisretorno_fund_data(FUND_CNPJS, MAISRETORNO_TOKEN)
    
    if not df_api.empty:
        df_api.index = df_api.index.to_period('M').to_timestamp('M')
        # Combina manual com API, priorizando API em caso de overlap
        df_final = df_api.combine_first(df_manual)
    else:
        df_final = df_manual

    return df_final.sort_index()

# ==========================================
# 2. FUNÇÕES DE DADOS (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    
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

        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
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
        st.error(f"Erro no download: {e}")
        return pd.DataFrame()

@st.cache_data
def get_benchmark_data(start_date, end_date):
    """Baixa o Ibovespa (^BVSP) como benchmark de mercado."""
    try:
        # Ibovespa
        ibov = yf.download("BOVA11", start=start_date, end=end_date, progress=False)['Adj Close']
        ibov = ibov.resample('ME').last().pct_change()
        if isinstance(ibov, pd.DataFrame):
             ibov = ibov.iloc[:, 0]
        ibov.name = "Ibovespa"
        return ibov
    except:
        return pd.Series()

# ==========================================
# 3. LÓGICA DE CÁLCULO
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    
    if not available_assets: return None, None, None

    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() 
    
    # 1. Performance Pura (Cota Base 100) - Sem aportes externos
    portfolio_pure_idx = [100.0]
    monthly_returns = []
    
    # 2. Performance com Aportes (Patrimônio)
    portfolio_wealth = [initial_cap]
    
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        r_t = asset_returns_np[i]
        
        # Retorno do mês (weighted average)
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        
        # Atualiza Cota (Base 100)
        new_idx = portfolio_pure_idx[-1] * (1 + port_ret)
        portfolio_pure_idx.append(new_idx)
        
        # Atualiza Patrimônio (Com aporte)
        new_wealth = (portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution
        portfolio_wealth.append(new_wealth)
        
        # Drift dos pesos
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        
        # Rebalanceamento
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
    """Cria tabela estilo Mais Retorno (Ano x Mês)."""
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'] = df_ret.index.year
    df_ret['Mes'] = df_ret.index.month
    
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    
    # Adicionar acumulado do ano
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    
    # Mapa de meses numérico para nome curto
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    pivot.rename(columns=month_map, inplace=True)
    
    return pivot

# ==========================================
# 4. INTERFACE
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    min_date = datetime(2012, 1, 1)
    max_date = datetime.today()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", datetime(2018, 1, 1), min_value=min_date, max_value=max_date)
    end_date = col_d2.date_input("Fim", max_date, min_value=min_date, max_value=max_date)
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate_annual = st.number_input("Taxa CDI/Livre de Risco (% a.a.)", value=10.0, step=0.5)
    rf_rate_monthly = (1 + rf_rate_annual/100)**(1/12) - 1

    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000.0, step=100.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=50000.0, step=1000.0)

    st.markdown("---")
    st.subheader("📦 Composição da Carteira")
    
    # Inputs com valores padrão
    default_stocks = "AGRO3, B3SA3, BBAS3, BBSE3, BPAC11, CMIG3, EGIE3, ITUB3, PRIO3, PSSA3, SAPR4, SBSP3, TAEE3, TOTS3, VIVT3, WEGE3"
    default_fiis = "ALZR11, BRCO11, BTLG11, HGLG11, HGRE11, HGRU11, KNCR11, KNRI11, LVBI11, MXRF11, PMLL11, TRXF11, VILG11, VISC11, XPLG11, XPML11"
    default_etfs = "IVVB11"
    
    with st.expander("Selecionar Ativos", expanded=False):
        stocks_input = st.text_area("Ações BR", default_stocks)
        fiis_input = st.text_area("FIIs", default_fiis)
        etfs_input = st.text_area("ETFs", default_etfs)
    
    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações", 0, 100, 15)
    w_fiis = st.slider("FIIs", 0, 100, 5)
    w_etfs = st.slider("ETFs", 0, 100, 30)
    
    st.markdown("**Fundos Ativos**")
    w_tarpon = st.number_input("Fundo Tarpon", 0, 100, 20)
    w_absolute = st.number_input("Fundo Absolute", 0, 100, 10)
    w_sparta = st.number_input("Fundo Sparta", 0, 100, 10)
    w_spx = st.number_input("Fundo SPX Patriot", 0, 100, 10)
    
    total_w = w_stocks + w_fiis + w_etfs + w_tarpon + w_absolute + w_sparta + w_spx
    if total_w != 100:
        st.warning(f"Total: {total_w}%. Será normalizado.")

# --- DADOS ---
stock_list = [x.strip() for x in stocks_input.split(',') if x.strip()]
fii_list = [x.strip() for x in fiis_input.split(',') if x.strip()]
etf_list = [x.strip() for x in etfs_input.split(',') if x.strip()]

# Chamada da nova função de Fundos (Manual + API)
df_funds = get_combined_funds_data()

with st.spinner('Consolidando dados de mercado...'):
    df_stocks = get_market_data(stock_list, start_date, end_date)
    df_fiis = get_market_data(fii_list, start_date, end_date)
    df_etfs = get_market_data(etf_list, start_date, end_date)
    ibov_ret = get_benchmark_data(start_date, end_date)

# Consolidar Master DF
all_dates = df_funds.index.union(df_stocks.index).union(df_fiis.index).union(df_etfs.index)
if not ibov_ret.empty:
    all_dates = all_dates.union(ibov_ret.index)
all_dates = all_dates.sort_values()

master_df = pd.DataFrame(index=all_dates)

# Preencher classes de ativos
if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)

# Preencher Fundos (Reindexando para garantir alinhamento de datas)
master_df['Tarpon GT'] = df_funds['Tarpon GT'].reindex(master_df.index)
master_df['Absolute Pace'] = df_funds['Absolute Pace'].reindex(master_df.index)
master_df['Sparta Infra'] = df_funds['Sparta Infra'].reindex(master_df.index)
master_df['SPX Patriot'] = df_funds['SPX Patriot'].reindex(master_df.index)

# Filtrar datas
mask = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
master_df = master_df.loc[mask].dropna(how='all').fillna(0)
ibov_ret = ibov_ret.reindex(master_df.index).fillna(0)

weights = {
    'Ações Consolidadas': w_stocks,
    'FIIs Consolidados': w_fiis,
    'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon,
    'Absolute Pace': w_absolute,
    'Sparta Infra': w_sparta,
    'SPX Patriot': w_spx
}

# CALCULAR
port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is not None:
    # Gerar Benchmarks Acumulados
    cdi_ret_series = pd.Series(rf_rate_monthly, index=port_ret.index)
    cdi_accum = (1 + cdi_ret_series).cumprod() * 100
    ibov_accum = (1 + ibov_ret).cumprod() * 100
    
    # Métricas Gerais
    total_ret = (port_pure.iloc[-1] / 100) - 1
    years = len(port_ret) / 12
    # Proteção contra divisão por zero se years < 1
    if years > 0:
        cagr = (1 + total_ret) ** (1/years) - 1
    else:
        cagr = 0
        
    vol = port_ret.std() * np.sqrt(12)
    sharpe = (port_ret.mean() - rf_rate_monthly) / port_ret.std() * np.sqrt(12)
    
    # Drawdown
    cum_ret = (1 + port_ret).cumprod()
    peak = cum_ret.cummax()
    dd_series = (cum_ret - peak) / peak
    max_dd = dd_series.min()

    # Layout Principal
    st.title("📊 Relatório de Performance")
    
    # --- HEADER KPI (Estilo Cards) ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='metric-card'><div class='metric-value'>{total_ret:.1%}</div><div class='metric-label'>Retorno Total</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-value'>{cagr:.1%}</div><div class='metric-label'>CAGR (a.a.)</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-value'>{vol:.1%}</div><div class='metric-label'>Volatilidade</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-value'>{sharpe:.2f}</div><div class='metric-label'>Sharpe</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-card'><div class='metric-value' style='color:red'>{max_dd:.1%}</div><div class='metric-label'>Max Drawdown</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS DE ANÁLISE ---
    tab_perf, tab_risk, tab_month, tab_patr = st.tabs([
        "📈 Rentabilidade Comparativa", 
        "🛡️ Análise de Risco", 
        "📅 Retornos Mensais",
        "💰 Evolução Patrimonial"
    ])
    
    with tab_perf:
        st.subheader("Evolução (Base 100)")
        df_chart = pd.DataFrame({
            'Seu Portfólio': port_pure,
            'Ibovespa': ibov_accum,
            'CDI (Teórico)': cdi_accum
        })
        
        fig = px.line(df_chart, title="Comparativo de Rentabilidade Acumulada")
        fig.update_layout(
            template="plotly_white", 
            xaxis_title="", 
            yaxis_title="Índice (Base 100)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Nota: O gráfico acima mostra a valorização pura das cotas (iniciando em 100), ignorando aportes, para permitir comparação justa com índices.")

    with tab_risk:
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("**Drawdown Submarino**")
            fig_dd = px.area(dd_series, title="")
            fig_dd.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
            fig_dd.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_dd, use_container_width=True)
            
        with col_r2:
            st.markdown("**Volatilidade Móvel (12 Meses)**")
            rolling_vol = port_ret.rolling(12).std() * np.sqrt(12)
            fig_vol = px.line(rolling_vol, title="")
            fig_vol.update_traces(line_color='#FF9800')
            fig_vol.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        # Estatísticas Adicionais
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
        
        # Colorir dataframe (Estilo Excel)
        st.dataframe(
            heatmap_data.style.format("{:.2%}")
            .background_gradient(cmap='RdYlGn', vmin=-0.05, vmax=0.05, axis=None)
            .highlight_null(color='white'),
            use_container_width=True,
            height=400
        )
        
        st.caption("YTD: Rentabilidade acumulada no ano corrente.")

    with tab_patr:
        st.subheader("Evolução do Saldo em Conta")
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            fig_wealth = px.area(port_wealth, title="Crescimento Patrimonial (Cotas + Aportes)")
            fig_wealth.update_traces(fillcolor='rgba(76, 175, 80, 0.3)', line_color='#4CAF50')
            fig_wealth.update_layout(template="plotly_white", yaxis_title="Saldo (R$)")
            st.plotly_chart(fig_wealth, use_container_width=True)
        
        with col_p2:
            final_val = port_wealth.iloc[-1]
            total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
            profit_loss = final_val - total_invested
            
            st.metric("Saldo Final", f"R$ {final_val:,.2f}")
            st.metric("Total Investido", f"R$ {total_invested:,.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_loss:,.2f}", 
                      delta=f"{(final_val/total_invested - 1):.1%}")

else:
    st.info("👈 Configure os parâmetros na barra lateral e aguarde o processamento.")
