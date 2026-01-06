import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. DADOS HARDCODED (FUNDOS ATIVOS)
# ==========================================
def get_hardcoded_funds():
    """
    Retorna um DataFrame com os retornos mensais dos fundos ativos específicos.
    Tratamento: Converte strings de data para datetime (fim do mês) e preenche lacunas.
    """
    tarpon_returns = {
        '2018-01': 0.0518, '2018-02': 0.0018, '2018-03': 0.0337, '2018-04': -0.0224, '2018-05': -0.1091,
        '2018-06': -0.0884, '2018-07': 0.0849, '2018-08': -0.0347, '2018-09': -0.0189, '2018-10': 0.2337,
        '2018-11': 0.0910, '2018-12': 0.0567, '2019-01': 0.0740, '2019-02': 0.0439, '2019-03': -0.0159,
        '2019-04': 0.0400, '2019-05': 0.0436, '2019-06': 0.0292, '2019-07': 0.0628, '2019-08': 0.0211,
        '2019-09': 0.0026, '2019-10': -0.0031, '2019-11': 0.0062, '2019-12': 0.1877, '2020-01': 0.0218,
        '2020-02': -0.0712, '2020-03': -0.2893, '2020-04': 0.0877, '2020-05': 0.0870, '2020-06': 0.2435,
        '2020-07': 0.1347, '2020-08': -0.0231, '2020-09': -0.0698, '2020-10': -0.0182, '2020-11': 0.1111,
        '2020-12': 0.0679, '2021-01': -0.0318, '2021-02': 0.0735, '2021-03': 0.0256, '2021-04': 0.1305,
        '2021-05': 0.0556, '2021-06': 0.0386, '2021-07': -0.0030, '2021-08': -0.0730, '2021-09': 0.0232,
        '2021-10': -0.0251, '2021-11': -0.0714, '2021-12': 0.1015, '2022-01': 0.0161, '2022-02': 0.0575,
        '2022-03': 0.0852, '2022-04': -0.0551, '2022-05': -0.0043, '2022-06': -0.1106, '2022-07': 0.1145,
        '2022-08': 0.1226, '2022-09': 0.0448, '2022-10': 0.1258, '2022-11': -0.1150, '2022-12': -0.0507,
        '2023-01': 0.0267, '2023-02': -0.0239, '2023-03': -0.0389, '2023-04': 0.0543, '2023-05': 0.1624,
        '2023-06': 0.1054, '2023-07': 0.0663, '2023-08': -0.0135, '2023-09': 0.0918, '2023-10': -0.0646,
        '2023-11': 0.0737, '2023-12': 0.1090, '2024-01': -0.0577, '2024-02': 0.0239, '2024-03': 0.0869,
        '2024-04': -0.0287, '2024-05': -0.0398, '2024-06': 0.0033, '2024-07': 0.0237, '2024-08': 0.0406,
        '2024-09': 0.0123, '2024-10': 0.0088, '2024-11': -0.0331, '2024-12': -0.0163, '2025-01': 0.0702,
        '2025-02': 0.0441, '2025-03': 0.0856, '2025-04': 0.0614, '2025-05': 0.0104, '2025-06': -0.0004,
        '2025-07': -0.0560, '2025-08': 0.0348, '2025-09': 0.0005, '2025-10': 0.0179, '2025-11': 0.0966,
        '2025-12': 0.0112
    }

    absolute_returns = {
        '2018-12': 0.0262, '2019-01': 0.1408, '2019-02': 0.0436, '2019-03': 0.0365, '2019-04': 0.0325,
        '2019-05': 0.0276, '2019-06': 0.0519, '2019-07': 0.0634, '2019-08': 0.0237, '2019-09': 0.0163,
        '2019-10': 0.0566, '2019-11': 0.0319, '2019-12': 0.1364, '2020-01': 0.0217, '2020-02': -0.0749,
        '2020-03': -0.2324, '2020-04': 0.1233, '2020-05': 0.0584, '2020-06': 0.0880, '2020-07': 0.0882,
        '2020-08': -0.0116, '2020-09': -0.0700, '2020-10': -0.0033, '2020-11': 0.1473, '2020-12': 0.0649,
        '2021-01': -0.0218, '2021-02': -0.0187, '2021-03': 0.0666, '2021-04': 0.0615, '2021-05': 0.0567,
        '2021-06': 0.0124, '2021-07': -0.0262, '2021-08': -0.0108, '2021-09': -0.0227, '2021-10': -0.0513,
        '2021-11': 0.0265, '2021-12': 0.0534, '2022-01': 0.0529, '2022-02': -0.0020, '2022-03': 0.0700,
        '2022-04': -0.0121, '2022-05': 0.0653, '2022-06': -0.0795, '2022-07': 0.0288, '2022-08': 0.0512,
        '2022-09': -0.0237, '2022-10': 0.0343, '2022-11': -0.0033, '2022-12': -0.0099, '2023-01': 0.0046,
        '2023-02': -0.0233, '2023-03': -0.0032, '2023-04': 0.0240, '2023-05': 0.0516, '2023-06': 0.1085,
        '2023-07': 0.0597, '2023-08': -0.0308, '2023-09': -0.0060, '2023-10': -0.0380, '2023-11': 0.1039,
        '2023-12': 0.0487, '2024-01': -0.0009, '2024-02': 0.0306, '2024-03': 0.0159, '2024-04': -0.0361,
        '2024-05': -0.0195, '2024-06': 0.0137, '2024-07': 0.0344, '2024-08': 0.0624, '2024-09': -0.0258,
        '2024-10': -0.0225, '2024-11': -0.0373, '2024-12': -0.0377, '2025-01': 0.0652, '2025-02': -0.0108,
        '2025-03': 0.0312, '2025-04': 0.0608, '2025-05': 0.0866, '2025-06': 0.0010, '2025-07': -0.0305,
        '2025-08': 0.0576, '2025-09': 0.0532, '2025-10': 0.0047, '2025-11': 0.0779, '2025-12': -0.0207
    }

    sparta_returns = {
        '2018-09': 0.0037, '2018-10': 0.0049, '2018-11': 0.0066, '2018-12': 0.0058, '2019-01': 0.0073,
        '2019-02': 0.0075, '2019-03': 0.0066, '2019-04': 0.0060, '2019-05': 0.0070, '2019-06': 0.0062,
        '2019-07': 0.0039, '2019-08': 0.0021, '2019-09': 0.0015, '2019-10': -0.0060, '2019-11': -0.0126,
        '2019-12': 0.0080, '2020-01': 0.0077, '2020-02': 0.0022, '2020-03': -0.0112, '2020-04': 0.0022,
        '2020-05': 0.0000, '2020-06': 0.0075, '2020-07': 0.0096, '2020-08': 0.0072, '2020-09': 0.0034,
        '2020-10': 0.0084, '2020-11': 0.0027, '2020-12': 0.0061, '2021-01': 0.0087, '2021-02': 0.0050,
        '2021-03': 0.0100, '2021-04': 0.0060, '2021-05': 0.0051, '2021-06': 0.0116, '2021-07': 0.0079,
        '2021-08': 0.0089, '2021-09': 0.0056, '2021-10': 0.0082, '2021-11': 0.0100, '2021-12': 0.0075,
        '2022-01': 0.0091, '2022-02': 0.0100, '2022-03': 0.0110, '2022-04': 0.0096, '2022-05': 0.0155,
        '2022-06': 0.0113, '2022-07': 0.0110, '2022-08': 0.0106, '2022-09': 0.0091, '2022-10': 0.0014,
        '2022-11': 0.0068, '2022-12': 0.0108, '2023-01': 0.0040, '2023-02': -0.0102, '2023-03': 0.0131,
        '2023-04': 0.0088, '2023-05': 0.0297, '2023-06': 0.0249, '2023-07': 0.0257, '2023-08': 0.0187,
        '2023-09': 0.0098, '2023-10': 0.0103, '2023-11': 0.0121, '2023-12': 0.0117, '2024-01': 0.0166,
        '2024-02': 0.0243, '2024-03': 0.0105, '2024-04': 0.0057, '2024-05': 0.0093, '2024-06': 0.0075,
        '2024-07': 0.0118, '2024-08': 0.0124, '2024-09': 0.0129, '2024-10': 0.0097, '2024-11': 0.0036,
        '2024-12': 0.0070, '2025-01': 0.0101, '2025-02': 0.0130, '2025-03': 0.0110, '2025-04': 0.0130,
        '2025-05': 0.0107, '2025-06': 0.0157, '2025-07': 0.0147, '2025-08': 0.0180, '2025-09': 0.0211,
        '2025-10': 0.0060, '2025-11': 0.0103, '2025-12': 0.0095
    }

    df = pd.DataFrame({
        'Tarpon GT': pd.Series(tarpon_returns),
        'Absolute Pace': pd.Series(absolute_returns),
        'Sparta Infra': pd.Series(sparta_returns)
    })

    # Converter índice string 'YYYY-MM' para datetime (final do mês)
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp('M')
    return df

# ==========================================
# 2. FUNÇÕES DE DADOS (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    """
    Baixa dados do Yahoo Finance e calcula retornos mensais.
    Adiciona sufixo .SA se necessário para ativos brasileiros.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Tratamento simples para garantir .SA (exceto para ETFs globais comuns conhecidos, 
    # mas assumiremos BR por padrão dado o contexto dos FIIs)
    processed_tickers = []
    for t in tickers:
        t = t.strip().upper()
        # Se não tiver ponto (ex: AAPL) e parecer ticker B3 (numérico no fim), adiciona .SA
        # Para ETFs internacionais (ex: IVV) listados lá fora, o usuário deve digitar o ticker correto.
        # Aqui assumimos contexto Brasil prioritário conforme prompt.
        if "." not in t and any(char.isdigit() for char in t): 
            processed_tickers.append(f"{t}.SA")
        else:
            processed_tickers.append(t)
            
    try:
        prices = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # Se for apenas um ticker, o yfinance retorna Series, converter para DF
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=processed_tickers[0])
            
        # Reamostragem Mensal (Pegar último preço do mês e calcular retorno simples)
        # Retorno Simples vs Log:
        # Usaremos Retorno Simples para agregação de portfólio (Média Ponderada).
        # Retornos logarítmicos não são aditivos através de ativos (cross-sectional), apenas no tempo.
        monthly_prices = prices.resample('ME').last() 
        returns = monthly_prices.pct_change()
        
        # Limpar tickers nas colunas (remover .SA para display)
        returns.columns = [c.replace('.SA', '') for c in returns.columns]
        
        return returns
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

# ==========================================
# 3. LÓGICA DE BACKTEST E MÉTRICAS
# ==========================================
def calculate_portfolio_performance(returns_df, weights, rebalance_freq='Mensal'):
    """
    Calcula a evolução do portfólio com rebalanceamento.
    
    Args:
        returns_df: DataFrame com retornos mensais de todos os ativos/classes.
        weights: Dicionário {ativo: peso_decimal}.
        rebalance_freq: 'Mensal' ou 'Anual'.
    """
    # Filtra apenas datas onde temos dados (drop na primeira linha de NaN do pct_change)
    returns_df = returns_df.dropna()
    
    # Alinha pesos com colunas disponíveis
    # Normaliza pesos caso o usuário não some 100% (segurança)
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    
    if not available_assets:
        return None, None

    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() # Normalização
    
    # Simulação da Curva de Equity
    # Começa com 100 base
    portfolio_value = [100.0]
    monthly_returns = []
    
    # Pesos correntes (flutuam se não rebalancear)
    current_weights = active_weights.copy()
    
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        # Retorno do mês t para cada ativo
        r_t = asset_returns_np[i]
        
        # Retorno do portfólio no mês
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        
        # Atualiza valor do portfólio
        new_value = portfolio_value[-1] * (1 + port_ret)
        portfolio_value.append(new_value)
        
        # Atualiza pesos (drift)
        # O peso do ativo i vira: w_i * (1+r_i) / (1+port_ret)
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        
        # Lógica de Rebalanceamento
        is_rebalance_time = False
        current_date = dates[i]
        
        if rebalance_freq == 'Mensal':
            is_rebalance_time = True
        elif rebalance_freq == 'Anual' and current_date.month == 12:
            is_rebalance_time = True
            
        if is_rebalance_time:
            current_weights = active_weights.copy()
            
    portfolio_series = pd.Series(portfolio_value[1:], index=dates)
    monthly_returns_series = pd.Series(monthly_returns, index=dates)
    
    return portfolio_series, monthly_returns_series

def calculate_metrics(daily_returns_series, rf_rate_annual=0.10):
    """Calcula métricas de risco e retorno."""
    # CAGR
    total_ret = (1 + daily_returns_series).prod() - 1
    n_years = len(daily_returns_series) / 12
    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    
    # Volatilidade (Anualizada)
    vol = daily_returns_series.std() * np.sqrt(12)
    
    # Sharpe (Simplificado: Rf constante)
    rf_monthly = (1 + rf_rate_annual)**(1/12) - 1
    excess_ret = daily_returns_series - rf_monthly
    sharpe = (excess_ret.mean() / daily_returns_series.std()) * np.sqrt(12) if daily_returns_series.std() != 0 else 0
    
    # Max Drawdown
    cum_returns = (1 + daily_returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    return cagr, vol, sharpe, max_dd, total_ret

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Asset Allocator Pro", layout="wide", page_icon="📈")

# --- CSS Customizado para visual profissional ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    h1 {color: #2c3e50;}
    h3 {color: #34495e;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Backtest Institucional Multiclasse")
st.markdown("Ferramenta de alocação de ativos e análise de risco (Ações, FIIs, ETFs e Fundos Ativos).")

# --- SIDEBAR: Configurações ---
with st.sidebar:
    st.header("⚙️ Configurações de Backtest")
    
    # Datas
    min_date = datetime(2018, 1, 1)
    max_date = datetime(2025, 12, 31)
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", min_date, min_value=min_date, max_value=max_date)
    end_date = col_d2.date_input("Fim", datetime.today(), min_value=min_date, max_value=max_date)
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate = st.number_input("Taxa Livre de Risco Anual (%)", value=10.0, step=0.5) / 100

    st.markdown("---")
    st.header("📦 Seleção de Ativos")
    
    # Inputs de Tickers (Default baseados nos arquivos do usuário)
    default_stocks = "PRIO3, PSSA3, SAPR4, SBSP3, AGRO3, B3SA3, BBAS3, BBSE3"
    default_fiis = "MXRF11, ALZR11, BRCO11, BTLG11, PMLL11, TRXF11"
    default_etfs = "GPUS11, VWRA11" # Considerando ETFs globais listados na B3 ou lá fora
    
    stocks_input = st.text_area("Ações (separadas por vírgula)", default_stocks)
    fiis_input = st.text_area("FIIs (separadas por vírgula)", default_fiis)
    etfs_input = st.text_area("ETFs (separadas por vírgula)", default_etfs)
    
    st.markdown("---")
    st.header("⚖️ Alocação de Pesos (%)")
    st.info("A soma deve ser 100%. Se diferir, será normalizado.")
    
    w_stocks = st.slider("Carteira Ações", 0, 100, 25)
    w_fiis = st.slider("Carteira FIIs", 0, 100, 25)
    w_etfs = st.slider("Carteira ETFs", 0, 100, 20)
    w_tarpon = st.number_input("Tarpon GT Master", 0, 100, 10)
    w_absolute = st.number_input("Absolute Pace", 0, 100, 10)
    w_sparta = st.number_input("Sparta Infra", 0, 100, 10)
    
    total_weight = w_stocks + w_fiis + w_etfs + w_tarpon + w_absolute + w_sparta
    if total_weight != 100:
        st.warning(f"Soma atual: {total_weight}%. Será normalizado para 100%.")

# --- PROCESSAMENTO ---

# 1. Carregar Fundos Hardcoded
df_funds = get_hardcoded_funds()

# 2. Carregar Dados de Mercado
stock_list = [x.strip() for x in stocks_input.split(',') if x.strip()]
fii_list = [x.strip() for x in fiis_input.split(',') if x.strip()]
etf_list = [x.strip() for x in etfs_input.split(',') if x.strip()]

with st.spinner('Baixando dados de mercado e consolidando...'):
    df_stocks = get_market_data(stock_list, start_date, end_date)
    df_fiis = get_market_data(fii_list, start_date, end_date)
    df_etfs = get_market_data(etf_list, start_date, end_date)

# 3. Consolidar Classes (Equal Weight dentro da classe)
consolidated_returns = pd.DataFrame(index=df_funds.index)

# Função auxiliar para média da classe
def get_class_return(df, name):
    if not df.empty:
        # Média simples dos retornos dos ativos selecionados (Equal Weight na classe)
        return df.mean(axis=1)
    return pd.Series(dtype=float)

# Alinhar índices (Datas) - Join Outer para garantir todas as datas possíveis, depois fillna
# Mas para consistência, vamos usar o índice dos fundos como base (mensal) e fazer merge
all_dates = df_funds.index.union(df_stocks.index).union(df_fiis.index).union(df_etfs.index)
all_dates = all_dates.sort_values()

# Criar DataFrame mestre de retornos de CLASSES
master_df = pd.DataFrame(index=all_dates)

# Inserir retornos calculados (Preencher NaN com 0 não é ideal, melhor forward fill ou drop. 
# Backtest rigoroso: se não tem dado, não investe. Vamos usar dropna no final)

if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)

# Inserir fundos hardcoded (reindexando para garantir alinhamento)
master_df['Tarpon GT'] = df_funds['Tarpon GT']
master_df['Absolute Pace'] = df_funds['Absolute Pace']
master_df['Sparta Infra'] = df_funds['Sparta Infra']

# Filtrar pelo range selecionado pelo usuário
mask = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
master_df = master_df.loc[mask].dropna()

# 4. Calcular Portfólio
weights_dict = {
    'Ações Consolidadas': w_stocks,
    'FIIs Consolidados': w_fiis,
    'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon,
    'Absolute Pace': w_absolute,
    'Sparta Infra': w_sparta
}

if master_df.empty:
    st.error("Dados insuficientes para o período ou ativos selecionados.")
else:
    portfolio_equity, portfolio_ret = calculate_portfolio_performance(master_df, weights_dict, rebalance_freq)
    
    # Calcular Métricas
    cagr, vol, sharpe, max_dd, tot_ret = calculate_metrics(portfolio_ret, rf_rate)

    # --- DASHBOARD PRINCIPAL ---
    
    # 1. Métricas de Topo
    st.markdown("### 📈 Performance Consolidada")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Retorno Total", f"{tot_ret:.1%}")
    kpi2.metric("CAGR", f"{cagr:.1%}")
    kpi3.metric("Volatilidade (a.a.)", f"{vol:.1%}")
    kpi4.metric(f"Sharpe (Rf={rf_rate:.0%})", f"{sharpe:.2f}")
    kpi5.metric("Max Drawdown", f"{max_dd:.1%}", delta_color="inverse")

    st.markdown("---")

    # 2. Gráficos
    tab1, tab2, tab3 = st.tabs(["Evolução Patrimonial", "Drawdown & Anual", "Correlações"])

    with tab1:
        st.subheader("Curva de Equity (Base 100)")
        fig_equity = px.line(portfolio_equity, title="Crescimento do Patrimônio")
        fig_equity.update_layout(xaxis_title="Data", yaxis_title="Valor Acumulado", template="plotly_white")
        
        # Adicionar benchmark simples (CDI ou IBOV seria ideal, aqui vamos plotar apenas o portfólio para clareza)
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab2:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Drawdown Histórico")
            cum_ret = (1 + portfolio_ret).cumprod()
            peak = cum_ret.cummax()
            dd_series = (cum_ret - peak) / peak
            
            fig_dd = px.area(dd_series, title="Drawdown Submarino")
            fig_dd.update_traces(fillcolor='red', line_color='red')
            fig_dd.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)
            
        with col_g2:
            st.subheader("Retornos Anuais")
            annual_ret = portfolio_ret.resample('YE').apply(lambda x: (1+x).prod() -1)
            # Formatar índice para ano string
            annual_ret.index = annual_ret.index.year.astype(str)
            
            fig_bar = px.bar(annual_ret, title="Retorno por Ano Calendário", text_auto='.1%')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Heatmap de Correlação")
        corr_matrix = master_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("**Nota:** A correlação é calculada com base nos retornos mensais das classes consolidadas e fundos individuais.")

# --- EXPORTAÇÃO ---
st.sidebar.markdown("---")
if not master_df.empty:
    csv = master_df.to_csv().encode('utf-8')
    st.sidebar.download_button("📥 Baixar Dados Consolidados", data=csv, file_name="backtest_data.csv", mime="text/csv")
