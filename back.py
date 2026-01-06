import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Hard-coded monthly returns for active funds (decimal format)
# Data extracted from reliable sources like investidor10.com.br
# Converted percentages to decimals
tarpon_returns = {
    '2018-01': 0.0518, '2018-02': 0.0018, '2018-03': 0.0337, '2018-04': -0.0224, '2018-05': -0.1091,
    '2018-06': -0.0884, '2018-07': 0.0849, '2018-08': -0.0347, '2018-09': -0.0189, '2018-10': 0.2337,
    '2018-11': 0.0910, '2018-12': 0.0567,
    '2019-01': 0.0740, '2019-02': 0.0439, '2019-03': -0.0159, '2019-04': 0.0400, '2019-05': 0.0436,
    '2019-06': 0.0292, '2019-07': 0.0628, '2019-08': 0.0211, '2019-09': 0.0026, '2019-10': -0.0031,
    '2019-11': 0.0062, '2019-12': 0.1877,
    '2020-01': 0.0218, '2020-02': -0.0712, '2020-03': -0.2893, '2020-04': 0.0877, '2020-05': 0.0870,
    '2020-06': 0.2435, '2020-07': 0.1347, '2020-08': -0.0231, '2020-09': -0.0698, '2020-10': -0.0182,
    '2020-11': 0.1111, '2020-12': 0.0679,
    '2021-01': -0.0318, '2021-02': 0.0735, '2021-03': 0.0256, '2021-04': 0.1305, '2021-05': 0.0556,
    '2021-06': 0.0386, '2021-07': -0.0030, '2021-08': -0.0730, '2021-09': 0.0232, '2021-10': -0.0251,
    '2021-11': -0.0714, '2021-12': 0.1015,
    '2022-01': 0.0161, '2022-02': 0.0575, '2022-03': 0.0852, '2022-04': -0.0551, '2022-05': -0.0043,
    '2022-06': -0.1106, '2022-07': 0.1145, '2022-08': 0.1226, '2022-09': 0.0448, '2022-10': 0.1258,
    '2022-11': -0.1150, '2022-12': -0.0507,
    '2023-01': 0.0267, '2023-02': -0.0239, '2023-03': -0.0389, '2023-04': 0.0543, '2023-05': 0.1624,
    '2023-06': 0.1054, '2023-07': 0.0663, '2023-08': -0.0135, '2023-09': 0.0918, '2023-10': -0.0646,
    '2023-11': 0.0737, '2023-12': 0.1090,
    '2024-01': -0.0577, '2024-02': 0.0239, '2024-03': 0.0869, '2024-04': -0.0287, '2024-05': -0.0398,
    '2024-06': 0.0033, '2024-07': 0.0237, '2024-08': 0.0406, '2024-09': 0.0123, '2024-10': 0.0088,
    '2024-11': -0.0331, '2024-12': -0.0163,
    '2025-01': 0.0702, '2025-02': 0.0441, '2025-03': 0.0856, '2025-04': 0.0614, '2025-05': 0.0104,
    '2025-06': -0.0004, '2025-07': -0.0560, '2025-08': 0.0348, '2025-09': 0.0005, '2025-10': 0.0179,
    '2025-11': 0.0966, '2025-12': 0.0112
}

absolute_returns = {
    '2018-12': 0.0262,
    '2019-01': 0.1408, '2019-02': 0.0436, '2019-03': 0.0365, '2019-04': 0.0325, '2019-05': 0.0276,
    '2019-06': 0.0519, '2019-07': 0.0634, '2019-08': 0.0237, '2019-09': 0.0163, '2019-10': 0.0566,
    '2019-11': 0.0319, '2019-12': 0.1364,
    '2020-01': 0.0217, '2020-02': -0.0749, '2020-03': -0.2324, '2020-04': 0.1233, '2020-05': 0.0584,
    '2020-06': 0.0880, '2020-07': 0.0882, '2020-08': -0.0116, '2020-09': -0.0700, '2020-10': -0.0033,
    '2020-11': 0.1473, '2020-12': 0.0649,
    '2021-01': -0.0218, '2021-02': -0.0187, '2021-03': 0.0666, '2021-04': 0.0615, '2021-05': 0.0567,
    '2021-06': 0.0124, '2021-07': -0.0262, '2021-08': -0.0108, '2021-09': -0.0227, '2021-10': -0.0513,
    '2021-11': 0.0265, '2021-12': 0.0534,
    '2022-01': 0.0529, '2022-02': -0.0020, '2022-03': 0.0700, '2022-04': -0.0121, '2022-05': 0.0653,
    '2022-06': -0.0795, '2022-07': 0.0288, '2022-08': 0.0512, '2022-09': -0.0237, '2022-10': 0.0343,
    '2022-11': -0.0033, '2022-12': -0.0099,
    '2023-01': 0.0046, '2023-02': -0.0233, '2023-03': -0.0032, '2023-04': 0.0240, '2023-05': 0.0516,
    '2023-06': 0.1085, '2023-07': 0.0597, '2023-08': -0.0308, '2023-09': -0.0060, '2023-10': -0.0380,
    '2023-11': 0.1039, '2023-12': 0.0487,
    '2024-01': -0.0009, '2024-02': 0.0306, '2024-03': 0.0159, '2024-04': -0.0361, '2024-05': -0.0195,
    '2024-06': 0.0137, '2024-07': 0.0344, '2024-08': 0.0624, '2024-09': -0.0258, '2024-10': -0.0225,
    '2024-11': -0.0373, '2024-12': -0.0377,
    '2025-01': 0.0652, '2025-02': -0.0108, '2025-03': 0.0312, '2025-04': 0.0608, '2025-05': 0.0866,
    '2025-06': 0.0010, '2025-07': -0.0305, '2025-08': 0.0576, '2025-09': 0.0532, '2025-10': 0.0047,
    '2025-11': 0.0779, '2025-12': -0.0207
}

sparta_returns = {
    '2018-09': 0.0037, '2018-10': 0.0049, '2018-11': 0.0066, '2018-12': 0.0058,
    '2019-01': 0.0073, '2019-02': 0.0075, '2019-03': 0.0066, '2019-04': 0.0060, '2019-05': 0.0070,
    '2019-06': 0.0062, '2019-07': 0.0039, '2019-08': 0.0021, '2019-09': 0.0015, '2019-10': -0.0060,
    '2019-11': -0.0126, '2019-12': 0.0080,
    '2020-01': 0.0077, '2020-02': 0.0022, '2020-03': -0.0112, '2020-04': 0.0022, '2020-05': 0.0000,
    '2020-06': 0.0075, '2020-07': 0.0096, '2020-08': 0.0072, '2020-09': 0.0034, '2020-10': 0.0084,
    '2020-11': 0.0027, '2020-12': 0.0061,
    '2021-01': 0.0087, '2021-02': 0.0050, '2021-03': 0.0100, '2021-04': 0.0060, '2021-05': 0.0051,
    '2021-06': 0.0116, '2021-07': 0.0079, '2021-08': 0.0089, '2021-09': 0.0056, '2021-10': 0.0082,
    '2021-11': 0.0100, '2021-12': 0.0075,
    '2022-01': 0.0091, '2022-02': 0.0100, '2022-03': 0.0110, '2022-04': 0.0096, '2022-05': 0.0155,
    '2022-06': 0.0113, '2022-07': 0.0110, '2022-08': 0.0106, '2022-09': 0.0091, '2022-10': 0.0014,
    '2022-11': 0.0068, '2022-12': 0.0108,
    '2023-01': 0.0040, '2023-02': -0.0102, '2023-03': 0.0131, '2023-04': 0.0088, '2023-05': 0.0297,
    '2023-06': 0.0249, '2023-07': 0.0257, '2023-08': 0.0187, '2023-09': 0.0098, '2023-10': 0.0103,
    '2023-11': 0.0121, '2023-12': 0.0117,
    '2024-01': 0.0166, '2024-02': 0.0243, '2024-03': 0.0105, '2024-04': 0.0057, '2024-05': 0.0093,
    '2024-06': 0.0075, '2024-07': 0.0118, '2024-08': 0.0124, '2024-09': 0.0129, '2024-10': 0.0097,
    '2024-11': 0.0036, '2024-12': 0.0070,
    '2025-01': 0.0101, '2025-02': 0.0130, '2025-03': 0.0110, '2025-04': 0.0130, '2025-05': 0.0107,
    '2025-06': 0.0157, '2025-07': 0.0147, '2025-08': 0.0180, '2025-09': 0.0211, '2025-10': 0.0060,
    '2025-11': 0.0103, '2025-12': 0.0095
}

# Convert to DataFrame
def create_funds_df():
    dates = pd.date_range(start='2018-01-01', end='2025-12-31', freq='M')
    df = pd.DataFrame(index=dates)
    for fund, rets in zip(['Tarpon', 'Absolute', 'Sparta'], [tarpon_returns, absolute_returns, sparta_returns]):
        for date_str, ret in rets.items():
            year, month = map(int, date_str.split('-'))
            date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            df.at[date, fund] = ret
    return df.dropna(how='all')

# List of assets from the document
acoes_list = ['AGRO3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'CMIG3.SA', 'EGIE3.SA', 'ITUB3.SA', 
              'PRIO3.SA', 'PSSA3.SA', 'SAPR4.SA', 'SBSP3.SA', 'TAEE3.SA', 'TOTS3.SA', 'VIVT3.SA', 'WEGE3.SA']
fiis_list = ['ALZR11.SA', 'BRCO11.SA', 'BTLG11.SA', 'HGLG11.SA', 'HGRE11.SA', 'HGRU11.SA', 'KNCR11.SA', 'KNRI11.SA', 
             'LVBI11.SA', 'MXRF11.SA', 'PMLL11.SA', 'TRXF11.SA', 'VILG11.SA', 'VISC11.SA', 'XPLG11.SA', 'XPML11.SA']
etfs_list = ['GPUS11.SA', 'VWRA11.SA', 'IVV']

# Function to download data and calculate monthly returns
def download_returns(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    monthly = data.resample('M').last()
    returns = monthly.pct_change().dropna()  # Simple returns
    return returns

# Function to calculate class returns (equal weight)
def class_returns(returns_df, selected_tickers):
    if not selected_tickers:
        return pd.Series(0, index=returns_df.index)
    return returns_df[selected_tickers].mean(axis=1)

# Backtest function with rebalancing
def backtest_portfolio(class_returns_df, weights, rebalance_freq='monthly', start_value=100):
    # class_returns_df: columns = ['Acoes', 'FIIs', 'ETFs', 'Fundos']
    # weights: dict {'Acoes': w, ...}
    # Rebalance every 1 (monthly) or 12 (annual) periods
    if rebalance_freq == 'annual':
        rebal_period = 12
    else:
        rebal_period = 1
    
    portfolio_value = [start_value]
    current_weights = np.array([weights['Acoes'], weights['FIIs'], weights['ETFs'], weights['Fundos']])
    for i in range(1, len(class_returns_df) + 1):
        period_returns = class_returns_df.iloc[i-1].values
        new_value = portfolio_value[-1] * (1 + np.dot(current_weights, period_returns))
        portfolio_value.append(new_value)
        if i % rebal_period == 0:
            current_weights = np.array([weights['Acoes'], weights['FIIs'], weights['ETFs'], weights['Fundos']])  # Rebalance to target
    
    equity = pd.Series(portfolio_value[1:], index=class_returns_df.index)
    port_returns = equity.pct_change().dropna()
    return equity, port_returns

# Calculate metrics
def calculate_metrics(returns, rf_rate=0.0):
    cum_return = (returns.cumprod().iloc[-1] - 1) if not returns.empty else 0
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (1 + cum_return) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(12) if not returns.empty else 0
    monthly_rf = rf_rate / 12
    sharpe = (returns.mean() - monthly_rf) / returns.std() * np.sqrt(12) if returns.std() != 0 else 0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0
    return cum_return, cagr, vol, sharpe, max_dd

# Plots
def plot_equity(equity):
    fig = px.line(equity, title='Evolução do Patrimônio')
    return fig

def plot_drawdown(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    fig = px.line(drawdown, title='Drawdown Histórico (%)')
    return fig

def plot_annual_returns(returns):
    annual = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
    fig = px.bar(annual, title='Retornos Anuais (%)')
    return fig

def plot_corr(class_returns_df):
    fig = go.Figure(data=go.Heatmap(z=class_returns_df.corr(), x=class_returns_df.columns, y=class_returns_df.columns, colorscale='RdBu'))
    fig.update_layout(title='Heatmap de Correlação')
    return fig

# Main Streamlit app
st.title('Backtest de Portfólio Multiclasse')

# Sidebar
with st.sidebar:
    st.header('Configurações')
    start_date = st.date_input('Data Inicial', value=datetime(2018, 1, 1))
    end_date = st.date_input('Data Final', value=datetime(2025, 12, 31))
    rebalance_freq = st.selectbox('Frequência de Rebalanceamento', ['monthly', 'annual'])
    rf_rate = st.number_input('Taxa Livre de Risco Anual (%)', value=5.0) / 100
    
    st.subheader('Pesos das Classes (%)')
    acoes_weight = st.slider('Ações', 0, 100, 25) / 100
    fiis_weight = st.slider('FIIs', 0, 100, 25) / 100
    etfs_weight = st.slider('ETFs', 0, 100, 25) / 100
    fundos_weight = st.slider('Fundos Ativos', 0, 100, 25) / 100
    
    if acoes_weight + fiis_weight + etfs_weight + fundos_weight != 1:
        st.warning('Pesos das classes devem somar 100%')
    
    st.subheader('Seleção de Ativos')
    selected_acoes = st.multiselect('Ações', acoes_list, default=acoes_list[:5])
    selected_fiis = st.multiselect('FIIs', fiis_list, default=fiis_list[:5])
    selected_etfs = st.multiselect('ETFs', etfs_list, default=etfs_list[:2])
    
    st.subheader('Pesos dos Fundos Ativos (%)')
    tarpon_w = st.slider('Tarpon GT Master FIP', 0, 100, 33) / 100
    absolute_w = st.slider('Absolute Pace Long Biased Master FIP Ações', 0, 100, 33) / 100
    sparta_w = st.slider('Sparta Master A Incentivado Investimento Financeiro Infra RF RL', 0, 100, 34) / 100
    
    if tarpon_w + absolute_w + sparta_w != 1:
        st.warning('Pesos dos fundos devem somar 100%')

# Main panel
if st.button('Executar Backtest'):
    # Download data
    acoes_returns = download_returns(selected_acoes, start_date, end_date)
    fiis_returns = download_returns(selected_fiis, start_date, end_date)
    etfs_returns = download_returns(selected_etfs, start_date, end_date)
    
    # Funds
    funds_df = create_funds_df()
    funds_df = funds_df.loc[(funds_df.index >= pd.to_datetime(start_date)) & (funds_df.index <= pd.to_datetime(end_date))]
    fundos_returns = funds_df['Tarpon'] * tarpon_w + funds_df['Absolute'] * absolute_w + funds_df['Sparta'] * sparta_w
    
    # Align all to common index (monthly)
    common_index = funds_df.index  # Use funds as base, since they are monthly
    acoes_class_ret = class_returns(acoes_returns.reindex(common_index), selected_acoes).ffill().fillna(0)
    fiis_class_ret = class_returns(fiis_returns.reindex(common_index), selected_fiis).ffill().fillna(0)
    etfs_class_ret = class_returns(etfs_returns.reindex(common_index), selected_etfs).ffill().fillna(0)
    fundos_class_ret = fundos_returns.reindex(common_index).ffill().fillna(0)
    
    class_returns_df = pd.DataFrame({
        'Acoes': acoes_class_ret,
        'FIIs': fiis_class_ret,
        'ETFs': etfs_class_ret,
        'Fundos': fundos_class_ret
    }).dropna(how='all')
    
    weights = {'Acoes': acoes_weight, 'FIIs': fiis_weight, 'ETFs': etfs_weight, 'Fundos': fundos_weight}
    
    # Backtest
    equity, port_returns = backtest_portfolio(class_returns_df, weights, rebalance_freq)
    
    # Metrics
    cum_ret, cagr, vol, sharpe, max_dd = calculate_metrics(port_returns, rf_rate)
    st.subheader('Métricas')
    metrics_df = pd.DataFrame({
        'Métrica': ['Retorno Acumulado', 'CAGR', 'Volatilidade Anualizada', 'Índice de Sharpe', 'Máximo Drawdown'],
        'Valor': [f'{cum_ret*100:.2f}%', f'{cagr*100:.2f}%', f'{vol*100:.2f}%', f'{sharpe:.2f}', f'{max_dd*100:.2f}%']
    })
    st.table(metrics_df)
    
    # Correlations
    st.subheader('Correlação entre Classes')
    st.plotly_chart(plot_corr(class_returns_df))
    
    # Visualizations
    st.subheader('Evolução do Patrimônio')
    st.plotly_chart(plot_equity(equity))
    
    st.subheader('Drawdown Histórico')
    st.plotly_chart(plot_drawdown(port_returns))
    
    st.subheader('Retornos Anuais')
    st.plotly_chart(plot_annual_returns(port_returns))

# Justification for returns: Simple returns are used as they accurately represent the multiplicative nature of portfolio growth over time.
# Log returns could be used for additive properties in volatility calculations, but simple returns are standard for backtesting equity curves.
