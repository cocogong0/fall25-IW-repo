import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_sp500_tickers(top_x_sectors=11, top_k_stocks=5):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    sp500_table = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})[0]
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)

    tickers_list = sp500_table['Symbol'].tolist()
    
    caps = []
    for ticker in tickers_list:
        t = yf.Ticker(ticker)
        m_cap = t.info.get('marketCap', 0)
        caps.append(m_cap)

    sp500_table['MarketCap'] = caps

    sp500_table = sp500_table.sort_values(by='MarketCap', ascending=False)

    top_sectors = sp500_table['GICS Sector'].value_counts().nlargest(top_x_sectors).index.tolist()
    
    ticker_map = {}
    for sector in top_sectors:
        sector_stocks = sp500_table[sp500_table['GICS Sector'] == sector].head(top_k_stocks)
        
        for _, row in sector_stocks.iterrows():
            ticker_map[row['Symbol']] = row['GICS Sector']
            
    return ticker_map

def download_panel_data(ticker_map, start_date='1990-01-01'):
    data = yf.download(list(ticker_map.keys()), start=start_date, auto_adjust=True, group_by='ticker')
    
    clean_tickers = [t for t in ticker_map.keys() if t in data.columns.levels[0]]
    panel_df = data[clean_tickers].stack(level=0, future_stack=True)
    panel_df.index.names = ['Date', 'Ticker']
    panel_df.reset_index(inplace=True)
    
    panel_df['Date'] = pd.to_datetime(panel_df['Date']).dt.tz_localize(None)

    panel_df['Sector'] = panel_df['Ticker'].map(ticker_map)
    
    return panel_df

def engineer_features(panel_df):

    # log returns/momentum
    panel_df['Ret_1d'] = panel_df.groupby('Ticker')['Close'].transform(lambda x: np.log(x).diff())
    panel_df['Ret_21d'] = panel_df.groupby('Ticker')['Close'].transform(lambda x: np.log(x).diff(21))

    # relative strength index
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    panel_df['RSI_14d'] = panel_df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

    # price vs moving average/trend
    panel_df['Price_vs_MA200'] = (
        panel_df['Close'] / panel_df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    ) - 1

    # relative volume
    panel_df['Rel_Volume'] = (
        panel_df['Volume'] /
        panel_df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
    )

    # sector dispersion
    panel_df['Ret1d_Std_Dev'] = panel_df.groupby(['Date', 'Sector'])['Ret_1d'].transform('std')

    return panel_df.dropna()

def define_regimes(start_date='1990-01-01'):
    spy = yf.download('^GSPC', start=start_date, progress=False, auto_adjust=True)['Close']
    if isinstance(spy, pd.DataFrame): 
        spy = spy.iloc[:, 0]
    
    market_df = pd.DataFrame(spy)
    market_df.columns = ['Market_Index']
    
    market_df['Returns'] = np.log(market_df['Market_Index']).diff()
    market_df['Realized_Vol'] = market_df['Returns'].rolling(window=21).std() * np.sqrt(252)
    
    window_days = 252 * 5
    market_df['Vol_High_Threshold'] = market_df['Realized_Vol'].rolling(window=window_days).quantile(0.80)
    
    def assign_vol_regime(row):
        if pd.isna(row['Realized_Vol']) or pd.isna(row['Vol_High_Threshold']):
            return np.nan
        return 1 if row['Realized_Vol'] > row['Vol_High_Threshold'] else 0

    # define volatility-based regime
    market_df['Current_Regime'] = market_df.apply(assign_vol_regime, axis=1)
    market_df['Target_Regime_5d'] = market_df['Current_Regime'].shift(-5)

    # define bull/bear regime
    market_df['SMA_200'] = market_df['Market_Index'].rolling(window=200).mean()
    market_df['Bull_Bear_Regime'] = np.where(market_df['Market_Index'] > market_df['SMA_200'], 'Bull', 'Bear')

    return market_df.dropna(subset=['Target_Regime_5d'])

def split_data(df, feature_cols, test_date='2020-01-01'):

    dev_mask = df['Date'] < test_date
    test_mask = df['Date'] >= test_date

    X_dev = df.loc[dev_mask, feature_cols + ['Ticker', 'Date']]
    y_dev = df.loc[dev_mask, 'Target_Regime_5d']
    
    X_test = df.loc[test_mask, feature_cols + ['Ticker', 'Date']]
    y_test = df.loc[test_mask, 'Target_Regime_5d']

    return (X_dev, y_dev), (X_test, y_test)

def aggregate_daily(df, index, y_true, y_probs):

    eval_df = pd.DataFrame({
        'Date': df.loc[index, 'Date'].values, 
        'Target': np.array(y_true),
        'Prob': np.array(y_probs)
    })
    
    # group by date using mean probability
    daily = eval_df.groupby('Date').agg({
        'Target': 'first', 
        'Prob': 'mean'
    }).sort_index()
    
    return daily['Target'].values, daily['Prob'].values

def generate_behavioral_features(sequences, feature_names):

    N, T, D = sequences.shape
    t = np.arange(T)
    mean_t, var_t = np.mean(t), np.var(t)
    half_idx = T // 2
    t1, t2 = np.arange(half_idx), np.arange(T - half_idx)
    mean_t1, var_t1 = np.mean(t1), np.var(t1)
    mean_t2, var_t2 = np.mean(t2), np.var(t2)

    feature_list = []
    new_col_names = []

    for i in range(D):
        data = sequences[:, :, i]
        name = feature_names[i]
        
        mean_x = np.mean(data, axis=1)
        mean_tx = np.mean(t * data, axis=1)
        slope = (mean_tx - mean_t * mean_x) / (var_t + 1e-9)
        
        data_1, data_2 = data[:, :half_idx], data[:, half_idx:]
        slope_1 = (np.mean(t1 * data_1, axis=1) - np.mean(t1) * np.mean(data_1, axis=1)) / (var_t1 + 1e-9)
        slope_2 = (np.mean(t2 * data_2, axis=1) - np.mean(t2) * np.mean(data_2, axis=1)) / (var_t2 + 1e-9)
        curvature = slope_2 - slope_1
        
        drawdown = np.max(data, axis=1) - data[:, -1]
        vol_accel = np.std(data_2, axis=1) - np.std(data_1, axis=1)
        z_score = (data[:, -1] - mean_x) / (np.std(data, axis=1) + 1e-9)
        
        diffs = data - mean_x.reshape(-1, 1)
        crossings = np.sum((diffs[:, :-1] > 0) ^ (diffs[:, 1:] > 0), axis=1)
        
        last = data[:, -1]

        feature_list.append(np.stack([slope, curvature, drawdown, vol_accel, z_score, crossings, last], axis=1))
        new_col_names.extend([f"{name}_Slope", f"{name}_Curve", f"{name}_MaxDD", 
                              f"{name}_VolAccel", f"{name}_ZScore", f"{name}_Chop", f"{name}_Last"])

    X_beh = np.hstack(feature_list)

    return X_beh, new_col_names

def create_sequences(df, feature_cols, seq_length):
    sequences, targets, indices = [], [], []
    for ticker, group in df.groupby('Ticker'):
        data_vals = group[feature_cols].values
        target_vals = group['Target_Regime_5d'].values
        idx_vals = group.index
        
        if len(group) >= seq_length:
            for i in range(len(group) - seq_length):
                sequences.append(data_vals[i : i+seq_length])
                targets.append(target_vals[i + seq_length - 1])
                indices.append(idx_vals[i + seq_length - 1]) 
                
    return np.array(sequences), np.array(targets), np.array(indices)

def prepare_distillation_data(X_df, y_series, feature_cols, seq_len):

    df_combined = X_df.copy()
    df_combined['Target_Regime_5d'] = y_series
    
    # for teacher, (N, seq_len, feature_dim)
    seqs, targets, indices = create_sequences(df_combined, feature_cols, seq_len)

    X_behavioral, new_feat_names = generate_behavioral_features(
        seqs, 
        feature_cols, 
    )
    
    # for student
    X_behavioral, new_feat_names = generate_behavioral_features(seqs, feature_cols)
    
    return {
        'seq': seqs,
        'beh': X_behavioral,
        'target': targets,
        'indices': indices,
        'feat_names': new_feat_names
    }
