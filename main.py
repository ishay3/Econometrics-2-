import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================

def load_and_prepare_data(consumption_data, income_data):
    dates = pd.date_range(start='1999-Q1', periods=len(consumption_data), freq='Q')
    
    df = pd.DataFrame({
        'consumption': consumption_data,
        'income': income_data
    }, index=dates)
    
    df['log_consumption'] = np.log(df['consumption'])
    df['log_income'] = np.log(df['income'])
    
    return df

# ============================================================================

def adf_test(series, variable_name):
    result = adfuller(series, autolag='AIC')
    
    print(f"\n{'='*60}")
    print(f"ADF Test Results for {variable_name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    return result

def kpss_test(series, variable_name):
    result = kpss(series, regression='c', nlags='auto')
    
    print(f"\n{'='*60}")
    print(f"KPSS Test Results for {variable_name}")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    return result

def test_stationarity(df):
    df['d_log_consumption'] = df['log_consumption'].diff()
    df['d_log_income'] = df['log_income'].diff()
    
    adf_test(df['log_consumption'], 'Log Consumption (Level)')
    kpss_test(df['log_consumption'], 'Log Consumption (Level)')
    
    adf_test(df['log_income'], 'Log Income (Level)')
    kpss_test(df['log_income'], 'Log Income (Level)')
    
    print("\nTesting First Differences:")
    adf_test(df['d_log_consumption'].dropna(), 'Δ Log Consumption')
    adf_test(df['d_log_income'].dropna(), 'Δ Log Income')
    
    return df

# ============================================================================

def select_ardl_order(df, max_lags=4):
    data = df[['log_consumption', 'log_income']].dropna()
    
    sel_res = ardl_select_order(
        data['log_consumption'], 
        max_lags, 
        data[['log_income']], 
        max_lags,
        ic='aic',
        trend='c'
    )
    
    print(f"\nOptimal ARDL Order: {sel_res.model.ardl_order}")
    return sel_res

def estimate_ardl(df, order=None):
    data = df[['log_consumption', 'log_income']].dropna()
    
    if order is None:
        sel_res = select_ardl_order(df)
        order = sel_res.model.ardl_order
    
    ardl_model = ARDL(data['log_consumption'], lags=order[0], exog=data[['log_income']], order=order[1:], trend='c')
    ardl_results = ardl_model.fit()
    
    print("\nARDL Model Summary:")
    print(ardl_results.summary())
    
    return ardl_results, order

# ============================================================================

def bounds_test_manual(ardl_results, case=3):
    """
    Manual implementation of ARDL bounds test for cointegration
    Case 3: Unrestricted constant and no trend
    """
    # Get the ECM coefficient (speed of adjustment)
    try:
        ecm_coef = ardl_results.params['ec']
        ecm_tstat = ardl_results.tvalues['ec']
    except:
        # If 'ec' not available, calculate from levels
        params = ardl_results.params
        ecm_coef = -1
        for param in params.index:
            if 'log_consumption.L' in param:
                ecm_coef += params[param]
        ecm_tstat = ecm_coef / ardl_results.bse.get('log_consumption.L1', 1)
    
    # Critical values for Case III (unrestricted constant, no trend)
    # Source: Pesaran, Shin, Smith (2001), Table CI(iii)
    # k=1 (one exogenous variable)
    critical_values = {
        0.10: {'I0': -2.57, 'I1': -3.21},
        0.05: {'I0': -2.86, 'I1': -3.53},
        0.01: {'I0': -3.43, 'I1': -4.10}
    }
    
    print("\n" + "="*70)
    print("ARDL Bounds Test for Cointegration (Manual Implementation)")
    print("="*70)
    print(f"ECM Coefficient: {ecm_coef:.4f}")
    print(f"t-statistic: {ecm_tstat:.4f}")
    print("\nCritical Values (Case III: Unrestricted constant, no trend, k=1):")
    print("-"*70)
    print(f"{'Significance':>15} {'I(0) Bound':>15} {'I(1) Bound':>15} {'Decision':>20}")
    print("-"*70)
    
    for sig_level, bounds in critical_values.items():
        if ecm_tstat < bounds['I0']:
            decision = "Cointegration"
        elif ecm_tstat > bounds['I1']:
            decision = "No Cointegration"
        else:
            decision = "Inconclusive"
        print(f"{sig_level:>15.0%} {bounds['I0']:>15.2f} {bounds['I1']:>15.2f} {decision:>20}")
    
    print("="*70 + "\n")
    
    return ecm_tstat, critical_values

def estimate_ecm(ardl_results):
    print("\n========================")
    print("Bounds Test for Cointegration")
    bounds_test_manual(ardl_results, case=3)
    print("========================\n")

    # Long-run income elasticity
    params = ardl_results.params
    
    # Get income coefficient at lag 0
    beta_y = params.get('log_income.L0', params.get('log_income', 0))

    # Get all lagged consumption coefficients
    a1 = params.get('log_consumption.L1', 0)
    a2 = params.get('log_consumption.L2', 0)
    a3 = params.get('log_consumption.L3', 0)
    a4 = params.get('log_consumption.L4', 0)

    denom = 1 - (a1 + a2 + a3 + a4)
    
    if abs(denom) > 0.001:  # Check for stability
        long_run_income_elasticity = beta_y / denom
    else:
        print("Warning: Model may be unstable (sum of AR coefficients close to 1)")
        long_run_income_elasticity = np.nan

    print(f"Estimated Long-Run Income Elasticity: {long_run_income_elasticity:.4f}")

    # ECM Model
    try:
        ecm_results = ardl_results.transform_ecm()
        print("\n========================")
        print("ECM Model Summary")
        print(ecm_results.summary())
        print("========================\n")
    except Exception as e:
        print(f"\nNote: Could not transform to ECM format: {e}")
        ecm_results = None

    return ecm_results, long_run_income_elasticity

# ============================================================================

def diagnostic_tests(ardl_results):
    residuals = ardl_results.resid
    
    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS")
    print("="*60)
    
    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"\nJarque-Bera Test (Normality):")
    print(f"  Test Statistic: {jb_stat:.4f}")
    print(f"  P-value: {jb_pval:.4f}")
    if jb_pval > 0.05:
        print("  Result: Residuals appear normally distributed (fail to reject H0)")
    else:
        print("  Result: Residuals are NOT normally distributed (reject H0)")
    
    # Ljung-Box test for serial correlation
    try:
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        print(f"\nLjung-Box Test (Serial Correlation):")
        print(f"  P-value at lag 10: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
        if lb_test['lb_pvalue'].iloc[-1] > 0.05:
            print("  Result: No evidence of serial correlation (fail to reject H0)")
        else:
            print("  Result: Evidence of serial correlation (reject H0)")
    except Exception as e:
        print(f"\nLjung-Box test could not be performed: {e}")
    
    print("="*60 + "\n")
    
    return residuals

# ============================================================================

def plot_results(df, ardl_results, residuals):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Log Consumption and Income
    axes[0, 0].plot(df.index, df['log_consumption'], label='Log Consumption', linewidth=2)
    axes[0, 0].plot(df.index, df['log_income'], label='Log Income', linewidth=2)
    axes[0, 0].set_title('Log Consumption and Income Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Log Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: First Differences
    axes[0, 1].plot(df.index, df['d_log_consumption'], label='Δ Log Consumption', linewidth=2)
    axes[0, 1].plot(df.index, df['d_log_income'], label='Δ Log Income', linewidth=2)
    axes[0, 1].set_title('First Differences', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Change')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    axes[1, 0].plot(residuals.index, residuals, linewidth=1.5, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('ARDL Model Residuals', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals Histogram
    axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Residual Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# ============================================================================

def analyze_ireland_pih(consumption_data, income_data):
    df = load_and_prepare_data(consumption_data, income_data)
    df = test_stationarity(df)
    ardl_results, order = estimate_ardl(df)
    ecm_results, lr_income_effect = estimate_ecm(ardl_results)
    residuals = diagnostic_tests(ardl_results)
    plot_results(df, ardl_results, residuals)
    return df, ardl_results, ecm_results

# ============================================================================

if __name__ == "__main__":
    file_path = "Ireland_Permanent_Income.xlsx" 
    
    data = pd.read_excel(file_path)
    data.columns = ['Time', 'consumption', 'income']
    data['Time'] = pd.PeriodIndex(data['Time'], freq='Q')
    data.set_index('Time', inplace=True)
    
    consumption = data['consumption'].values
    income = data['income'].values
    
    analyze_ireland_pih(consumption, income)