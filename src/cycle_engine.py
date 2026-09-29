import datetime
import logging
import math
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)

CYCLE_TEMPLATES = {
    "stock_18y": [
        {
            "name": "Stock Market Cycle (18Y)",
            "start": datetime.date(1948, 9, 1),
            "cycle_len": relativedelta(years=18, months=0, days=0),
            "up_len": relativedelta(years=15, months=6, days=-4),
            "down_len": relativedelta(years=3, months=6, days=0),
            "color_up": "rgba(8, 50, 26, 0.45)",
            "color_down": "rgba(58, 15, 20, 0.45)"
        }
    ],
    "commodity_16y": [
        {
            "name": "Commodities Cycle (16Y)",
            "start": datetime.date(1969, 1, 1),
            "cycle_len": relativedelta(years=16, months=0, days=0),
            "up_len": relativedelta(years=10, months=8, days=0),
            "down_len": relativedelta(years=5, months=4, days=0),
            "color_up": "rgba(20, 40, 60, 0.45)",
            "color_down": "rgba(60, 40, 20, 0.45)"
        }
    ],
    "crypto_4y": [
        {
            "name": "Crypto Market Cycle (4Y)",
            "start": datetime.date(2010, 10, 6),
            "cycle_len": relativedelta(years=4, months=0, days=0),
            "up_len": relativedelta(years=3, months=0, days=0),
            "down_len": relativedelta(years=1, months=0, days=0),
            "color_up": "rgba(40, 20, 60, 0.45)",
            "color_down": "rgba(60, 20, 40, 0.45)"
        }
    ],
    "forex_8y": [
        {
            "name": "Forex Market Cycle (8Y)",
            "start": datetime.date(1977, 1, 1),
            "cycle_len": relativedelta(years=8, months=0, days=0),
            "up_len": relativedelta(years=4, months=0, days=0),
            "down_len": relativedelta(years=4, months=0, days=0),
            "color_up": "rgba(20, 60, 40, 0.45)",
            "color_down": "rgba(60, 20, 20, 0.45)"
        }
    ],
    "all": [
        {
            "name": "Stock Market Cycle (18Y)",
            "start": datetime.date(1948, 9, 1),
            "cycle_len": relativedelta(years=18, months=0, days=0),
            "up_len": relativedelta(years=15, months=6, days=-4),
            "down_len": relativedelta(years=3, months=6, days=0),
            "color_up": "rgba(8, 50, 26, 0.45)",
            "color_down": "rgba(58, 15, 20, 0.45)"
        },
        {
            "name": "Commodities Cycle (16Y)",
            "start": datetime.date(1969, 1, 1),
            "cycle_len": relativedelta(years=16, months=0, days=0),
            "up_len": relativedelta(years=10, months=8, days=0),
            "down_len": relativedelta(years=5, months=4, days=0),
            "color_up": "rgba(20, 40, 60, 0.45)",
            "color_down": "rgba(60, 40, 20, 0.45)"
        },
        {
            "name": "Crypto Market Cycle (4Y)",
            "start": datetime.date(2010, 10, 6),
            "cycle_len": relativedelta(years=4, months=0, days=0),
            "up_len": relativedelta(years=3, months=0, days=0),
            "down_len": relativedelta(years=1, months=0, days=0),
            "color_up": "rgba(40, 20, 60, 0.45)",
            "color_down": "rgba(60, 20, 40, 0.45)"
        },
        {
            "name": "Forex Market Cycle (8Y)",
            "start": datetime.date(1977, 1, 1),
            "cycle_len": relativedelta(years=8, months=0, days=0),
            "up_len": relativedelta(years=4, months=0, days=0),
            "down_len": relativedelta(years=4, months=0, days=0),
            "color_up": "rgba(20, 60, 40, 0.45)",
            "color_down": "rgba(60, 20, 20, 0.45)"
        }
    ]
}

def generate_cycle_phases(start_date, cycle_len, up_len, down_len, start_plot, end_plot):
    """Generate forward and backward cycle phases from anchor date."""
    phases = []
    
    # Forward generation
    current = pd.Timestamp(start_date)
    end_limit = pd.Timestamp(end_plot) + pd.DateOffset(years=20)
    while current < end_limit:
        up_end = current + up_len
        cycle_end = current + cycle_len
        if cycle_end <= current:
            break
        phases.append({'type': 'Bullish', 'start': current, 'end': up_end})
        phases.append({'type': 'Bearish', 'start': up_end, 'end': cycle_end})
        current = cycle_end
        
    # Backward generation
    current = pd.Timestamp(start_date)
    start_limit = pd.Timestamp(start_plot) - pd.DateOffset(years=20)
    while current > start_limit:
        prev_cycle_start = current - cycle_len
        if prev_cycle_start >= current:
            break
        up_end = prev_cycle_start + up_len
        phases.append({'type': 'Bullish', 'start': prev_cycle_start, 'end': up_end})
        phases.append({'type': 'Bearish', 'start': up_end, 'end': current})
        current = prev_cycle_start
        
    # Filter and sort
    phases = [p for p in phases if p['end'] >= pd.Timestamp(start_plot) and p['start'] <= pd.Timestamp(end_plot)]
    phases.sort(key=lambda x: x['start'])
    return phases

def get_cycle_wave_value(t, start, end, phase_type):
    """Compute triangular wave momentum [-1.0, 1.0]."""
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return 0.0
    progress = (t - start).total_seconds() / total_seconds
    progress = max(0.0, min(1.0, progress))
    if phase_type == 'Bullish':
        return -1.0 + 2.0 * progress
    else:
        return 1.0 - 2.0 * progress

def compute_fft_dominant_cycles(prices: pd.Series, top_k: int = 4):
    """
    Perform Fast Fourier Transform (FFT) on detrended log-prices to extract dominant periodic cycles.
    """
    if len(prices) < 24:
        return []
    
    y = np.log(prices.values)
    x = np.arange(len(y))
    
    # Linear Detrend
    p = np.polyfit(x, y, 1)
    detrended = y - (p[0] * x + p[1])
    
    # FFT
    n = len(detrended)
    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per month
    
    # Power Spectral Density (PSD)
    psd = np.abs(fft_vals) ** 2
    
    # Skip DC component (index 0)
    psd[0] = 0
    
    peaks, _ = find_peaks(psd, distance=2)
    if len(peaks) == 0:
        top_indices = np.argsort(psd)[-top_k:][::-1]
    else:
        peak_psd = psd[peaks]
        sorted_peak_idx = np.argsort(peak_psd)[-top_k:][::-1]
        top_indices = peaks[sorted_peak_idx]
        
    dominant_cycles = []
    total_power = np.sum(psd)
    
    for idx in top_indices:
        freq = fft_freqs[idx]
        if freq > 0:
            period_months = round(1.0 / freq, 1)
            period_years = round(period_months / 12.0, 2)
            power_pct = round((psd[idx] / total_power) * 100.0, 2) if total_power > 0 else 0.0
            
            dominant_cycles.append({
                "periodMonths": period_months,
                "periodYears": period_years,
                "frequency": round(float(freq), 4),
                "powerSharePct": power_pct
            })
            
    dominant_cycles.sort(key=lambda x: x["powerSharePct"], reverse=True)
    return dominant_cycles

def fit_arima_sarimax_models(prices: pd.Series, forecast_steps: int = 36):
    """
    Fit ARMA(2,0,2), ARIMA(2,1,2), and SARIMA(2,1,2)(1,1,1)[12] on monthly log-prices.
    """
    log_series = np.log(prices)
    dates_fc = pd.date_range(
        start=prices.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='ME' if hasattr(pd, 'DateOffset') else 'M'
    )
    
    results = {}
    
    def _fit_safe(order, seasonal_order=None):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.tsa.statespace.SARIMAX(
                    log_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                return res
        except Exception:
            return None

    # ARIMA(2,1,2)
    arima_res = _fit_safe((2, 1, 2))
    if arima_res is not None:
        fc = arima_res.get_forecast(steps=forecast_steps)
        mean_exp = np.exp(fc.predicted_mean)
        ci = fc.conf_int(alpha=0.05)
        lower_exp = np.exp(ci.iloc[:, 0])
        upper_exp = np.exp(ci.iloc[:, 1])
        
        results["arima"] = {
            "modelName": "ARIMA(2,1,2)",
            "aic": round(float(arima_res.aic), 2),
            "bic": round(float(arima_res.bic), 2),
            "forecast": [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "predicted": round(float(m), 2),
                    "lowerCi": round(float(l), 2),
                    "upperCi": round(float(u), 2)
                }
                for d, m, l, u in zip(dates_fc, mean_exp, lower_exp, upper_exp)
            ]
        }
        
    # SARIMA(2,1,2)(1,1,1)[12]
    if len(prices) >= 36:
        sarima_res = _fit_safe((2, 1, 2), (1, 1, 1, 12))
        if sarima_res is not None:
            fc = sarima_res.get_forecast(steps=forecast_steps)
            mean_exp = np.exp(fc.predicted_mean)
            ci = fc.conf_int(alpha=0.05)
            lower_exp = np.exp(ci.iloc[:, 0])
            upper_exp = np.exp(ci.iloc[:, 1])
            
            results["sarima"] = {
                "modelName": "SARIMA(2,1,2)(1,1,1)[12]",
                "aic": round(float(sarima_res.aic), 2),
                "bic": round(float(sarima_res.bic), 2),
                "forecast": [
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "predicted": round(float(m), 2),
                        "lowerCi": round(float(l), 2),
                        "upperCi": round(float(u), 2)
                    }
                    for d, m, l, u in zip(dates_fc, mean_exp, lower_exp, upper_exp)
                ]
            }

    return results

def compute_historical_phase_ledger(fit_series: pd.Series, phases: list):
    """
    Build historical performance ledger for all past and active cycle phases.
    """
    last_date = fit_series.index[-1]
    summary_data = []
    
    for p in phases:
        duration_months = int((p['end'] - p['start']).days // 30)
        
        if p['start'] > last_date:
            summary_data.append({
                "phaseType": p['type'],
                "startDate": p['start'].strftime('%Y-%m-%d'),
                "endDate": p['end'].strftime('%Y-%m-%d'),
                "durationMonths": duration_months,
                "startPrice": None,
                "endPrice": None,
                "returnPct": None,
                "maxDrawdownPct": None,
                "accuracy": "Future",
                "status": "Future"
            })
        else:
            status = "Completed" if p['end'] <= last_date else "Active"
            p_start_val = fit_series.asof(p['start'])
            p_end_val = fit_series.asof(p['end']) if p['end'] <= last_date else fit_series.iloc[-1]
            
            if pd.isna(p_start_val) or pd.isna(p_end_val) or p_start_val == 0:
                ret_pct = None
                max_dd_pct = None
                accuracy = "N/A"
            else:
                ret = ((p_end_val - p_start_val) / p_start_val) * 100.0
                ret_pct = round(float(ret), 2)
                
                sub_series = fit_series[(fit_series.index >= p['start']) & (fit_series.index <= p['end'])]
                if not sub_series.empty:
                    cum_max = sub_series.cummax()
                    drawdowns = (sub_series - cum_max) / cum_max * 100.0
                    max_dd_pct = round(float(drawdowns.min()), 2)
                else:
                    max_dd_pct = 0.0
                    
                if p['type'] == 'Bullish':
                    accuracy = "Success" if ret > 0 else "Fail"
                else:
                    accuracy = "Success" if ret < 0 else "Fail"
                    
            summary_data.append({
                "phaseType": p['type'],
                "startDate": p['start'].strftime('%Y-%m-%d'),
                "endDate": p['end'].strftime('%Y-%m-%d'),
                "durationMonths": duration_months,
                "startPrice": round(float(p_start_val), 2) if pd.notna(p_start_val) else None,
                "endPrice": round(float(p_end_val), 2) if pd.notna(p_end_val) else None,
                "returnPct": ret_pct,
                "maxDrawdownPct": max_dd_pct,
                "accuracy": accuracy,
                "status": status
            })
            
    return summary_data

def run_cycle_forecaster_analysis(
    symbol: str,
    df_monthly: pd.DataFrame,
    cycle_template_key: str = "all",
    forecast_horizon_months: int = 36,
    enable_arima: bool = True,
    enable_fft: bool = True,
    include_ledger: bool = True,
    include_timeseries: bool = True
) -> dict:
    """
    Main quantitative analysis orchestrator for Time Cycle Forecaster.
    """
    active_cycles = CYCLE_TEMPLATES.get(cycle_template_key, CYCLE_TEMPLATES["all"])
    fit_series = df_monthly['Close']
    current_price = round(float(fit_series.iloc[-1]), 2)
    last_date = fit_series.index[-1]
    future_end_date = last_date + pd.DateOffset(months=forecast_horizon_months)
    
    # 1. Generate Phases for all active cycles
    all_phases = {}
    for cyc in active_cycles:
        all_phases[cyc['name']] = generate_cycle_phases(
            cyc['start'],
            cyc['cycle_len'],
            cyc['up_len'],
            cyc['down_len'],
            df_monthly.index[0],
            future_end_date
        )
        
    primary_cycle = active_cycles[0]
    primary_phases = all_phases[primary_cycle['name']]
    
    # 2. Identify Current State
    current_time = pd.Timestamp(datetime.date.today())
    current_phase_obj = None
    for p in primary_phases:
        if p['start'] <= current_time <= p['end']:
            current_phase_obj = p
            break
            
    if current_phase_obj is None:
        current_phase_obj = primary_phases[-1]
        
    days_left = max(0, (current_phase_obj['end'] - current_time).days)
    months_left = days_left // 30
    
    # 3. Composite Wave Historical & Future
    full_timeline = pd.date_range(start=df_monthly.index[0], end=future_end_date, freq='D')
    all_waves = {}
    for cyc in active_cycles:
        phases = all_phases[cyc['name']]
        wave_vals = []
        for t in full_timeline:
            val = 0.0
            for p in phases:
                if p['start'] <= t <= p['end']:
                    val = get_cycle_wave_value(t, p['start'], p['end'], p['type'])
                    break
            wave_vals.append(val)
        all_waves[cyc['name']] = np.array(wave_vals)
        
    composite_wave_arr = np.mean(list(all_waves.values()), axis=0) if all_waves else np.zeros(len(full_timeline))
    
    # Current Composite Wave Score
    curr_t_idx = np.argmin(np.abs((full_timeline - current_time).total_seconds()))
    curr_composite_score = round(float(composite_wave_arr[curr_t_idx]), 3) if len(composite_wave_arr) > 0 else 0.0
    
    # 4. OLS Regression on Monthly Cycle Wave
    wave_monthly_vals = []
    for t in df_monthly.index:
        wave_val = 0.0
        count = 0
        for cyc in active_cycles:
            phases = all_phases[cyc['name']]
            for p in phases:
                if p['start'] <= t <= p['end']:
                    wave_val += get_cycle_wave_value(t, p['start'], p['end'], p['type'])
                    count += 1
                    break
        wave_monthly_vals.append(wave_val / count if count > 0 else 0.0)
        
    df_monthly_copy = df_monthly.copy()
    df_monthly_copy['Cycle_Wave'] = wave_monthly_vals
    
    ols_r2_pct = 0.0
    ols_slope = 0.0
    ols_intercept = 0.0
    try:
        X_reg = sm.add_constant(df_monthly_copy['Cycle_Wave'])
        y_reg = df_monthly_copy['Close']
        model_ols = sm.OLS(y_reg, X_reg).fit()
        ols_r2_pct = round(float(model_ols.rsquared * 100.0), 2)
        ols_intercept = round(float(model_ols.params.get('const', 0.0)), 2)
        ols_slope = round(float(model_ols.params.get('Cycle_Wave', 0.0)), 2)
    except Exception as e:
        logger.warning(f"OLS regression failed: {e}")

    # 5. Historical Cycle Analog Forecast
    lookback = primary_cycle['cycle_len']
    is_analog_fallback = True
    analog_forecast_points = []
    
    if lookback.years > 0 or lookback.months > 0:
        past_start = last_date - lookback
        past_start_val = fit_series.asof(past_start)
        
        if pd.notna(past_start_val) and past_start_val > 0:
            is_analog_fallback = False
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_horizon_months,
                freq='ME' if hasattr(pd, 'DateOffset') else 'M'
            )
            last_val = fit_series.iloc[-1]
            
            for t in forecast_dates:
                past_t = t - lookback
                past_val = fit_series.asof(past_t)
                if pd.isna(past_val):
                    past_val = fit_series.asof(fit_series.index[-1] - lookback)
                ret_factor = past_val / past_start_val if past_start_val > 0 else 1.0
                proj_val = last_val * ret_factor
                analog_forecast_points.append({
                    "date": t.strftime("%Y-%m-%d"),
                    "projectedPrice": round(float(proj_val), 2),
                    "lowerCi": round(float(proj_val * 0.90), 2),
                    "upperCi": round(float(proj_val * 1.10), 2)
                })

    if is_analog_fallback:
        # Fallback: Cycle-Modulated Trend
        n = len(fit_series)
        x = np.arange(n)
        y = fit_series.values
        slope, intercept = np.polyfit(x, y, 1)
        
        dates = pd.date_range(
            start=fit_series.index[-1] + pd.DateOffset(months=1),
            periods=forecast_horizon_months,
            freq='ME' if hasattr(pd, 'DateOffset') else 'M'
        )
        
        flat_phases = []
        for k, v in all_phases.items():
            flat_phases.extend(v)
            
        for i, t in enumerate(dates):
            trend = slope * (n + i) + intercept
            wave_sum = 0.0
            count = 0
            for p in flat_phases:
                if p['start'] <= t <= p['end']:
                    wave_sum += get_cycle_wave_value(t, p['start'], p['end'], p['type'])
                    count += 1
            avg_wave = wave_sum / count if count > 0 else 0.0
            proj_val = max(0.1, trend * (1.0 + 0.1 * avg_wave))
            analog_forecast_points.append({
                "date": t.strftime("%Y-%m-%d"),
                "projectedPrice": round(float(proj_val), 2),
                "lowerCi": round(float(proj_val * 0.85), 2),
                "upperCi": round(float(proj_val * 1.15), 2)
            })

    # Projected returns
    proj_1y_ret = 0.0
    proj_3y_ret = 0.0
    if len(analog_forecast_points) >= 12:
        p12 = analog_forecast_points[11]["projectedPrice"]
        proj_1y_ret = round(((p12 - current_price) / current_price) * 100.0, 2)
    if len(analog_forecast_points) >= 36:
        p36 = analog_forecast_points[35]["projectedPrice"]
        proj_3y_ret = round(((p36 - current_price) / current_price) * 100.0, 2)
    elif len(analog_forecast_points) > 0:
        plast = analog_forecast_points[-1]["projectedPrice"]
        proj_3y_ret = round(((plast - current_price) / current_price) * 100.0, 2)

    # 6. Historical Ledger & Win Rates
    ledger_records = compute_historical_phase_ledger(fit_series, primary_phases)
    bullish_phases = [x for x in ledger_records if x['phaseType'] == 'Bullish' and x['returnPct'] is not None and x['status'] == 'Completed']
    bearish_phases = [x for x in ledger_records if x['phaseType'] == 'Bearish' and x['returnPct'] is not None and x['status'] == 'Completed']
    
    b_success = sum(1 for x in bullish_phases if x['accuracy'] == 'Success')
    b_win_rate = round((b_success / len(bullish_phases) * 100.0), 2) if bullish_phases else 0.0
    
    s_success = sum(1 for x in bearish_phases if x['accuracy'] == 'Success')
    s_win_rate = round((s_success / len(bearish_phases) * 100.0), 2) if bearish_phases else 0.0

    # 7. Next Turning Points Detector
    next_turning_point_date = current_phase_obj['end'].strftime('%Y-%m-%d')
    next_turning_type = "Trough / Bottom" if current_phase_obj['type'] == 'Bearish' else "Peak / Top"

    # 8. Spectral FFT Harmonics
    fft_cycles = []
    if enable_fft:
        fft_cycles = compute_fft_dominant_cycles(fit_series)

    # 9. ARIMA / SARIMAX Projections
    arima_models_output = {}
    if enable_arima:
        arima_models_output = fit_arima_sarimax_models(fit_series, forecast_horizon_months)

    # 10. Seasonal Decomposition & Autocorrelations
    seasonal_stats = {}
    try:
        decomp = seasonal_decompose(fit_series, model='additive', period=12)
        seasonal_stats = {
            "trendVariance": round(float(np.nanvar(decomp.trend)), 2),
            "seasonalVariance": round(float(np.nanvar(decomp.seasonal)), 2),
            "residualVariance": round(float(np.nanvar(decomp.resid)), 2)
        }
    except Exception:
        pass

    # Build Executive Summary
    regime_emoji = "🟢" if current_phase_obj['type'] == 'Bullish' else "🔴"
    exec_summary = (
        f"{symbol} is currently in a {regime_emoji} {current_phase_obj['type'].upper()} regime of the "
        f"{primary_cycle['name']} ending on {current_phase_obj['end'].strftime('%Y-%m-%d')} ({months_left} months / {days_left} days remaining). "
        f"Composite momentum score is {curr_composite_score:+.2f}. "
        f"Historical Bullish win rate is {b_win_rate:.1f}%, Bearish win rate is {s_win_rate:.1f}%. "
        f"Analog cycle projection indicates a 1-Year expected return of {proj_1y_ret:+.2f}% and 3-Year expected return of {proj_3y_ret:+.2f}%."
    )

    # Time series output
    monthly_ts = []
    if include_timeseries:
        for idx, row in df_monthly_copy.iterrows():
            monthly_ts.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "cycleWave": round(float(row['Cycle_Wave']), 3)
            })

    output_dossier = {
        "symbol": symbol,
        "cycleProfile": cycle_template_key,
        "currentPrice": current_price,
        "primaryCycle": primary_cycle['name'],
        "currentPhase": f"{current_phase_obj['type']} {regime_emoji}",
        "phaseStartDate": current_phase_obj['start'].strftime('%Y-%m-%d'),
        "phaseEndDate": current_phase_obj['end'].strftime('%Y-%m-%d'),
        "daysRemaining": days_left,
        "monthsRemaining": months_left,
        "compositeWaveScore": curr_composite_score,
        "historicalBullishAccuracyPct": b_win_rate,
        "historicalBearishAccuracyPct": s_win_rate,
        "olsR2Pct": ols_r2_pct,
        "olsRegressionEquation": f"{symbol} = {ols_intercept:.2f} {'+' if ols_slope >= 0 else '-'} {abs(ols_slope):.2f} * Cycle_Wave",
        "nextTurningPointDate": next_turning_point_date,
        "nextTurningPointType": next_turning_type,
        "projectedReturn1YPct": proj_1y_ret,
        "projectedReturn3YPct": proj_3y_ret,
        "analogProjectionType": "Historical Cycle Shift" if not is_analog_fallback else "Cycle-Modulated Trend Fallback",
        "fftDominantCyclesMonths": [c["periodMonths"] for c in fft_cycles],
        "fftHarmonics": fft_cycles,
        "statisticalModels": arima_models_output,
        "seasonalDecomposition": seasonal_stats,
        "historicalLedger": ledger_records if include_ledger else [],
        "analogForecastSeries": analog_forecast_points,
        "monthlyTimeSeries": monthly_ts,
        "executiveSummary": exec_summary,
        "calculatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    return output_dossier
