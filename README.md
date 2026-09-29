# Time Cycle Forecaster & Market Harmonics 🔄📈

**Time Cycle Forecaster** is an institutional quantitative Apify Actor that models and forecasts multi-year financial market cycles, spectral harmonics (via Fast Fourier Transform), historical analog trajectories, and statistical time series models (ARIMA / SARIMAX).

---

## 🌟 Key Features

1. **Multi-Year Institutional Cycle Profiles:**
   - **Stock Market Cycle (18-Year Real Estate & Business Cycle):** 15.5Y Bullish phase, 3.5Y Bearish phase.
   - **Commodities Supercycle (16-Year Cycle):** 10.7Y Expansion phase, 5.3Y Contraction phase.
   - **Crypto Market Cycle (4-Year Halving Cycle):** 3Y Expansion phase, 1Y Contraction phase.
   - **Forex Dollar Cycle (8-Year Currency Cycle):** 4Y Bullish phase, 4Y Bearish phase.
   - **All Cycles Composite Overlay:** Weighted superposition of all macro cycles into a single unified momentum wave score ($-1.0$ to $+1.0$).

2. **Historical Cycle Analog Forecasting:**
   - Shifts back 1 full cycle duration ($t - \text{cycle\_len}$) to compute relative percentage return trajectory from that baseline and projects forward with 90% and 110% confidence bands.
   - Includes automatic cycle-modulated linear trend fallback.

3. **Fast Fourier Transform (FFT) Spectral Decomposition:**
   - Detrends log prices to extract empirical dominant harmonic cycles (periods in months and years) and calculates power spectral density (PSD) shares.

4. **Statistical Forecasting & Diagnostics:**
   - Fits **ARIMA(2,1,2)** and seasonal **SARIMA(2,1,2)(1,1,1)[12]** models on monthly log prices with AIC/BIC metrics and forward projection paths.
   - **OLS Regression:** `Price = a + b * Cycle_Wave` ($R^2$ fit percentage and p-values).
   - **Seasonal Decomposition:** Trend, Seasonal, and Residual component variances.

5. **Historical Cycle Phase Ledger & Win Rates:**
   - Comprehensive ledger of all completed, active, and future phases with exact start/end prices, % returns, max peak drawdowns, and regime prediction accuracy (Bullish vs. Bearish win rates).

6. **Turning Point Countdown:**
   - Predicts exact upcoming cycle turning points (Peaks / Tops and Troughs / Bottoms) with countdown in days and months.

---

## 📥 Input Configuration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `symbols` | `array` | `["SPY", "QQQ", "BTC-USD", "GC=F"]` | List of tickers, indices, commodities, currencies, or crypto. |
| `cycleTemplate` | `string` | `"all"` | `"all"`, `"stock_18y"`, `"commodity_16y"`, `"crypto_4y"`, or `"forex_8y"`. |
| `forecastHorizonMonths` | `integer` | `36` | Number of months into the future to project (6 to 120 months). |
| `yearsBack` | `integer` | `30` | Number of historical years of market data to fetch. |
| `enableArimaForecasts` | `boolean` | `true` | Fit ARMA, ARIMA, and seasonal SARIMA statistical models. |
| `enableSpectralFFT` | `boolean` | `true` | Compute Fast Fourier Transform to extract empirical dominant frequencies. |
| `includeHistoricalLedger` | `boolean` | `true` | Include historical Bullish/Bearish phase performance table. |
| `includeTimeSeries` | `boolean` | `true` | Include monthly OHLC and cycle wave historical series. |

---

## 📤 Output Schema

Each record in the dataset provides:
```json
{
  "symbol": "SPY",
  "cycleProfile": "all",
  "currentPrice": 585.20,
  "primaryCycle": "Stock Market Cycle (18Y)",
  "currentPhase": "Bullish 🟢",
  "phaseStartDate": "2024-03-01",
  "phaseEndDate": "2039-09-01",
  "daysRemaining": 4721,
  "monthsRemaining": 157,
  "compositeWaveScore": 0.452,
  "historicalBullishAccuracyPct": 88.5,
  "historicalBearishAccuracyPct": 75.0,
  "olsR2Pct": 68.4,
  "nextTurningPointDate": "2039-09-01",
  "nextTurningPointType": "Peak / Top",
  "projectedReturn1YPct": 12.8,
  "projectedReturn3YPct": 38.5,
  "fftDominantCyclesMonths": [216.0, 96.0, 48.0],
  "fftHarmonics": [
    {
      "periodMonths": 216.0,
      "periodYears": 18.0,
      "powerSharePct": 52.4
    }
  ],
  "statisticalModels": {
    "arima": { ... },
    "sarima": { ... }
  },
  "analogForecastSeries": [
    { "date": "2026-10-31", "projectedPrice": 595.4, "lowerCi": 535.86, "upperCi": 654.94 }
  ],
  "historicalLedger": [ ... ],
  "executiveSummary": "SPY is currently in a 🟢 BULLISH regime of the Stock Market Cycle (18Y)..."
}
```

---

## 🚀 Running Locally

```bash
uv run --with apify --with pandas --with numpy --with scipy --with statsmodels --with yfinance --with requests --with pytz python -m src.main
```
