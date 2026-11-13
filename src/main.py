import asyncio
import logging
from apify import Actor
from src.data_engine import fetch_historical_prices
from src.cycle_engine import run_cycle_forecaster_analysis

logger = logging.getLogger(__name__)

async def main():
    async with Actor:
        actor_input = await Actor.get_input() or {}
        
        symbols = actor_input.get("symbols", ["SPY", "QQQ", "BTC-USD", "GC=F"])
        cycle_template = actor_input.get("cycleTemplate", "all")
        forecast_horizon = int(actor_input.get("forecastHorizonMonths", 36))
        years_back = int(actor_input.get("yearsBack", 30))
        enable_arima = bool(actor_input.get("enableArimaForecasts", True))
        enable_fft = bool(actor_input.get("enableSpectralFFT", True))
        include_ledger = bool(actor_input.get("includeHistoricalLedger", True))
        include_timeseries = bool(actor_input.get("includeTimeSeries", True))
        
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
            
        Actor.log.info("🔄 Starting Time Cycle Forecaster & Market Harmonics Actor...")
        Actor.log.info(f"Target Assets: {symbols} | Profile: {cycle_template} | Horizon: {forecast_horizon}M | Lookback: {years_back}Y")
        
        results = []
        
        for symbol in symbols:
            try:
                Actor.log.info(f"Analyzing time cycles and spectral harmonics for {symbol}...")
                df_monthly = fetch_historical_prices(symbol, years_back=years_back)
                
                if df_monthly.empty or len(df_monthly) < 12:
                    Actor.log.warning(f"Insufficient historical data for {symbol}. Skipping.")
                    continue
                    
                dossier = run_cycle_forecaster_analysis(
                    symbol=symbol,
                    df_monthly=df_monthly,
                    cycle_template_key=cycle_template,
                    forecast_horizon_months=forecast_horizon,
                    enable_arima=enable_arima,
                    enable_fft=enable_fft,
                    include_ledger=include_ledger,
                    include_timeseries=include_timeseries
                )
                
                await Actor.push_data(dossier)
                results.append(dossier)
                
            except Exception as e:
                Actor.log.error(f"Failed to analyze cycles for {symbol}: {str(e)}")
                
        # Store executive summary in default Key-Value store
        summary_payload = {
            "totalAnalyzed": len(results),
            "symbols": [r["symbol"] for r in results],
            "cycleProfile": cycle_template,
            "forecastHorizonMonths": forecast_horizon,
            "summaries": [
                {
                    "symbol": r["symbol"],
                    "currentPhase": r["currentPhase"],
                    "monthsRemaining": r["monthsRemaining"],
                    "compositeWaveScore": r["compositeWaveScore"],
                    "projectedReturn1YPct": r["projectedReturn1YPct"],
                    "projectedReturn3YPct": r["projectedReturn3YPct"],
                    "nextTurningPoint": r["nextTurningPointDate"],
                    "executiveSummary": r["executiveSummary"]
                }
                for r in results
            ]
        }
        await Actor.set_value("OUTPUT", summary_payload)
        if not results:
            await Actor.fail(status_message="Time Cycle Forecaster produced 0 asset dossiers.")
            return
        
        Actor.log.info(f"✅ Time Cycle Forecaster completed! {len(results)} asset dossiers pushed to Apify Dataset.")

if __name__ == "__main__":
    asyncio.run(main())
