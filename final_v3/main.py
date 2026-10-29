"""Runs signal training, portfolio construction and reporting."""

import argparse
import json
import pathlib
import shutil

import pandas as pd

import checkpoints
import config
import evaluate
import identity
import optimizer
import report
import timing

HEADLINE = [
    "ir",
    "cagr",
    "ann_active",
    "sharpe",
    "max_dd",
    "alpha_ann",
    "alpha_t",
    "beta",
    "beta_se",
    "alpha4_ann",
    "alpha4_t_hac",
    "turnover_avg",
    "oos_r2_universe",
    "compliance_all_pass",
    "submission_ready",
    "annualised_realised_volatility_daily",
]


def check_inputs(*, frozen_local_reporting=False) -> None:
    names = [
        config.PANEL_FILE,
        config.FACTOR_LIST_FILE,
        config.FILINGS_FILE,
        evaluate.DSF_FILE,
    ]
    missing = [n for n in names if not (config.DATA_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing in {config.DATA_DIR}: {', '.join(missing)}. See data/README.md."
        )
    if not config.REPORTING_FILE.is_file():
        raise SystemExit(f'Missing Treasury/S&P 500 reporting file: {config.REPORTING_FILE}')
    ibes_file = getattr(config, 'IBES_FILE', config.FINAL.parent/'external_data/external.parquet')
    if config.USE_IBES and not frozen_local_reporting and not ibes_file.is_file():
        raise SystemExit(f'Missing licensed I/B/E/S input for conf_lm_gated: {ibes_file}. No strategy sleeve has been silently removed.')


def build_predictions() -> pd.DataFrame:
    """Steps 1-3: monthly stock scores with point-in-time labels."""
    import combiner
    from sleeves import base
    context = base.Context(
        chars=base.load_panel(config.DATA_DIR),
        data_dir=config.DATA_DIR,
        work_dir=config.WORK_DIR,
        device=config.DEVICE,
    )
    predictions = combiner.predict(context)
    labelled = identity.label(
        predictions,
        config.DATA_DIR / config.PANEL_FILE,
        config.DATA_DIR / config.FILINGS_FILE,
    )

    timing.validate_predictions(labelled)
    return labelled


def nav_schedule(
    portfolio: pd.DataFrame,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    endowment: float,
) -> pd.Series:
    """Capital at the formation of each holding month.

    The mandate endows US$1bn at the start of the out-of-sample period and the
    fund compounds from there (Instructions 3.1, 3.3): month t is sized on the
    endowment grown by every prior month's net return, so a successful fund
    tightens its own liquidity limits.
    """
    books = {
        month: dict(zip(g.permno.astype(int), g.target_weight, strict=True))
        for month, g in portfolio.groupby("return_month")
    }
    returns = evaluate.evaluate(books, universe, market).total
    return (1 + returns).cumprod().shift(1).fillna(1.0) * float(endowment)


def score_column(predictions: pd.DataFrame) -> str:
    """The score the optimizer ranks on at this fund size."""
    liquid = (
        optimizer.ENABLE_MICRO_SLEEVE_SWITCH
        and config.NAV > optimizer.MICRO_SLEEVES_MAX_NAV
        and "final_score_liquid" in predictions
    )
    return "final_score_liquid" if liquid else "final_score"


def configure_run(argv=None):
    parser = argparse.ArgumentParser(description="FIAM 2026 book, data to report.")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="keep the models and data already in output/ instead of rebuilding",
    )
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument('--no-discovery', action='store_true', help='Disable all LLM research/scouting; run existing sleeves only (default)')
    discovery.add_argument('--with-discovery', action='store_true', help='Enable the agent research/scouting path on top of the same baseline strategy')
    parser.add_argument('--device', choices=('cpu','cuda'), default=config.DEVICE)
    parser.add_argument('--official-only', action='store_true', help='Explicitly exclude the licensed I/B/E/S analyst-revision sleeve')
    parser.add_argument('--data-dir', type=pathlib.Path, help='Competition input directory')
    parser.add_argument('--external-file', type=pathlib.Path, help='Licensed external input container')
    parser.add_argument('--reporting-file', type=pathlib.Path, help='Treasury/S&P 500 reporting parquet')
    arcana_beta = parser.add_mutually_exclusive_group()
    arcana_beta.add_argument('--arcana-beta-file', type=pathlib.Path,
        help='Override the default Arcana-US stock beta to SPY CSV')
    arcana_beta.add_argument('--no-arcana-beta', action='store_true',
        help='Disable the default Arcana beta input and use local beta constraints only')
    parser.add_argument('--max-abs-beta', type=float,
        help='Override the final bound applied to each enabled formation-beta constraint (positive, at most 0.05)')
    parser.add_argument('--replay-with-local-reporting', action='store_true',
        help='Explicit frozen replay: verify competition inputs, retain archived I/B/E/S forecasts without reopening training inputs, and record the supplied local reporting source')
    parser.add_argument('--output-dir', type=pathlib.Path, help='Separate destination for this run')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--save-checkpoint', type=pathlib.Path, help='Archive the completed output, model weights and forecasts; do not run training')
    mode.add_argument('--from-predictions', type=pathlib.Path, help='Rebuild portfolio/report from a verified checkpoint; no model or graph rebuild')
    mode.add_argument('--from-weights', type=pathlib.Path, help='Replay a verified checkpoint portfolio and generate its report; no training or optimization')
    args = parser.parse_args(argv)
    if args.max_abs_beta is not None:
        import math
        if not math.isfinite(args.max_abs_beta) or not 0 < args.max_abs_beta <= .05:
            parser.error('--max-abs-beta must be positive and at most 0.05')
        if args.from_weights or args.save_checkpoint:
            parser.error('--max-abs-beta requires portfolio construction')
        optimizer.MAX_ABS_BETA = args.max_abs_beta
    if args.arcana_beta_file and (args.from_weights or args.save_checkpoint):
        parser.error('--arcana-beta-file requires portfolio construction, not saved-weight replay or checkpoint export')
    if args.arcana_beta_file:
        config.ARCANA_BETA_FILE = args.arcana_beta_file.resolve()
    elif args.no_arcana_beta:
        config.ARCANA_BETA_FILE = None
    elif config.ARCANA_BETA_FILE is not None:
        config.ARCANA_BETA_FILE = pathlib.Path(config.ARCANA_BETA_FILE).resolve()
    if config.ARCANA_BETA_FILE and not config.ARCANA_BETA_FILE.is_file():
        parser.error('Arcana beta input file does not exist')
    if args.replay_with_local_reporting and not (args.from_predictions or args.from_weights):
        parser.error('--replay-with-local-reporting requires --from-predictions or --from-weights')
    if args.no_discovery:
        config.DISCOVERY = False
    elif args.with_discovery:
        config.DISCOVERY = True
    config.DEVICE = args.device
    if args.official_only:
        config.USE_IBES = False
    from disclosed_ibes import FILE as default_ibes_file
    config.IBES_FILE = default_ibes_file
    if args.data_dir:
        config.DATA_DIR = args.data_dir.resolve()
    if args.external_file:
        import disclosed_ibes
        config.IBES_FILE = args.external_file.resolve()
        disclosed_ibes.FILE = config.IBES_FILE
        if config.REPORTING_FILE.name == 'external.parquet':
            config.REPORTING_FILE = config.IBES_FILE
    if args.reporting_file:
        config.REPORTING_FILE = args.reporting_file.resolve()
    replay = args.from_predictions or args.from_weights
    if replay:
        saved = checkpoints.metadata(replay)
        if args.no_discovery and saved['discovery_enabled']:
            raise ValueError('Cannot label agent-enabled archived forecasts as no-discovery')
        if args.with_discovery and not saved['discovery_enabled']:
            raise ValueError('Cannot add discovery proposals to frozen forecasts; rebuild predictions without --from-predictions')
        if args.official_only and saved['ibes_enabled']:
            raise ValueError('Cannot remove I/B/E/S effects from already archived forecasts')
        config.DISCOVERY = saved['discovery_enabled']
        config.USE_IBES = saved['ibes_enabled']
    if args.output_dir:
        config.WORK_DIR = args.output_dir.resolve()
    elif replay:
        config.WORK_DIR = config.FINAL / 'replay_output'
    if replay:
        checkpoints.require_separate_paths(config.WORK_DIR, replay)
    if config.ARCANA_BETA_FILE and config.ARCANA_BETA_FILE.is_relative_to(config.WORK_DIR):
        raise ValueError('Arcana beta input must be outside the output directory')
    work = config.WORK_DIR.resolve()
    protected = [config.FINAL.resolve(), config.DATA_DIR.resolve(),
                 config.IBES_FILE.resolve(), config.REPORTING_FILE.resolve()]
    if any(path.is_relative_to(work) for path in protected):
        raise ValueError('Output directory must not contain application source or inputs')
    if work.is_relative_to(config.DATA_DIR.resolve()):
        raise ValueError('Output directory must be separate from the input directory')
    return args


def main() -> None:
    args = configure_run()
    if args.save_checkpoint:
        saved = checkpoints.export(config.WORK_DIR, args.save_checkpoint,
            config.DATA_DIR, config.IBES_FILE, config.REPORTING_FILE)
        print(json.dumps({'checkpoint':str(args.save_checkpoint.resolve()),
                          'neural_checkpoints':saved['neural_checkpoints'],
                          'holding_months':saved['holding_months']}, indent=2))
        return
    print(f'Run mode: discovery={config.DISCOVERY}, device={config.DEVICE}, I/B/E/S={config.USE_IBES}, Arcana beta={config.ARCANA_BETA_FILE or "disabled"}', flush=True)
    check_inputs(frozen_local_reporting=args.replay_with_local_reporting)
    replay = args.from_predictions or args.from_weights
    loaded = None
    if replay:
        loaded = checkpoints.load(replay, config.DATA_DIR, config.IBES_FILE, config.REPORTING_FILE,
                                  local_reporting=args.replay_with_local_reporting)
    if not args.reuse and config.WORK_DIR.exists():
        if args.output_dir and any(config.WORK_DIR.iterdir()) and not any(
            (config.WORK_DIR/name).is_file() for name in (timing.CACHE_FILE, 'replay_provenance.json')
        ):
            raise ValueError('Refusing to clear a non-run output directory; choose an empty destination')
        shutil.rmtree(config.WORK_DIR)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not replay:
        timing.prepare_cache(config.WORK_DIR, config.DATA_DIR, config.DEVICE)
    panel = config.DATA_DIR / config.PANEL_FILE

    predictions = loaded[1] if loaded else build_predictions()
    predictions.to_parquet(config.WORK_DIR / "predictions.parquet", index=False)
    if replay:
        checkpoints.record_replay(config.WORK_DIR, replay, loaded[0],
            'frozen_weights' if args.from_weights else 'frozen_predictions')
        checkpoints.restore_forecast_assets(config.WORK_DIR, replay, loaded[0])
    if not args.replay_with_local_reporting:
        checkpoints.capture_predictions(config.WORK_DIR, config.DATA_DIR, config.IBES_FILE, config.REPORTING_FILE)

    crowded = None
    print("Short-interest crowding screen: not applied (no official short-interest data)", flush=True)
    universe = evaluate.load_universe(panel)
    market = evaluate.market()

    # Settle each holding month before sizing the next month, so dollar
    # liquidity limits use capital compounded only through prior returns.
    if args.from_weights:
        portfolio = loaded[2].copy()
        portfolio['return_month'] = pd.to_datetime(portfolio.return_month).dt.to_period('M')
        nav = nav_schedule(portfolio, universe, market, config.NAV)
        for name in ('portfolio.parquet','optimizer_diagnostics.csv','optimizer_config.json'):
            shutil.copy2(args.from_weights/name, config.WORK_DIR/name)
    else:
        capital = evaluate.CausalCapital(universe, market, config.NAV)
        portfolio, diagnostics = optimizer.optimize(
            predictions, panel, nav=capital, short_interest=crowded, risk_cache={})
        nav = capital.schedule
        recomputed = nav_schedule(portfolio, universe, market, config.NAV)
        if not (nav/recomputed-1).abs().le(1e-12).all():
            raise ValueError('Causal sizing capital does not reconcile with final portfolio returns')
        optimizer.write(portfolio, diagnostics, config.WORK_DIR, nav=config.NAV)

    stats = report.run(
        portfolio,
        predictions,
        universe,
        market,
        crowded,
        config.WORK_DIR / "report",
        score_column(predictions),
        nav=nav,
        data_dir=config.DATA_DIR,
    )
    headline = {key: stats.get(key) for key in HEADLINE}
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
