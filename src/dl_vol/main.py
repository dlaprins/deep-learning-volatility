from pathlib import Path

from loguru import logger

from dl_vol.eval.dummy import backtest_dummy
from dl_vol.eval.metrics import evaluate
from dl_vol.har.har_model import backtest_har, train_har
from dl_vol.preprocessing.preprocess import build_har_panel, build_panel
from dl_vol.tcn.train import train

CSV_PATH = Path(__file__).parents[2] / "data" / "oxfordmanrealizedvolatilityindices.csv"


def main():
    panel = build_panel(csv_path=CSV_PATH, batch_size=128)
    logger.info(
        f"panel: F={panel.num_features} L={panel.seq_len} H={panel.num_horizons} "
        f"head_weights={tuple(round(w, 4) for w in panel.head_weights)}"
    )

    har_data = build_har_panel(csv_path=CSV_PATH)
    har_models = train_har(har_data)

    model = train(
        train_loader=panel.train_loader,
        val_loader=panel.val_loader,
        num_features=panel.num_features,
        num_horizons=panel.num_horizons,
        head_weights=panel.head_weights,
        num_epochs=20,
    )

    device = next(model.parameters()).device.type
    tcn_qlike = evaluate(model, panel.test_loader, device=device, num_horizons=panel.num_horizons)
    dummy_qlike = backtest_dummy(panel.test_dummy_preds, panel.test_loader)
    har_qlike = backtest_har(har_models, har_data)

    horizons = panel.target_names
    logger.info("--- test QLIKE (lower is better) ---")
    logger.info(f"{'horizon':<8}  {'dummy':>10}  {'har':>10}  {'tcn':>10}  {'tcn-har':>10}")
    for h, dq, hq, tq in zip(horizons, dummy_qlike, har_qlike, tcn_qlike):
        logger.info(f"{h:<8}  {dq:>10.6f}  {hq:>10.6f}  {tq:>10.6f}  {tq - hq:>+10.6f}")
    return model


if __name__ == "__main__":
    main()
