from pathlib import Path

from dl_vol.preprocessing.preprocess import build_panel
from dl_vol.tcn.train import train
from dl_vol.eval.metrics import evaluate


CSV_PATH = Path(__file__).parents[2] / 'data' / 'oxfordmanrealizedvolatilityindices.csv'


def main():
    panel = build_panel(csv_path=CSV_PATH, batch_size=128)
    print(
        f'panel: F={panel.num_features} L={panel.seq_len} H={panel.num_horizons} '
        f'head_weights={tuple(round(w, 4) for w in panel.head_weights)}'
    )

    model = train(
        train_loader=panel.train_loader,
        val_loader=panel.val_loader,
        num_features=panel.num_features,
        num_horizons=panel.num_horizons,
        head_weights=panel.head_weights,
        num_epochs=20,
    )

    device = next(model.parameters()).device.type
    test_mse, test_qlike = evaluate(
        model, panel.test_loader, device=device, num_horizons=panel.num_horizons
    )
    print(f'test  MSE={test_mse}  QLIKE={test_qlike}')
    return model


if __name__ == '__main__':
    main()