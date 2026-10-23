import pandas as pd
import matplotlib.pyplot as plt
import time
while True:
    csv_path = r"E:\Đồ án môn\do_luong_va_dieu_khien\models\HEAD_RB\runs\head_det\results.csv"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(9,4))

    # Box loss
    axes[0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="blue")
    axes[0].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="orange")
    axes[0].set_title("Box Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # mAP50
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="green")
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="red")
    axes[1].set_title("mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"Training progress - epoch hiện tại: {int(df['epoch'].max())}", fontsize=13)
    plt.tight_layout()
    plt.savefig("progress_snapshot_scut.png", dpi=150)
    plt.pause(60)
    plt.close(fig)
    print(f"Epoch {int(df['epoch'].max())} | mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
