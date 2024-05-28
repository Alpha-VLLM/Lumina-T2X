import os
import re

import matplotlib.pyplot as plt


def extract_loss_from_log(log_file):
    with open(log_file, "r") as f:
        log_text = f.read()
        pattern = r"\(step=(\d+)\) Train Loss: ([\d.]+)"
        matches = re.findall(pattern, log_text)
        steps = []
        losses = []
        for match in matches:
            step, train_loss = match
            steps.append(int(step))
            # losses.append(min(float(train_loss), 0.9))
            losses.append(float(train_loss))
        return steps, losses


def smooth_loss(losses, alpha):
    smoothed = [losses[0]]
    for i in range(1, len(losses)):
        smoothed.append((1 - alpha) * losses[i] + alpha * smoothed[i - 1])
    return smoothed


def plot_losses(log_folder):
    plt.figure(figsize=(10, 6))

    # 设置全局字体大小和粗细
    plt.rcParams["font.size"] = 12  # 字体大小
    plt.rcParams["font.weight"] = "bold"  # 字体粗细

    for log_file in os.listdir(log_folder):
        if log_file.endswith(".txt"):  # 假设日志文件都以'.txt'结尾
            steps, losses = extract_loss_from_log(os.path.join(log_folder, log_file))
            losses = smooth_loss(losses, 0.8)
            steps = [i / 1000 for i in steps]  # 每1000步绘制一个点
            plt.plot(steps[80:], losses[80:], label=log_file.replace(".txt", "").replace("_", " "))

            # 设置x轴和y轴的标签字体样式
    plt.xlabel("Training Iterations (k)", fontweight="bold", fontsize=14)
    plt.ylabel("Loss", fontweight="bold", fontsize=14)

    # 设置标题字体样式
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(log_folder, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    log_folder = "/mnt/petrelfs/share_data/liuwenze/results/Large-SiT-rope2d/3B_optimal_lr"
    plot_losses(log_folder)
