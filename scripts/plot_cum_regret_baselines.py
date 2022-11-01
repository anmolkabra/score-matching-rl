import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

envs = ["custom_sin"]
sampling_methods = ["normal", "custom_sin", "custom_sin__baseline_self"]
labels = {
    "normal": "LDS",
    "custom_sin": "SM with given $\\mathcal{{P}}$",
    "custom_sin__baseline_self": "Ground truth",
}
fontsize = 20
markers = itertools.cycle(("s", "^", "*", "."))
for env in envs:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    results = []
    for sampling_method in sampling_methods:
        all_x = []
        all_avg_regret = []
        all_reward = []
        all_cum_reward = []
        for seed in range(5):
            exp_name = f"simulate_{env}__sampl_{sampling_method}__seed{seed}"
            df = pd.read_csv(f"runs/{exp_name}/results.csv")
            x = df["episode"].values
            avg_regret = df["avg_regret"].values
            reward = df["reward"].values
            cum_reward = np.cumsum(reward)
            all_x.append(x)
            all_avg_regret.append(avg_regret)
            all_reward.append(reward)
            all_cum_reward.append(cum_reward)

        all_x = np.concatenate(all_x)
        all_avg_regret = np.concatenate(all_avg_regret)
        all_reward = np.concatenate(all_reward)
        all_cum_reward = np.concatenate(all_cum_reward)

        marker = next(markers)
        sns.lineplot(
            x=all_x,
            y=all_cum_reward,
            label=labels[sampling_method],
            ax=ax,
            marker=marker,
            markersize=10,
            alpha=0.5,
        )

    ax.legend(fontsize=fontsize)
    ax.set_xlabel("Episode", fontsize=fontsize)
    ax.set_ylabel("Cumulative rewards", fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.savefig("notebooks/plots/custom_sin__cum_rewards.png")
    plt.savefig("notebooks/plots/custom_sin__cum_rewards.pdf")
