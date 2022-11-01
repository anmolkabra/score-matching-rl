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
    for sampling_method in sampling_methods:
        all_x = []
        all_count_plus_ones = []
        for seed in range(5):
            exp_name = f"simulate_{env}__sampl_{sampling_method}__seed{seed}"
            try:
                df = pd.read_csv(f"runs/{exp_name}/planner_action_counts.csv")
            except:
                continue
            x = df["episode"].values
            count_plus_ones = df["count_plus_ones"].values
            all_x.append(x)
            all_count_plus_ones.append(count_plus_ones)

        all_x = np.concatenate(all_x)
        all_count_plus_ones = np.concatenate(all_count_plus_ones)

        marker = next(markers)
        sns.lineplot(
            x=all_x,
            y=all_count_plus_ones,
            label=f"{labels[sampling_method]}",
            ax=ax,
            marker=marker,
            markersize=10,
            alpha=0.5,
        )

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    # ax.legend()
    plt.legend([], [], frameon=False)
    ax.set_ylabel("Count of +1 played", fontsize=fontsize)
    ax.set_xlabel("Episode", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"notebooks/plots/custom_sin__planner_actions.png")
    plt.savefig(f"notebooks/plots/custom_sin__planner_actions.pdf")
