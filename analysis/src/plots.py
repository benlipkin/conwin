import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONTEXT = [0, 8, 16, 32, 64, 128, 256, 512]


def load_data(path):
    data = pd.read_csv(path)
    data["scores"] = (
        data["scores"]
        .str.replace("nan", "-100")
        .apply(json.loads)
        .apply(np.array)
        .apply(lambda x: x[x != -100])
    )
    data["conwin"] = data["model"].apply(lambda x: int(x.split("_")[1]))
    data["layer"] = data["layer"].apply(lambda x: int(x.split(".")[2]))
    data["n"] = data["scores"].apply(len)
    data["med"] = data["scores"].apply(np.median)
    data["mu"] = data["scores"].apply(np.mean)
    data["sd"] = data["scores"].apply(np.std)
    data["se"] = data["sd"] / np.sqrt(data["n"])
    data["ci"] = 1.96 * data["se"]
    return data


def plot(f, ax, x, y, e, c, t, xl, yl, xt, xs="linear", ylim=(0, 0.2), lab="", leg=""):
    ax.errorbar(x, y, yerr=e, label=lab, color=c, capsize=3, capthick=1)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(t)
    ax.set_xscale(xs)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    ax.set_ylim(ylim)
    if leg:
        ax.legend(title=leg, bbox_to_anchor=(1, 1), loc="upper left")


def plot_max_layer_per_conwin(data, benchmark, context):
    f, ax = plt.subplots()
    maxes = data.groupby("conwin").max("mu")
    x = maxes.index
    y = maxes["mu"]
    e = maxes["ci"]
    c = "black"
    t = f"{benchmark} (Eval Context: {context} Tokens)"
    xl = "Training Context Window (Max N Tokens)"
    yl = "Mean Brain Score (± 95% CI)"
    xt = maxes.index
    plot(f, ax, x, y, e, c, t, xl, yl, xt, xs="log")
    f.savefig(f"../plots/{benchmark}_{context}_max_layer.png", bbox_inches="tight")
    plt.close(f)


def plot_all_layer_by_conwin(data, benchmark, context):
    f, ax = plt.subplots()
    for conwin, group in data.groupby("conwin"):
        x = group["layer"]
        y = group["mu"]
        e = group["ci"]
        s = (np.log2(conwin) - 3) / 7
        c = [0, s, 1 - s]
        t = f"{benchmark} (Eval Context: {context} Tokens)"
        xl = "Layer"
        yl = "Mean Brain Score (± 95% CI)"
        xt = range(12)
        lab = conwin
        leg = "Training Context Window (Max N Tokens)"
        plot(f, ax, x, y, e, c, t, xl, yl, xt, lab=lab, leg=leg)
    f.savefig(f"../plots/{benchmark}_{context}_all_layer.png", bbox_inches="tight")
    plt.close(f)


def plot_all_conwin_by_layer(data, benchmark, context):
    f, ax = plt.subplots()
    for i, (layer, group) in enumerate(data.groupby("layer")):
        x = group["conwin"]
        y = group["mu"]
        e = group["ci"]
        s = i / 11
        c = [s, 0, 1 - s]
        t = f"{benchmark} (Eval Context: {context} Tokens)"
        xl = "Training Context Window (Max N Tokens)"
        yl = "Mean Brain Score (± 95% CI)"
        xt = group["conwin"].unique()
        lab = layer
        leg = "Layer"
        plot(f, ax, x, y, e, c, t, xl, yl, xt, xs="log", lab=lab, leg=leg)
    f.savefig(f"../plots/{benchmark}_{context}_all_conwin.png", bbox_inches="tight")
    plt.close(f)


def main(context):
    data = load_data(f"../scores/results_{context}.csv")
    context = context if context else "Max"
    for id, benchmark in data.groupby("benchmark"):
        plot_max_layer_per_conwin(benchmark, id, context)
        plot_all_layer_by_conwin(benchmark, id, context)
        plot_all_conwin_by_layer(benchmark, id, context)


if __name__ == "__main__":
    for context in CONTEXT:
        main(context)
