import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_scores(df, title):
    fig, ax1 = plt.subplots(1, 2, figsize=(18,5))

    # --- F1 Macro ---
    ax1[0].bar(df["model"], df["f1_macro"])
    ax1[0].set_title(f"{title} — F1 Macro")

    ax1[0].set_xticks(range(len(df["model"])))
    ax1[0].set_xticklabels(df["model"], rotation=45, ha="right")

    # --- Accuracy ---
    ax1[1].bar(df["model"], df["accuracy"], color="orange")
    ax1[1].set_title(f"{title} — Accuracy")

    ax1[1].set_xticks(range(len(df["model"])))
    ax1[1].set_xticklabels(df["model"], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def show_best_cm(result_df, prefix):
    best = result_df.sort_values("f1_macro", ascending=False).iloc[0]
    model = best["model"]
    cm_img = f"outputs/classification/{prefix}/{prefix}_{model}_AVERAGE_CM.png"

    img = mpimg.imread(cm_img)
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{prefix.upper()} Best Model: {model}")
    plt.show()