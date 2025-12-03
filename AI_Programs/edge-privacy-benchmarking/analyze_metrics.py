import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_all_metrics(output_dir):
    """
    loads all metrics_*.csv files from output_dir into a single dataframe.
    """
    pattern = os.path.join(output_dir, "metrics_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"no metrics_*.csv files found in {output_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        # ensure filter column exists; if not, infer from filename
        if "filter" not in df.columns:
            # example: metrics_gaussian.csv -> gaussian
            base = os.path.basename(f)
            name = base.replace("metrics_", "").replace(".csv", "")
            df["filter"] = name

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # make sure numeric columns are numeric
    for col in ["frame_index", "fps", "cpu_percent", "memory_percent", "gpu_percent"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    return all_df


def plot_fps_over_time(df, output_dir):
    """
    line plot: fps vs frame_index for each filter.
    """
    plt.figure(figsize=(10, 6))

    for flt, group in df.groupby("filter"):
        group = group.sort_values("frame_index")
        plt.plot(group["frame_index"], group["fps"], label=flt)

    plt.xlabel("frame index")
    plt.ylabel("fps")
    plt.title("fps over time by filter")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    out_path = os.path.join(output_dir, "fps_over_time.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")


def plot_mean_fps(df, output_dir):
    """
    bar chart: average fps per filter.
    """
    mean_fps = df.groupby("filter")["fps"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    mean_fps.plot(kind="bar")

    plt.ylabel("average fps")
    plt.title("average fps by filter")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(output_dir, "mean_fps_by_filter.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")


def plot_resource_usage(df, output_dir):
    """
    line plots of cpu and gpu usage vs frame_index, averaged per filter.
    """
    # for smoother lines, group by filter + frame_index and take mean
    grouped = df.groupby(["filter", "frame_index"]).mean(numeric_only=True).reset_index()

    # cpu usage
    plt.figure(figsize=(10, 6))
    for flt, group in grouped.groupby("filter"):
        if "cpu_percent" not in group.columns:
            continue
        group = group.sort_values("frame_index")
        plt.plot(group["frame_index"], group["cpu_percent"], label=flt)

    plt.xlabel("frame index")
    plt.ylabel("cpu percent")
    plt.title("cpu usage over time by filter")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    out_path = os.path.join(output_dir, "cpu_usage_over_time.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")

    # gpu usage (if available)
    if "gpu_percent" in grouped.columns:
        non_null = grouped.dropna(subset=["gpu_percent"])
        if not non_null.empty:
            plt.figure(figsize=(10, 6))
            for flt, group in non_null.groupby("filter"):
                group = group.sort_values("frame_index")
                plt.plot(group["frame_index"], group["gpu_percent"], label=flt)

            plt.xlabel("frame index")
            plt.ylabel("gpu percent")
            plt.title("gpu usage over time by filter")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            out_path = os.path.join(output_dir, "gpu_usage_over_time.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"saved {out_path}")
        else:
            print("no non-null gpu_percent values found; skipping gpu plot.")
    else:
        print("gpu_percent column not found; skipping gpu plot.")


def parse_args():
    parser = argparse.ArgumentParser(description="analyze privacy filter metrics csv files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="directory containing metrics_*.csv files (default: ./output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir

    print(f"loading metrics from: {output_dir}")
    df = load_all_metrics(output_dir)

    print("data summary:")
    print(df.groupby("filter")["fps"].describe())

    plot_fps_over_time(df, output_dir)
    plot_mean_fps(df, output_dir)
    plot_resource_usage(df, output_dir)

    print("done.")


if __name__ == "__main__":
    main()
