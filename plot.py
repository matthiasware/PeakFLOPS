import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


figsize = (10, 10)
linewidth = 3.
legend_fontsize = 24
xyticks_fontsize = 20
bbox_to_anchor = (1, 0.2)
axis_label_fontsize = 24


gflops = [13.76, 26.88, 41.28, 55.04,
          65.6, 78.72, 91.84, 107.52,
          106.56, 105.6, 109.12, 106.52]


if __name__ == "__main__":
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x = list(range(1, len(gflops) + 1, 1))
    ax.plot(x, gflops, linewidth=linewidth)
    ax.plot(x, gflops, "ro", label="benchmarks")
    ax.axhline(112, color="green", label="Theoretical Peak GFLOPS",
               linewidth=linewidth)
    ax.tick_params(axis="both", which="both", labelsize=xyticks_fontsize)
    ax.set_xlabel("# independent instructions")
    ax.set_ylabel("GFLOPS")
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    fig.legend(frameon=False,
               loc='lower right',
               fontsize=legend_fontsize,
               bbox_to_anchor=bbox_to_anchor
               )
    ax.set_title("KabyLake - Experimental Peak FLOPS",
                 fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig("peak_flops.png")
    # plt.show()
