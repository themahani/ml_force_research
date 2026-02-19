import matplotlib as mpl


def set_thesis_style():
    mpl.rcParams.update(
        {
            "axes.formatter.use_mathtext": True,
            "mathtext.fontset": "cm",
            "font.family": "Nimbus Roman",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 16,
            "figure.labelsize": 18,
            "font.style": "normal",
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
        }
    )
