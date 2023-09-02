import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss(
    losses,
    warmup_epochs = 0, 
    xscale        = 'linear', 
    yscale        = 'log',
    output_path   = None,
    output_ext    = 'svg',
    style         = 'darkgrid',
    label_replace = ['-', '_']
):
    plot = False

    for name in losses.keys():
        y_values = losses[name][warmup_epochs:] 
        if len(y_values) < 2:
            continue
        sns.set_style(style)
        plot = True

        epochs = list(range(1, len(y_values)+1))
        
        for rep in label_replace:
            name = name.replace(rep, ' ')
        name = name.capitalize()

        line = sns.lineplot(x=epochs, y=y_values, label=name)
        line.set(yticks=y_values, xticks=epochs)
        line.set(xscale=xscale,   yscale=yscale)

    if plot:
        plt.xlabel("Epocs")
        plt.title("Metrics")
        plt.tight_layout()
        if output_path:
            plt.savefig(f'{output_path}.{output_ext}', format=output_ext)
        plt.show(block=False)
