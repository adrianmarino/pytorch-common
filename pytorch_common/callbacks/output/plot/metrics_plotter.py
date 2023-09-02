from IPython.display import clear_output

from pytorch_common.callbacks.output import OutputCallback
from pytorch_common.callbacks.output.plot.metric_logger import MetricLogger
from pytorch_common.plot import plot_loss
from pytorch_common.util import filter_by_keys
from datetime import datetime


class MetricsPlotter(OutputCallback):
    def __init__(
            self,
            warmup_count=0,
            plot_each_n_epochs = 2,
            reg_each_n_epochs  = 1,
            metrics            = ['train_loss'],
            xscale             = 'linear', 
            yscale             = 'log',
            output_path        = None,
            output_ext         = 'svg',
            time_pattern       = '%Y-%m-%d_%H-%M-%S',
            disable_plot       = False
    ):
        super().__init__(plot_each_n_epochs)
        self.logger = MetricLogger()
        self.warmup_count = warmup_count
        self.reg_each_n_epochs = reg_each_n_epochs
        self.metrics = metrics + ['epoch']
        self.xscale            = xscale 
        self.yscale            = yscale
        self.output_path       = output_path
        self.output_ext        = output_ext
        self.time_pattern      = time_pattern
        self.disable_plot      = disable_plot
        
    def on_after_train(self, ctx):
        super().on_after_train(ctx)
        if ctx.epoch % self.reg_each_n_epochs == 0:
            [self.logger.append(metric, ctx[metric]) for metric in self.metrics]

    def on_show(self, ctx):
        if not self.logger.is_empty():
            clear_output(wait=True)

            if self.disable_plot:
                plt.ioff()

            plot_loss(
                losses        = filter_by_keys(self.logger.logs, keys = list(self.logger.logs.keys())[:-1]), 
                xscale        = self.xscale, 
                yscale        = self.yscale,
                output_path   = self._build_output_path(ctx),
                output_ext    = self.output_ext,                 
                warmup_epochs = self.warmup_count
            )

    def _build_output_path(self, ctx):
        if self.output_path is None:
            return None

        path_parts = self.output_path.split('/')

        str_now = datetime.now().strftime(self.time_pattern)
        filename = f'{str_now}-{path_parts[-1]}-epoch_{ctx.epoch}'

        if len(path_parts) == 1: return filename

        path = '/'.join(path_parts[:-1])
        return f'{path}/{filename}'
