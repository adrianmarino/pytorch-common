from IPython.display import clear_output

from pytorch_common.callbacks.output import OutputCallback
from pytorch_common.callbacks.output.plot.metric_logger import MetricLogger
from pytorch_common.plot import plot_loss
from pytorch_common.utils import filter_by_keys


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
            output_ext         = 'svg'
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
        
    def on_after_train(self, ctx):
        super().on_after_train(ctx)
        if ctx.epoch % self.reg_each_n_epochs == 0:
            [self.logger.append(metric, ctx[metric]) for metric in self.metrics]

    def on_show(self, ctx):
        if not self.logger.is_empty():
            if self.output_path == None:
                clear_output(wait=True)

            plot_loss(
                losses        = filter_by_keys(self.logger.logs, keys = list(self.logger.logs.keys())[:-1]), 
                xscale        = self.xscale, 
                yscale        = self.yscale,
                output_path   = self.output_path,
                output_ext    = self.output_ext,                 
                warmup_epochs = self.warmup_count
            )
