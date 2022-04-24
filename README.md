# pytorch-common

A [Pypi module](https://pypi.org/project/pytorch-common/) with pytorch common tools like:

## Features

* **Callbacks** (keras style)
  * **Validation**: Model validation.
  * **ReduceLROnPlateau**:     
    * Reduce learning rate when a metric has stopped improving. 
    * Models often benefit from reducing the learning rate by a factor
      of 2-10 once learning stagnates. This scheduler reads a metrics
      quantity and if no improvement is seen for a 'patience' number
      of epochs, the learning rate is reduced.
  * **EarlyStop**:
    * Stop training when model has stopped improving a specified metric.
  * **SaveBestModel**: 
    * Save model weights to file while model validation metric improve.
  * **Logger**:
    * Logs context properties. 
    * In general is used to log performance metrics every n epochs.
  * **MetricsPlotter**:
    * Plot evaluation metrics. 
    * This graph is updated every n epochs during training process.
  * **Callback** and **OutputCallback**: 
    * Base classes.
  * **CallbackManager**:
    * Simplify callbacks support to fit custom models.
* **StratifiedKFoldCV**: 
  * Support parallel fold processing on CPU.
* **Mixins**
  * FiMixin
  * CommonMixin
* **Utils**
  * device management
  * stopwatch
  * data split
  * os
  * model
  * LoggerBuilder

## Examples

### Device management


```python
import pytorch_common.util as pu

# Setup prefered device.
pu.set_device_name('gpu') # / 'cpu'

# Setup GPU memory fraction for a process (%).
pu.set_device_memory(
  'gpu' # / 'cpu',
  process_memory_fraction=0.5
)

# Get prefered device. 
# Note: In case the preferred device is not found, it returns CPU as fallback.
device = pu.get_device()
```

### Logging


```python
import logging
import pytorch_common.util as pu

## Default loggin in console...
pu.LoggerBuilder() \
 .on_console() \
 .build()

## Setup format and level...
pu.LoggerBuilder() \
 .level(logging.ERROR) \
 .on_console('%(asctime)s - %(levelname)s - %(message)s') \
 .build()
```
 
 
### Stopwatch


```python
import logging
import pytorch_common.util as pu

sw = pu.Stopwatch()

# Call any demanding process...

# Get resposne time.
resposne_time = sw.elapsed_time()

# Log resposne time.
logging.info(sw.to_str())
```


### Dataset split


```python
import pytorch_common.util as pu

dataset = ... # <-- Torch.utils.data.Dataset

train_subset, test_subset = pu.train_val_split(
  dataset, 
  train_percent = .7
)

train_subset, val_subset, test_subset = pu.train_val_test_split(
  dataset, 
  train_percent = .7, 
  val_percent   = .15
)
```


### Kfolding

```python
import logging
from pytorch_common.kfoldcv import StratifiedKFoldCV, \
                                   ParallelKFoldCVStrategy, \
                                   NonParallelKFoldCVStrategy

# Call your model under this function..
def train_fold_fn(dataset, train_idx, val_idx, params, fold):
  pass

# Get dataset labels
def get_y_values_fn(dataset):
  pass

cv = StratifiedKFoldCV(
  train_fold_fn, 
  get_y_values_fn, 
  strategy=NonParallelKFoldCVStrategy() # or ParallelKFoldCVStrategy()
  k_fold = 5
)

# Model hyperparams...
params = {
    'seed': 42,
    'lr': 0.01,
    'epochs': 50,
    'batch_size': 4000,
    ...
}

# Train model...
result = cv.train(dataset, params)

logging.info('CV results: {}'.format(result))
```


### Assertions


```python
from pytorch_common.error import Assertions, Checker

# Check functions and construtor params usign assertions..

param_value = -1

# Raise an exception with 404103 eror code when the condition is not met 
Assertions.positive_int(404103, param_value, 'param name')

Assertions.positive_float(404103, param_value, 'param name')

# Other options
Assertions.is_class(404205, param_value, 'param name', aClass)
 
Assertions.is_tensor(404401, param_value, 'param name')

Assertions.has_shape(404401, param_value, (3, 4), 'param name')

# Assertions was impelemented using a Checker builder:

 Checker(error_code, value, name) \
    .is_not_none() \
    .is_int() \
    .is_positive() \
    .check()
   
# Other checker options..
#   .is_not_none()
#   .is_int()
#   .is_float()
#   .is_positive()
#   .is_a(aclass)
#   .is_tensor()
#   .has_shape(shape)
```


### Callbacks

```python
from pytorch_common.callbacks import CallbackManager

from pytorch_common.callbacks import EarlyStop, \
                                     ReduceLROnPlateau, \
                                     Validation

from pytorch_common.callbacks.output import Logger, \
                                            MetricsPlotter


def train_method(model, epochs, optimizer, loss_fn, callbacks):
 
 callback_manager = CallbackManager(epochs, optimizer, loss_fn, model, callbacks)

 for epoch in range(epochs):
            callback_manager.on_epoch_start(epoch)

            # train model...

            callback_manager.on_epoch_end(train_loss)

            if callback_manager.break_training():
                break

  return callback_manager.ctx
        

model     = # Create my model...
optimizer = # My optimizer...
loss_fn   = # my lost function

callbacks = [
   # Log context variables after each epoch...
   Logger(['fold', 'time', 'epoch', 'lr', 'train_loss', 'val_loss', ... ]),
   
   EarlyStop(metric='val_auc', mode='max', patience=3),
  
   ReduceLROnPlateau(metric='val_auc'),
  
   Validation(
       val_set,
       metrics = {
           'my_metric_name': lambda y_pred, y_true: # calculate validation metic,
           ...
       },
       each_n_epochs=5
   ),
   
   SaveBestModel(metric='val_loss'),
   
   MetricsPlotter(metrics=['train_loss', 'val_loss'])
]


train_method(model, epochs=100, optimizer, loss_fn, callbacks)
```

Go to next projects to see funcional code examples:

- https://github.com/adrianmarino/deep-fm
- https://github.com/adrianmarino/attention


