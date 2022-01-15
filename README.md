# pytorch-common

A [Pypi module](https://pypi.org/project/pytorch-common/) with pytorch common tools like:

* **Callbacks** (keras style)
  * **Validation**: Model validation.
  * **ReduceLROnPlateau**
  * **EarlyStop**: Stop training when model has stopped improving a specified metric.
  * **SaveBestModel**: Save model weights to file while model validation metric improve.
  * **Logger**: Logs context properties. In general is used to log performance metrics every n epochs.
  * **JupyterMetricsPlotter**
  * **Callback** and **OutputCallback**: Classes to implement new callbacks.
* **StratifiedKFoldCV**: Parallel an non parallel processing support.
* **Mixins**
  * FiMixin
  * CommonMixin
* **Utils**
  * device management
  * stopwatch
  * data split
  * os

