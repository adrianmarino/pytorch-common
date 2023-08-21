from os import listdir
from os.path import isfile, join


class WeightsFileResolver:
    def __init__(
        self, 
        path='./weights', 
        file_part_separator='--', 
        metric_part_separator = '_'
    ):
        self._path = path
        self._file_part_separator   = file_part_separator
        self._metric_part_separator = metric_part_separator

    def _path_files(self): return [f for f in listdir(self._path) if isfile(join(self._path, f))]

    def __call__(self, experiment, metric='val_loss', min_value=True):
        files_by_metric_value = self._get_files_by_metric_value(experiment, metric)
        
        values = list(files_by_metric_value.keys())
        if len(values) == 0:
            return None
        
        values.sort(reverse=not min_value)
        
        filename = files_by_metric_value[values[0]]

        return f'{self._path}/{filename}' if self._path else filename
        
        
    def _get_files_by_metric_value(self, experiment, metric):
        files_by_metric_value = {}
        
        for file in self._path_files():
            if experiment not in file or metric not in file:
                continue

            file_parts = file.split(self._file_part_separator)

            metric_part = [part for part in file_parts if metric in part][0]
            metric_value = metric_part.split(self._metric_part_separator)[-1]
            
            files_by_metric_value[metric_value] = file
        return files_by_metric_value