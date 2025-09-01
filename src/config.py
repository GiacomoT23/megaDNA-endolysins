import os
import yaml

# Server config class
class Config:
  def __init__(self, data):
    self._data = data

  @staticmethod
  def from_file(file_path):
    if not os.path.exists(file_path):
        raise Exception('Unable to find file: {}'.format(file_path))
    
    with open(file_path, 'r') as file:
      data = yaml.safe_load(file)
    return Config(data)

  def get(self, attr, default=None):
    if attr in self._data:
      return self._data[attr]

    return default

  def __getattr__(self, attr):
    if attr in self._data:
      return self._data[attr]
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")