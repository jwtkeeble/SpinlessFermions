import pandas as pd

from typing import Optional, Union
NoneType = type(None)

class WriteToFile(object):
  """
  Class to write a dict to a Pandas' dataframe and saved to a .csv
  """

  def __init__(self, load: Optional[Union[str, NoneType]], filename: str) -> None:
    r"""Writer Object
    """
    self.dataframe=None
    if(isinstance(load, str)):
      self.load(load)
    self._filename=filename #why?

  def load(self, filename: str) -> None:
    r"""Method to load an existing .csv file in which to write.

        :param filename: The filename to which the data is saved
        :type filename: str
    """
    self.dataframe = pd.read_csv(filename, index_col=[0])

  def write_to_file(self, filename: str) -> None:
    r"""Method to write current dataframe to file with given filename

        :param filename: The filename to which the data is saved
        :type filename: str
    """
    self.dataframe.to_csv(filename)

  def __call__(self, dic: dict) -> None:
    r"""Method to write to file by concatenating a new `pd.DataFrame` object
        to the existing `pd.DataFrame` object. The current `pd.DataFrame` object
        is stored as a class attribute and continually updated via the `__call__` method.

        :param dic: A Dict object containing the properties being saved (along with their corresponding values)
        :type dic: dict
    """
    if(self.dataframe is None):
      self.dataframe = pd.DataFrame.from_dict(dic)
    else:
      row = pd.DataFrame.from_dict(dic)
      frames = [self.dataframe, row]
      self.dataframe = pd.concat(frames, axis=0, ignore_index=True)