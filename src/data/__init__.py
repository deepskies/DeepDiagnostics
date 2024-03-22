
from typing import TypeVar

from src.data.data import Data
from src.data.h5_data import H5Data
from src.data.pickle_data import PickleData

data = TypeVar("data", Data)

DataModules = {
    "H5Data": H5Data, 
    "PickleData": PickleData
}