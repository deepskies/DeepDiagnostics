from typing import TypeVar

from src.models.sbi_model import SBIModel
from src.models.model import Model 

model = TypeVar("model", Model)
ModelModules = {
    "SBIModel": SBIModel
}