from abc import ABC

import botorch


class Model(botorch.models.model.Model, ABC):
    pass
