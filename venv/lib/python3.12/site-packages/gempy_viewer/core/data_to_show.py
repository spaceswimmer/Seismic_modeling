import dataclasses
from typing import Union


@dataclasses.dataclass
class DataToShow:
    n_axis: int
    show_data: Union[bool, list] = True
    _show_results: Union[bool, list] = True
    show_surfaces: Union[bool, list] = True
    show_lith: Union[bool, list] = True
    show_scalar: Union[bool, list] = False
    show_boundaries: Union[bool, list] = True
    show_topography: Union[bool, list] = False
    show_section_traces: Union[bool, list] = True
    show_values: Union[bool, list] = False
    show_block: Union[bool, list] = False
    
    def __post_init__(self):
        if self.show_results is False:
            self.show_lith = False
            self.show_values = False
            self.show_block = False
            self.show_scalar = False
            self.show_boundaries = False


        if type(self.show_data) is bool:
            self.show_data = [self.show_data] * self.n_axis
        if type(self.show_lith) is bool:
            self.show_lith = [self.show_lith] * self.n_axis
        if type(self.show_values) is bool:
            self.show_values = [self.show_values] * self.n_axis
        if type(self.show_block) is bool:
            self.show_block = [self.show_block] * self.n_axis
        if type(self.show_scalar) is bool:
            self.show_scalar = [self.show_scalar] * self.n_axis
        if type(self.show_boundaries) is bool:
            self.show_boundaries = [self.show_boundaries] * self.n_axis
        if type(self.show_topography) is bool:
            self.show_topography = [self.show_topography] * self.n_axis

    @property
    def show_results(self):
        return self._show_results
    
    @show_results.setter
    def show_results(self, value):
        self._show_results = value
        if value is False:
            self.show_lith = [False]
            self.show_values = [False]
            self.show_block = [False]
            self.show_scalar = [False]
            self.show_boundaries = [False]
            