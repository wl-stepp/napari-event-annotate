__version__ = "0.0.1"
from ._sample_data import make_sample_data
from ._widget import Editor_Widget, example_magic_widget

__all__ = (
    "make_sample_data",
    "Editor_Widget",
    "example_magic_widget",
)
