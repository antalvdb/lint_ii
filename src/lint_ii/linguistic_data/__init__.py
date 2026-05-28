from enum import Enum


class SuperSemTypes(str, Enum):
    CONCRETE = 'concrete'
    ABSTRACT = 'abstract'
    UNDEFINED = 'undefined'
    UNKNOWN = 'unknown'


WORD_FREQ_COMPOUND_ADJUSTMENT = True
