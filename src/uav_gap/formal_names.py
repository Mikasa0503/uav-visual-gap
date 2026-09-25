"""One namespace for versioned formal training, evaluation, reporting and film."""
import re


def protocol_version(value):
    if not isinstance(value,str) or not re.fullmatch(r'v[1-9][0-9]*',value):
        raise ValueError('Protocol version must be v followed by a positive integer')
    return value


def test_prefix(version):
    version=protocol_version(version)
    return 'formal_test' if version=='v1' else 'formal_'+version+'_test'
