# pyiron_vasp
Parser for the Vienna Ab initio Simulation Package (VASP)

## Installation 
Via pip
```
pip install pyiron_vasp
```

Via conda
```
conda install -c conda-forge pyiron_vasp
```

## Usage
Parse an directory with VASP output files 
```python
from pyiron_vasp.vasp.output import parse_vasp_output

output_dict = parse_vasp_output(working_directory="path/to/calculation")
```
