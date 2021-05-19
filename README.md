# Isolator case using [MIRGE-Com](https://github.com/illinois-ceesd/mirgecom)

## Installation

### Build [emirge](https://github.com/illinois-ceesd/emirge)
```
git clone https://github.com/illinois-ceesd/emirge
cd emirge
./install.sh
```
### Checkout needed MIRGE-Com branch

```
cd mirgecom
git checkout add-av-to-ns
```

### Get this case file
```
git clone https://github.com/w-hagen/isolator
```

## Building mesh

### Install gmsh
In the emirge directory activate the ceesd environment and install gmsh
```
source ./config/activate_env.sh
conda install gmsh
```

### Run gmsh
In the directory containing this case file generate the mesh
```
gmsh -o isolator.msh -nopopup -format msh2 ./isolator.geo -2
```

## Running Case

The case can the be run similar to other MIRGE-Com applications.
For examples see the MIRGE-Com [documentation](https://mirgecom.readthedocs.io/en/latest/running/systems.html)
