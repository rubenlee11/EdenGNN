project('edengnn', 'c', 'fortran', version: '0.1.1')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

incdir_numpy = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
incdir_f2py = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()

fortranobject_c = incdir_f2py / 'fortranobject.c'

# ==========================================
# compile io_chgcar
# ==========================================

f2py_source_chgcar = custom_target('io_chgcar_f2py',
  input: 'src/edengnn/data/io/io_chgcar.f90',
  output: ['io_chgcarmodule.c', 'io_chgcar-f2pywrappers2.f90'], 
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'io_chgcar', '--lower']
)

py.extension_module('io_chgcar',
  [
    'src/edengnn/data/io/io_chgcar.f90', 
    f2py_source_chgcar, 
    fortranobject_c
  ],
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  dependencies: py.dependency(),
  install: true,
  subdir: 'edengnn/data/io'
)

# ==========================================
# compile io_cube
# ==========================================
f2py_source_cube = custom_target('io_cube_f2py',
  input: 'src/edengnn/data/io/io_cube.f90',
  output: ['io_cubemodule.c', 'io_cube-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'io_cube', '--lower']
)

py.extension_module('io_cube',
  [
    'src/edengnn/data/io/io_cube.f90',
    f2py_source_cube,
    fortranobject_c
  ],
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  dependencies: py.dependency(),
  install: true,
  subdir: 'edengnn/data/io'
)

# ==========================================
# compile utils
# ==========================================
f2py_source_utils = custom_target('utils_f2py',
  input: 'src/edengnn/data/io/utils.f90',
  output: ['utils_fmodule.c', 'utils_f-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'utils_f', '--lower']
)

py.extension_module('utils_f',
  [
    'src/edengnn/data/io/utils.f90',
    f2py_source_utils,
    fortranobject_c
  ],
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  dependencies: py.dependency(),
  install: true,
  subdir: 'edengnn/data/io'
)

install_subdir('src/edengnn', install_dir: py.get_install_dir())
