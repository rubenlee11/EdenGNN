project('edengnn', 'c', 'fortran', version: '0.1.0')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

incdir_numpy = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
incdir_f2py = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()

fortranobject_c = incdir_f2py / 'fortranobject.c'

f2py_source = custom_target('io_chgcar_f2py',
  input: 'src/edengnn/data/io_chgcar.f90',
  output: ['io_chgcarmodule.c'], 
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'io_chgcar', '--lower']
)

py.extension_module('io_chgcar',
  [
    'src/edengnn/data/io_chgcar.f90', 
    f2py_source, 
    fortranobject_c
  ],
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  dependencies: py.dependency(),
  install: true,
  subdir: 'edengnn/data'
)

install_subdir('src/edengnn', install_dir: py.get_install_dir())
