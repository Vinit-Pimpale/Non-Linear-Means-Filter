xxd_exe = find_program('xxd')
xxd = generator(xxd_exe,
    output : '@PLAINNAME@.xxd.c',
    arguments : [ '-i', '-n', '@PLAINNAME@', '@INPUT@', '@OUTPUT@' ]
)

util_incdir = include_directories('.')

subdir('Core')
# subdir('vx')
subdir('OpenCL')
