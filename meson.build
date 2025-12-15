project(
    'Non-Local_Means_Filter_WS23',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

opencv_dep = dependency('opencv4')

subdir('lib')

executable('Non-Local_Means_Filter',
    [
        'src/Non-Local_Means_Filter.cpp',
        xxd.process('src/Non_Local_Means_Filter.cl'),
    ],
    extra_files: [
        'src/Non_Local_Means_Filter.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
        opencv_dep
    ],
)


