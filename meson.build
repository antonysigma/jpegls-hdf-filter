project('jpegls-hdf-filter', 'cpp',
    version: '0.1',
    license: 'MIT',
    default_options : [
        'buildtype=debugoptimized',
        'b_ndebug=if-release',
        'cpp_std=c++17',
])

charls_proj = subproject('charls',
    default_options: 'default_library=static'
)
charls_lib = charls_proj.get_variable('charls_lib')
charls_inc = charls_proj.get_variable('charls_inc')

hdf5_dep = dependency('hdf5-serial')
threads_dep = dependency('threads')
openmp_dep = dependency('openmp')

h5jpegls_lib = library('h5jpegls',
    sources: [
        'h5jpegls.cc',
    ],
    include_directories: [
        charls_inc,
    ],
    cpp_args: [
        '-DNO_DEBUG',
    ],
    link_with: [
        charls_lib,
    ],
    dependencies: [
        hdf5_dep,
        threads_dep,
        openmp_dep,
    ],
)

h5repack_exe = find_program('h5repack')
test_data = files('test-vector/bloated.hdf5')

compressed_data = custom_target('compressed.hdf5',
    input: test_data,
    output: 'compressed.hdf5',
    env: {
      'HDF5_PLUGIN_PATH': meson.current_build_dir(),
    },
    command: [
        h5repack_exe,
        '-f', 'ones:UD=32012,3,0,0,0',
        '@INPUT@',
        '@OUTPUT@',
    ],
    build_by_default: false,
)

restored_data = custom_target('restored.hdf5',
    input: compressed_data,
    output: 'restored.hdf5',
    env: {
      'HDF5_PLUGIN_PATH': meson.current_build_dir(),
    },
    command: [
        h5repack_exe,
        '-f', 'ones:NONE',
        '@INPUT@',
        '@OUTPUT@',
    ],
    build_by_default: false,
)

h5ls_exe = find_program('h5ls')
test('JPEG-LS compression w/ h5repack',
    h5ls_exe,
    args: [
        '-v',
        compressed_data,
    ],
)

test('JPEG-LS decoding w/ h5repack',
    h5ls_exe,
    args: [
        '-v',
        restored_data,
    ],
)
