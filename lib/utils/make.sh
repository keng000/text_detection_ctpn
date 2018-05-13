error_massage=`cat << EOF
[Usage]\$ sh make.sh {cpu / gpu}\n
\tThe device should be specified.\n
\tInput: ${1}
EOF`

if test $# != 1; then
    echo $error_massage
    exit 1

else
    cython bbox.pyx
    cython cython_nms.pyx
    cython gpu_nms.pyx

    if test $1 = "cpu"; then
        python setup_for_cpu.py build_ext --inplace

    elif test $1 = "gpu"; then
        python setup_for_gpu.py build_ext --inplace

    else
        echo $error_massage
        exit 1
    fi
fi

mv utils/* ./
rm -rf build
rm -rf utils

