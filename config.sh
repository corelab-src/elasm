export HECATE=$( cd -- "$( dirname -- "$0" )" &> /dev/null && pwd )

alias hopt=$HECATE/build/bin/hecate-opt
alias hopt-debug=$HECATE/build-debug/bin/hecate-opt

mkdir -p $HECATE/examples/traced
mkdir -p $HECATE/examples/optimized/eva
mkdir -p $HECATE/examples/optimized/elasm

build-hopt()(
cd $HECATE/build
ninja
)

build-hoptd()(
cd $HECATE/build-debug
ninja
)

hc-trace()(
cd $HECATE/examples
python3 $HECATE/examples/benchmarks/$1.py
)

hc-test()(
cd $HECATE/examples
python3 $HECATE/examples/tests/$3.py $1 $2
)


hopt-print(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-debug-print(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure
}

hopt-debug-print-all(){
hopt-debug --$1 --ckks-config="$HECATE/config.json" --waterline=$2 --enable-debug-printer $HECATE/examples/traced/$3.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading  --mlir-timing --mlir-print-ir-after-failure   --debug
}

hopt-timing-only(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2  $HECATE/examples/traced/$3.mlir --mlir-timing -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hopt-silent(){
hopt --$1 --ckks-config="$HECATE/config.json" --waterline=$2  $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
}

hc-opt-test() {
hopt-silent $1 $2 $3 && hc-test $1 $2 $3
}

hc-opt-test-timing() {
hopt-timing-only $1 $2 $3 && hc-test $1 $2 $3
}


alias hoptd=hopt-debug-print
alias hopta=hopt-debug-print-all
alias hopts=hopt-silent
alias hoptt=hopt-timing-only
alias hoptp=hopt-print
alias hcot=hc-opt-test
alias hcott=hc-opt-test-timing
