
#ifndef FORWARD_MANAGEMENT_INTERFACE
#define FORWARD_MANAGEMENT_INTERFACE

include "mlir/IR/OpBase.td"
include "hecate/Dialect/Earth/IR/HEParameterInterface.td"

def ForwardMgmtInterface : OpInterface<"ForwardMgmtInterface"> {
  let description = [{
    An op interface of CKKS parameter optimization.
      Ops that implement this interface can be analyzed and managed by scale management scheme.
  }];
  let cppNamespace = "::hecate::earth";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processOperandsEVA",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return; }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processResultsEVA",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return; }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processOperandsPARS",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.processOperandsEVA(param); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processResultsPARS",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.processResultsEVA(param); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processOperandsSNR",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.processOperandsPARS(param); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"void",
    /*methodName=*/"processResultsSNR",
    /*args=*/(ins "int64_t":$param),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.processResultsPARS(param); }]
      >
      ];
}

#endif
