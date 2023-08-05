
#ifndef HE_PARAMETER_INTERFACE
#define HE_PARAMETER_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/IR/OpBase.td"


def PolyTypeInterface : TypeInterface<"PolyTypeInterface"> {

  let description = [{
    A type interface for CKKS API level optimization.
      Type that implement this interface can be analyzed and managed by CKKS API-level optimization.
  }];
  let cppNamespace = "::hecate::ckks";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Get a number of polynomials in a given type.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getNumPoly"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of a given type.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getLevel"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of a given type.
      }],
    /*retType=*/"::hecate::ckks::PolyTypeInterface",
    /*methodName=*/"switchLevel",
    /*args=*/(ins "unsigned":$level)
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of a given type.
      }],
    /*retType=*/"::hecate::ckks::PolyTypeInterface",
    /*methodName=*/"switchNumPoly",
    /*args=*/(ins "unsigned":$num_poly)
      >,
    ];
}

def HEVMOpInterface : OpInterface<"HEVMOpInterface"> {
  let description = [{
    A printer interface for HEVM API .
      Operations that implement this interface can be printed to HEVM.
  }];
  let cppNamespace = "::hecate::ckks";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Get a number of polynomials in a given type.
      }],
    /*retType=*/"HEVMOperation",
    /*methodName=*/"getHEVMOperation",
    /*args=*/(ins "llvm::DenseMap<mlir::Value, int64_t>":$plainMap, "llvm::DenseMap<mlir::Value, int64_t>":$cipherMap)
      >
  ];
}



#endif
