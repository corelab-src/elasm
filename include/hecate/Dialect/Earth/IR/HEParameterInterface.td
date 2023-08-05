#ifndef HE_PARAMETER_INTERFACE
#define HE_PARAMETER_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/IR/OpBase.td"

def HEScaleOpInterface : OpInterface<"HEScaleOpInterface"> {
  let description = [{
    An op interface of CKKS parameter optimization.
      Ops that implement this interface can be analyzed and managed by scale management scheme.
  }];
  let cppNamespace = "::hecate::earth";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Check that this is constant op.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isConst",
    /*args=*/(ins ),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op->template hasTrait<mlir::OpTrait::ConstantLike>(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Check that there is only one ciphertext operand.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isSingle"
      >,
    InterfaceMethod<
      /*desc=*/[{Check that the scale consumption is appeared.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isConsume"
      >,
    InterfaceMethod<
      /*desc=*/[{Check that the scale consumption is appeared.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isNoisy",
    /*args=*/(ins ),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return false; }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get an operand scale type of the operation.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"getOperandScaleType",
    /*args=*/(ins "unsigned":$idx)
      >,
    InterfaceMethod<
      /*desc=*/[{Get a result scale type of the operation.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"getScaleType"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of the result.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isOperandCipher",
    /*args=*/(ins "unsigned":$idx),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getOperandScaleType(idx).isCipher(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of the result.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getOperandLevel",
    /*args=*/(ins "unsigned":$idx),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getOperandScaleType(idx).getLevel(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of the result.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getOperandScale",
    /*args=*/(ins "unsigned":$idx),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getOperandScaleType(idx).getScale(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of the result.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isCipher",
    /*args=*/(ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getScaleType().isCipher(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of the result.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getRescaleLevel",
    /*args=*/(ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getScaleType().getLevel(); }]
      >,
    InterfaceMethod<
      /*desc=*/[{Get a scale of the result.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getScale",
    /*args=*/(ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{ return $_op.getScaleType().getScale(); }]
      >
      ];
}


def HEProfInterface : OpInterface<"HEProfInterface"> {
  let description = [{
    Profile data reader interface for profile-based scale optimization.
      This interface depends on the HEScaleOpInterface.
  }];
  let cppNamespace = "::hecate::earth";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Get an estimated latency of the given operation on a given level.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getLatencyOf",
    /*args=*/(ins "unsigned":$level)
      >,
    InterfaceMethod<
      /*desc=*/[{Get an estimated latency of the given operation on a given level.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getNum",
    /*args=*/(ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
      int num = 0;
      for (auto&& v : $_op->getResults()){
        if (auto ranked = v.getType().template dyn_cast<mlir::RankedTensorType>()){
          num += ranked.getNumElements();
        } else {
          num +=1;
        }
      }
      return num;
    }]
    >,
      InterfaceMethod<
        /*desc=*/[{Get a scale of the result.
        }],
      /*retType=*/"unsigned",
      /*methodName=*/"getNoiseScale",
      /*args=*/(ins)
        >,
      InterfaceMethod<
        /*desc=*/[{Check that the scale consumption is appeared.
        }],
      /*retType=*/"unsigned",
      /*methodName=*/"getCipherLevel"
        >,
      InterfaceMethod<
        /*desc=*/[{Get an estimated noise of the given operation on a given level.
        }],
      /*retType=*/"double",
      /*methodName=*/"getNoiseOf",
      /*args=*/(ins "unsigned":$level)
        >,
      InterfaceMethod<
        /*desc=*/[{Get an estimated latency of the given operation.
        }],
      /*retType=*/"unsigned",
      /*methodName=*/"getLatency",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getLatencyOf($_op.getCipherLevel()); }]
        >,
      InterfaceMethod<
        /*desc=*/[{Get an estimated noise of the given operation.
        }],
      /*retType=*/"double",
      /*methodName=*/"getNoise",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getNoiseOf($_op.getCipherLevel()); }]
        >,
      InterfaceMethod<
        /*desc=*/[{Get an estimated first derivative of the latency of the given operation on a given level.
        }],
      /*retType=*/"unsigned",
      /*methodName=*/"getLatencyDiffOf",
      /*args=*/(ins "unsigned":$level),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{
        if (level >0)
          return $_op.getLatencyOf(level)-$_op.getLatencyOf(level-1);
        else
          return $_op.getLatencyOf(level);
      }]
      >,
        InterfaceMethod<
          /*desc=*/[{Get an estimated first derivative of the noise of the given operation on a given level.
          }],
        /*retType=*/"double",
        /*methodName=*/"getNoiseDiffOf",
        /*args=*/(ins "unsigned":$level),
        /*methodBody=*/[{}], /*defaultImplementation=*/[{
          if (level >0)
            return $_op.getNoiseOf(level)-$_op.getNoiseOf(level-1);
          else
            return $_op.getNoiseOf(level);
        }]
      >,
        InterfaceMethod<
          /*desc=*/[{Get an estimated first derivative of the latency of the given operation.
          }],
        /*retType=*/"unsigned",
        /*methodName=*/"getLatencyDiff",
        /*args=*/(ins),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{ return getLatencyDiffOf($_op.getCipherLevel()); }]
          >,
        InterfaceMethod<
          /*desc=*/[{Get an estimated first derivative of the noise of the given operation .
          }],
        /*retType=*/"double",
        /*methodName=*/"getNoiseDiff",
        /*args=*/(ins),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{ return getNoiseDiffOf($_op.getCipherLevel()); }]
          >
          ];
}


def HEAutoDiffInterface : OpInterface<"HEAutoDiffInterface"> {
  let description = [{
    Auto differentiation framework for HE.
      Assuming element-wise differentiation and ciphertext-wise result to reduce the overhead.
      fp64 elements takes 512KiB per each ciphertext.
      Use backward analysis for diff and forward analysis for value.
  }];
  let cppNamespace = "::hecate::earth";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Analyze error inference structure by automatic differentiation.
      }],
    /*retType=*/"::llvm::SmallVector<double, 2>",
    /*methodName=*/"differentiate",
    /*args=*/(ins "::llvm::ArrayRef<double>":$gradient,"::llvm::ArrayRef<double>":$estimation)
      >,
    InterfaceMethod<
      /*desc=*/[{Estimate value.
      }],
    /*retType=*/"::llvm::SmallVector<double, 1>",
    /*methodName=*/"estimateValue",
    /*args=*/(ins "::llvm::ArrayRef<double>":$estimation)
      >
  ];
}

def HEScaleTypeInterface : TypeInterface<"HEScaleTypeInterface"> {

  let description = [{
    A type interface of CKKS parameter optimization.
      Type that implement this interface can be analyzed and managed by scale management scheme.
  }];
  let cppNamespace = "::hecate::earth";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Check that there is only one ciphertext operand.
      }],
    /*retType=*/"bool",
    /*methodName=*/"isCipher"
      >,
    InterfaceMethod<
      /*desc=*/[{Check that there is only one ciphertext operand.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"toCipher"
      >,
    InterfaceMethod<
      /*desc=*/[{Check that there is only one ciphertext operand.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"toPlain"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a scale of a given type.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getScale"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a scale of a given type.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"switchScale",
    /*args=*/(ins "unsigned":$scale)
      >,
    InterfaceMethod<
      /*desc=*/[{Get a level of a given type. This can be either rescale or ciphertext level.
      }],
    /*retType=*/"unsigned",
    /*methodName=*/"getLevel"
      >,
    InterfaceMethod<
      /*desc=*/[{Get a scale of a given type.
      }],
    /*retType=*/"::hecate::earth::HEScaleTypeInterface",
    /*methodName=*/"switchLevel",
    /*args=*/(ins "unsigned":$level)
      >,
    ];
}



#endif
