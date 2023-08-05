

#ifndef HECATE_DIALECT_CKKS_TRANSFORMS_PASSES
#define HECATE_DIALECT_CKKS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def RemoveLevel : Pass<"remove-level", "::mlir::func::FuncOp"> {
  let summary = "Remove Level Informations for Ciphertext-granularity Register Allocation";
  let description = [{
  }];
}

def ReuseBuffer : Pass<"reuse-buffer", "::mlir::func::FuncOp"> {
  let summary = "Remove Level Informations for Ciphertext-granularity Register Allocation";
  let description = [{
  }];
}

def EmitHEVM : Pass<"emit-hevm", "::mlir::func::FuncOp"> {
  let summary = "Remove Level Informations for Ciphertext-granularity Register Allocation";
  let description = [{
  }];
  let options = [
    Option<"prefix", "prefix", "std::string", /*default=*/[{""}],
    "Name prefix of output file">,
  ];

}

#endif
