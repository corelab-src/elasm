
#ifndef HECATE_CONVERSION_PASSES
#define HECATE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def EarthToCKKSConversion : Pass<"convert-earth-to-ckks", "::mlir::func::FuncOp"> {
  let summary = "Convert Earth dialect to CKKS dialect";
  let description = [{
    This pass converts supported Earth ops to CKKS dialect instructions.
  }];
  let constructor = "::hecate::earth::createEarthToCKKSConversionPass()";
  let dependentDialects = ["hecate::earth::EarthDialect", "hecate::ckks::CKKSDialect", "mlir::tensor::TensorDialect"];
  let options = [
  ];
}
def UpscaleToMulcpConversion : Pass<"convert-upscale-to-mulcp", "::mlir::func::FuncOp"> {
  let summary = "Convert upscale operation to mulcp operation";
  let description = [{
    This pass converts upscale ops to mulcp instructions.
  }];
  let constructor = "::hecate::ckks::createUpscaleToMulcpConversionPass()";
  let dependentDialects = ["hecate::ckks::CKKSDialect", "mlir::tensor::TensorDialect"];
  let options = [
  ];
}

#endif
