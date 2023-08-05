
add_mlir_conversion_library(HecateCKKSCommonConversion
  PolyTypeConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  EarthDialect
  CKKSDialect
  MLIRLLVMCommonConversion
  MLIRTransforms
  )
