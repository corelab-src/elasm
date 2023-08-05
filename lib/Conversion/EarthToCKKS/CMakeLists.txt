add_mlir_conversion_library(HecateEarthToCKKS
  EarthToCKKS.cpp

  DEPENDS
  HecateConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  EarthDialect
  CKKSDialect
  MLIRLLVMCommonConversion
  MLIRTransforms
  )
