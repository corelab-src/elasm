
add_mlir_dialect_library(CKKSTransforms
  RemoveLevel.cpp
  ReuseBuffer.cpp
  EmitHEVM.cpp
  DEPENDS
  HecateCKKSTransformsIncGen
  LINK_LIBS
  PUBLIC
  CKKSDialect
  MLIRIR
  )
