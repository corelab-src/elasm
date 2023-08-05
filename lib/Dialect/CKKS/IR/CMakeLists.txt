add_mlir_dialect_library(CKKSDialect
  CKKSDialect.cpp
  DEPENDS
  HecateCKKSOpsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  )
