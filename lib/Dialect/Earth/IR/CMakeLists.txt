add_mlir_dialect_library(EarthDialect
  EarthDialect.cpp
  HEParameterInterface.cpp
  ForwardManagementInterface.cpp
  DEPENDS
  HecateEarthOpsIncGen
  HecateEarthTransformsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  )
