
add_library (HecateFrontend
  SHARED
  frontend.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(HecateFrontend
  EarthDialect
  EarthTransforms
  ${dialect_libs}
  ${conversion_libs}
  )

add_executable (hecate-opt
  optimizer.cpp
  )

target_link_libraries(hecate-opt
  MLIREmitCDialect
  MLIROptLib
  EarthDialect
  CKKSDialect
  EarthTransforms
  CKKSTransforms
  HecateCKKSCommonConversion
  HecateEarthToCKKS
  HecateCKKSToCKKS
  ${dialect_libs}
  ${conversion_libs}
  )
