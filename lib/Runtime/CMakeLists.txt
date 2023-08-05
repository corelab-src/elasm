
add_library (SEAL_HEVM
  SHARED
  SEAL_HEVM.cpp)
target_link_libraries(SEAL_HEVM
  PUBLIC
  SEAL::seal
  )
