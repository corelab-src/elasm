
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CKKS)
add_public_tablegen_target(HecateCKKSTransformsIncGen)
