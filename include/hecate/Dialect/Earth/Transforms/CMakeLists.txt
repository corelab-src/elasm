set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Earth)
add_public_tablegen_target(HecateEarthTransformsIncGen)

# add_mlir_doc(Passes EarthPasses ./ -gen-pass-doc)
