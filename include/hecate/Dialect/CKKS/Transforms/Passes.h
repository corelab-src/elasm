

#ifndef HECATE_DIALECT_CKKS_TRANSFORMS_PASSES_H_
#define HECATE_DIALECT_CKKS_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <map>

namespace hecate {

namespace ckks {

#define GEN_PASS_DECL
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"

} // namespace ckks
} // namespace hecate

#endif
