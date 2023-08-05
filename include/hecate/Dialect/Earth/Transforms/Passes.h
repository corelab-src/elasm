

#ifndef HECATE_DIALECT_EARTH_TRANSFORMS_PASSES_H_
#define HECATE_DIALECT_EARTH_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <map>

namespace hecate {

namespace earth {

#define GEN_PASS_DECL
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"

/* #define GEN_PASS_DECL_ARITHINTRANGEOPTS */
/* #include "hecate/Dialect/Earth/Transforms/Passes.h.inc" */

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"

} // namespace earth
} // namespace hecate
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"

#endif
