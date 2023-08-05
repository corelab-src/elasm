
#ifndef HECATE_CONVERSION_PASSES_H
#define HECATE_CONVERSION_PASSES_H

#include "hecate/Conversion/CKKSToCKKS/UpscaleToMulcp.h"
#include "hecate/Conversion/EarthToCKKS/EarthToCKKS.h"

namespace hecate {

#define GEN_PASS_REGISTRATION
#include "hecate/Conversion/Passes.h.inc"

} // namespace hecate

#endif
