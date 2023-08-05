#ifndef HECATE_DIALECT_ARITH_IR_ARITHOPS_H
#define HECATE_DIALECT_ARITH_IR_ARITHOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>

#include "hecate/Support/Support.h"

#include "hecate/Dialect/Earth/IR/ForwardManagementInterface.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"

#include "hecate/Dialect/Earth/IR/EarthOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOps.h.inc"

namespace hecate {
namespace earth {
::hecate::earth::HEScaleTypeInterface getScaleType(mlir::Value v);
::mlir::RankedTensorType getTensorType(mlir::Value v);
} // namespace earth
} // namespace hecate
#endif
