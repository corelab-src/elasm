
#ifndef HECATE_DIALECT_CKKS_IR_ARITHOPS_H
#define HECATE_DIALECT_CKKS_IR_ARITHOPS_H

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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>

#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"

#include "hecate/Dialect/CKKS/IR/CKKSOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOps.h.inc"

template <typename ConcreteType>
class SameOperandsAndResultLevel
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameOperandsAndResultLevel> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return mlir::success();
  }
};

template <typename ConcreteType>
class SameOperandsAndLowerResultLevel
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SameOperandsAndLowerResultLevel> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return mlir::success();
  }
};

namespace hecate {
namespace ckks {
::hecate::ckks::PolyTypeInterface getPolyType(mlir::Value v);
::mlir::RankedTensorType getTensorType(mlir::Value v);
} // namespace ckks
} // namespace hecate
#endif
