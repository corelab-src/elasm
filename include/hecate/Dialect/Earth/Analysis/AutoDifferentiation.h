
#ifndef HECATE_ANALYSIS_AUTODIFFERENTIATION
#define HECATE_ANALYSIS_AUTODIFFERENTIATION

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallSet.h"
#include <set>

namespace hecate {

struct AutoDifferentiation {
public:
  AutoDifferentiation(mlir::Operation *op);

  // Passing Operation and Value gives same result
  double getBackDiff(mlir::Operation *op);
  double getBackDiff(mlir::Value v);

  double getBackDiff(mlir::OpOperand &oper);

  double getValueEstimation(mlir::Operation *op);
  double getValueEstimation(mlir::Value v);

private:
  void build();

  llvm::DenseMap<mlir::OpOperand *, double> operandDiffMap;
  llvm::DenseMap<mlir::Value, double> valueDiffMap;
  llvm::DenseMap<mlir::Value, double> valueMap;
  mlir::Operation *_op;
};
} // namespace hecate

#endif
