#ifndef HECATE_ANALYSIS_SCALEMANAGEMENTUNIT
#define HECATE_ANALYSIS_SCALEMANAGEMENTUNIT

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallSet.h"
#include <set>

namespace hecate {

struct ScaleManagementUnit {
public:
  ScaleManagementUnit(mlir::Operation *op);

  // Default Implementation
  int64_t getID(mlir::Value v) const;
  int64_t getID(mlir::Operation *op) const;
  int64_t getEdge(mlir::OpOperand *op) const;
  mlir::SmallVector<mlir::OpOperand *, 4> getEdgeSet(int64_t edge) const;
  mlir::SmallVector<mlir::Value, 4> getValueSet(int64_t id) const;
  int64_t getNumEdges() const;
  int64_t getNumSMUs() const;

  // Helper for ELASM
  bool inNoisyGroup(mlir::Operation *op) const;
  bool inNoisyGroup(mlir::Value v) const;

  // This is a helper for passes.
  // Should not be called in analysis.
  void attach();
  void detach();

  bool verify() const;
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &);

private:
  void build();

  int64_t idMax;
  int64_t edgeMax;

  llvm::DenseMap<mlir::Value, int64_t> smuIds;
  llvm::DenseMap<mlir::OpOperand *, int64_t> smuEdges;
  llvm::SmallVector<mlir::SmallVector<mlir::Value, 4>, 4> idToValue;
  llvm::SmallVector<mlir::SmallVector<mlir::OpOperand *, 4>, 4> edgeToOper;

  llvm::SmallVector<bool, 4> noisyMap;
  /* llvm::DenseMap<int64_t, mlir::SmallVector<mlir::OpOperand *, 4>>
   * edgeToOper; */
  /* llvm::DenseMap<int64_t, mlir::SmallVector<mlir::Value, 4>> idToValue; */
  mlir::Operation *_op;
};
} // namespace hecate

#endif
