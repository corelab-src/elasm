#include "hecate/Dialect/Earth/Analysis/AutoDifferentiation.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"

using namespace hecate;
using namespace mlir;

hecate::AutoDifferentiation::AutoDifferentiation(mlir::Operation *op)
    : _op(op) {
  build();
}

void hecate::AutoDifferentiation::build() {

  SmallVector<hecate::earth::HEAutoDiffInterface, 4> hops;
  _op->walk([&](mlir::Block *block) {
    for (auto &&arg : block->getArguments()) {
      valueMap[arg] = 0.0;
    }
  });

  _op->walk([&](hecate::earth::HEAutoDiffInterface sop) {
    SmallVector<double, 4> estimation;
    hops.push_back(sop);
    for (auto &&arg : sop->getOperands()) {
      auto it = valueMap.try_emplace(arg, 1.0);
      estimation.push_back(it.first->second);
    }
    auto &&resultEst = sop.estimateValue(estimation);
    for (auto &&[val, est] :
         llvm::zip(sop.getOperation()->getResults(), resultEst)) {
      valueMap[val] = est;
    }
  });

  // We may need masking-aware propagation
  for (auto &&hop : llvm::reverse(hops)) {
    SmallVector<double, 4> gradients;
    for (auto &&val : hop.getOperation()->getResults()) {
      double gradient = 0.0;
      for (auto &&uses : val.getUses()) {
        auto it = operandDiffMap.try_emplace(&uses, 1.0);
        gradient += it.first->second;
      }
      valueDiffMap[val] = gradient;
      gradients.push_back(gradient);
    }
    SmallVector<double, 4> estimation;
    for (auto &&arg : hop->getOperands()) {
      auto it = valueMap.try_emplace(arg, 1.0);
      estimation.push_back(it.first->second);
    }
    auto &&resultGrad = hop.differentiate(gradients, estimation);
    for (auto &&[oper, grad] : llvm::zip(hop->getOpOperands(), resultGrad)) {
      operandDiffMap[&oper] = grad;
    }
  }
}

double AutoDifferentiation::getBackDiff(mlir::Operation *op) {
  if (op->getNumResults()) {
    return getBackDiff(op->getResult(0));
  } else {
    return 1.0;
  }
}

double AutoDifferentiation::getBackDiff(mlir::Value v) {
  auto &&i = valueDiffMap.find(v);
  if (i != valueDiffMap.end()) {
    return i->second;
  } else {
    return 1.0;
  }
}

double AutoDifferentiation::getBackDiff(mlir::OpOperand &oper) {
  auto &&i = operandDiffMap.find(&oper);
  if (i != operandDiffMap.end())
    return i->second;
  else {
    return 1.0;
  }
}

double AutoDifferentiation::getValueEstimation(mlir::Operation *op) {
  if (op->getNumResults()) {
    getBackDiff(op->getResult(0));
  } else {
    return 1.0;
  }
}
double AutoDifferentiation::getValueEstimation(mlir::Value v) {
  auto &&i = valueMap.find(v);
  if (i != valueMap.end())
    return i->second;
  else {
    return 1.0;
  }
}
