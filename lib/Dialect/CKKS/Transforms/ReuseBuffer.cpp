

#include <map>

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_REUSEBUFFER
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ReuseBufferPass
    : public hecate::ckks::impl::ReuseBufferBase<ReuseBufferPass> {
  ReuseBufferPass() {}

  void runOnOperation() override {
    auto &&func = getOperation();
    mlir::OpBuilder builder(func);
    mlir::Liveness l(func);
    SmallVector<Value, 4> garbage;
    func.walk([&](mlir::DestinationStyleOpInterface op) {
      for (int i = 0; i < op.getNumDpsInputs(); i++) {
        auto v = op.getDpsInputOperand(i);
        if (auto tt = hecate::ckks::getPolyType(v->get())) {
          if (tt.getNumPoly() == 1)
            continue;
          if (l.isDeadAfter(v->get(), op) && !garbage.empty() &&
              v->get() != garbage.back()) {
            garbage.push_back(v->get());
          }
        }
      }
      for (int i = 0; i < op.getNumDpsInits(); i++) {
        auto v = op.getDpsInitOperand(i);
        if (auto tt = hecate::ckks::getPolyType(v->get())) {
          if (tt.getNumPoly() == 1)
            continue;
          if (!garbage.empty()) {
            op.getDpsInitOperand(i)->set(garbage.pop_back_val());
          }
        }
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
  }
};
} // namespace
