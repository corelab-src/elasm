
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_PRIVATIZECONSTANT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct PrivatizeConstantPass
    : public hecate::earth::impl::PrivatizeConstantBase<PrivatizeConstantPass> {
  PrivatizeConstantPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    func.walk([&](hecate::earth::HEScaleOpInterface hop) {
      OpBuilder builder(hop);
      for (size_t i = 0; i < hop.getOperation()->getNumOperands(); i++) {
        if (auto &&cop =
                hop->getOperand(i).getDefiningOp<hecate::earth::ConstantOp>()) {
          hop->setOperand(i, builder.create<hecate::earth::ConstantOp>(
                                 cop.getLoc(), cop.getType(), cop.getValue(),
                                 cop.getRmsVar()));
        }
      }
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
