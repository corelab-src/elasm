
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_LATENCYESTIMATOR
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct LatencyEstimatorPass
    : public hecate::earth::impl::LatencyEstimatorBase<LatencyEstimatorPass> {
  LatencyEstimatorPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    markAllAnalysesPreserved();
    double latency = 0;

    func.walk([&](hecate::earth::HEProfInterface pop) {
      latency += pop.getLatency() * pop.getNum();
    });

    func->setAttr("est_latency", builder.getF64FloatAttr(latency));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
