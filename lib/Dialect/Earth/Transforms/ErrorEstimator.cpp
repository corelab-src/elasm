
#include "hecate/Dialect/Earth/Analysis/AutoDifferentiation.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_ERRORESTIMATOR
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_ub"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ErrorEstimatorPass
    : public hecate::earth::impl::ErrorEstimatorBase<ErrorEstimatorPass> {
  ErrorEstimatorPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    auto &&diff = getAnalysis<hecate::AutoDifferentiation>();

    markAllAnalysesPreserved();
    double error_square = 0;

    // We cannot track both self term (x+x then error should be doubled but
    // forward analysis makes the error sqrt(2)-ed) and quadratic term (x1*x2
    // adds e1*e2 term) efficiently.
    //
    // Forward analysis cannot track self term but can track quadratic term
    // Backward analysis can track self term but cannot track quadratic term
    //
    // ELASM paper uses forward analysis but I changed implementation
    // Because quadratic term should be smaller enough to guarantee correctness
    //
    // I think neither quadratic term tracking on backward nor
    // self term tracking on forward is practical.
    // We can utilize high level information to selectively use analysis
    // Additional evaluation is required.

    func.walk([&](hecate::earth::HEProfInterface pop) {
      auto df = diff.getBackDiff(pop.getOperation());
      error_square += pop.getNoise() * pop.getNum() *
                      std::pow(diff.getBackDiff(pop.getOperation()), 2) /
                      std::exp2(pop.getNoiseScale());
    });

    func->setAttr("est_error",
                  builder.getF64FloatAttr(std::sqrt(error_square)));
    LLVM_DEBUG(llvm::dbgs() << __FILE__ << ":" << __LINE__ << "\n");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
