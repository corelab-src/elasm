

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"
#include <fstream>
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_SCALEMANAGEMENTSCHEDULER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_sms"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ScaleManagementSchedulerPass
    : public hecate::earth::impl::ScaleManagementSchedulerBase<
          ScaleManagementSchedulerPass> {
  ScaleManagementSchedulerPass() {}

  void runOnOperation() override {
    markAnalysesPreserved<hecate::ScaleManagementUnit>();
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    hecate::ScaleManagementUnit smu =
        getAnalysis<hecate::ScaleManagementUnit>();

    if (!func->hasAttr("sm_plan_edge")) {
      // Add Empty Plan
      func->setAttr("sm_plan_edge", builder.getDenseI64ArrayAttr({}));
      func->setAttr("sm_plan_scale", builder.getDenseI64ArrayAttr({}));
      func->setAttr("sm_plan_level", builder.getDenseI64ArrayAttr({}));
    } else if (!func->hasAttr("no_mutation") ||
               !func->getAttrOfType<BoolAttr>("no_mutation")) {
      // Mutate Plan
      std::random_device rd;
      std::mt19937 gen(rd());
      std::poisson_distribution<int64_t> planRange(
          static_cast<int>(std::sqrt(smu.getNumEdges())));
      /* std::uniform_int_distribution<int64_t> planRange(1, 6); */
      std::uniform_int_distribution<int64_t> smRange(0, smu.getNumEdges() - 1);
      std::uniform_int_distribution<int64_t> scaleRange(-15, 15);
      std::uniform_int_distribution<int64_t> levelRange(0, 2);
      DenseMap<int64_t, std::pair<int64_t, int64_t>> planMap;
      auto plan_num = planRange(gen);
      for (int64_t i = 0; i < plan_num; i++) {
        planMap[smRange(gen)] = {std::max(0L, scaleRange(gen)),
                                 std::max(0L, levelRange(gen))};
      }
      SmallVector<int64_t, 4> planEdge;
      SmallVector<int64_t, 4> planScale;
      SmallVector<int64_t, 4> planLevel;
      for (auto &&it : planMap) {
        planEdge.push_back(it.first);
        planScale.push_back(it.second.first);
        planLevel.push_back(it.second.second);
      }
      func->setAttr("sm_plan_edge", builder.getDenseI64ArrayAttr(planEdge));
      func->setAttr("sm_plan_scale", builder.getDenseI64ArrayAttr(planScale));
      func->setAttr("sm_plan_level", builder.getDenseI64ArrayAttr(planLevel));
    }

    auto &&smEdge = func->getAttrOfType<mlir::DenseI64ArrayAttr>("sm_plan_edge")
                        .asArrayRef();
    auto &&smScale =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("sm_plan_scale")
            .asArrayRef();
    auto &&smLevel =
        func->getAttrOfType<mlir::DenseI64ArrayAttr>("sm_plan_level")
            .asArrayRef();

    for (uint64_t i = 0; i < smEdge.size(); i++) {
      auto &&edges = smu.getEdgeSet(smEdge[i]);
      for (auto &&edge : edges) {
        if (edge->get()
                .getType()
                .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                .isCipher()) {
          builder.setInsertionPoint(edge->getOwner());
          edge->set(builder.create<hecate::earth::ApplyScheduleOp>(
              edge->getOwner()->getLoc(), edge->get(), smScale[i], smLevel[i]));
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << __FILE__ << ":" << __LINE__ << "\n");
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
