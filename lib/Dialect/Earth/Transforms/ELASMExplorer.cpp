

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_ELASMEXPLORER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "elasm"

namespace {
/// Pass to bufferize Arith ops.
struct ELASMExplorerPass
    : public hecate::earth::impl::ELASMExplorerBase<ELASMExplorerPass> {
  ELASMExplorerPass() {}

  ELASMExplorerPass(hecate::earth::ELASMExplorerOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
    this->parallel = ops.parallel;
    this->num_iter = ops.num_iter;
    this->beta = ops.beta;
    this->gamma = ops.gamma;
  }
  ELASMExplorerPass(std::pair<int64_t, int64_t> ops) {
    this->waterline = ops.first;
    this->output_val = ops.second;
    this->parallel = 20;
    this->num_iter = 1000;
    this->beta = 50;
    this->gamma = 50;
  }

  double costFunc(double cost, double noise) {
    return std::sqrt(cost) * (beta + std::log2(noise));
  }

  void runOnOperation() override {

    auto func = getOperation();
    mlir::OpBuilder builder(func);

    markAnalysesPreserved<hecate::ScaleManagementUnit>();
    hecate::ScaleManagementUnit smu =
        getAnalysis<hecate::ScaleManagementUnit>();

    smu.attach();

    SmallVector<func::FuncOp> funcs;
    SmallVector<std::tuple<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>,
                           SmallVector<int64_t, 4>>>
        plans(parallel);

    SmallVector<double> costs(parallel, std::numeric_limits<double>::max());
    double optcost = std::numeric_limits<double>::max();
    std::tuple<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>,
               SmallVector<int64_t, 4>>
        optplan;

    auto mod = mlir::ModuleOp::create(func.getLoc());

    PassManager pm(mod.getContext());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createScaleManagementScheduler());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createSNRRescaling({waterline, output_val}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createUpscaleBubbling());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createErrorEstimator());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());

    std::uniform_real_distribution<double> dd(0.0, 1.0);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int n = 0; n < num_iter; n++) {
      /* llvm::errs() << n << "th Run Start\n"; */
      for (int i = 0; i < parallel; i++) {
        auto dup = func.clone();
        dup.setName((func.getName() + "_" + std::to_string(i)).str());
        mod.push_back(dup);
        funcs.push_back(dup);
        dup->setAttr("sm_plan_edge",
                     builder.getDenseI64ArrayAttr(std::get<0>(plans[i])));
        dup->setAttr("sm_plan_scale",
                     builder.getDenseI64ArrayAttr(std::get<1>(plans[i])));
        dup->setAttr("sm_plan_level",
                     builder.getDenseI64ArrayAttr(std::get<2>(plans[i])));
      }

      if (pm.run(mod).failed()) {
        assert(0 && "Pass failed inside ELASM explorer");
        pm.dump();
      }
      /* llvm::errs() << n << "th Run Done\n"; */
      for (int i = 0; i < parallel; i++) {

        double thres = dd(gen);
        func::FuncOp &target = funcs[i];
        double cost =
            costFunc(target->getAttrOfType<mlir::FloatAttr>("est_latency")
                         .getValueAsDouble(),
                     target->getAttrOfType<mlir::FloatAttr>("est_error")
                         .getValueAsDouble());

        double alpha =
            std::min(1.0, std::pow(2.0, -gamma * (1.0 - costs[i] / cost)));

        if (thres < alpha) {
          plans[i] = {
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_edge")
                      .asArrayRef()),
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_scale")
                      .asArrayRef()),
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_level")
                      .asArrayRef())};
          costs[i] = cost;
        }
        if (cost < optcost) {
          LLVM_DEBUG(llvm::dbgs()
                     << optcost << " " << cost << " "
                     << target->getAttrOfType<mlir::FloatAttr>("est_latency")
                            .getValueAsDouble()
                     << " "
                     << target->getAttrOfType<mlir::FloatAttr>("est_error")
                            .getValueAsDouble()
                     << "\n");
          optplan = {
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_edge")
                      .asArrayRef()),
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_scale")
                      .asArrayRef()),
              SmallVector<int64_t, 4>(
                  target->getAttrOfType<DenseI64ArrayAttr>("sm_plan_level")
                      .asArrayRef())};
          optcost = cost;
        }
        funcs[i].erase();
      }
      funcs.clear();
    }
    func->setAttr("sm_plan_edge",
                  builder.getDenseI64ArrayAttr(std::get<0>(optplan)));
    func->setAttr("sm_plan_scale",
                  builder.getDenseI64ArrayAttr(std::get<1>(optplan)));
    func->setAttr("sm_plan_level",
                  builder.getDenseI64ArrayAttr(std::get<2>(optplan)));
    func->setAttr("no_mutation", builder.getBoolAttr(true));

    /* auto p = pm.run(func); */
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
