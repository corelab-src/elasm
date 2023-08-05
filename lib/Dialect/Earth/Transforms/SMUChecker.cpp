
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_SMUCHECKER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct SMUCheckerPass
    : public hecate::earth::impl::SMUCheckerBase<SMUCheckerPass> {
  SMUCheckerPass() {}

  void runOnOperation() override {
    auto func = getOperation();

    auto smu = getAnalysis<hecate::ScaleManagementUnit>();
    markAllAnalysesPreserved();

    llvm::errs() << smu.getNumSMUs() << " " << smu.getNumEdges() << "\n";
    smu.verify();
    for (int i = 0; i < smu.getNumSMUs(); i++) {
      llvm::errs() << "$$$$$$    " << i << "    $$$$\n";
      int j = 0;
      for (auto &&vv : smu.getValueSet(i)) {
        if (++j > 10)
          break;
        llvm::errs() << "##  " << vv << "\n";
      }
    }
    llvm::errs() << "--------------------\n\n\n\n\n\n ";

    std::map<int64_t, std::pair<int64_t, int64_t>> smuLS;
    bool success = true;
    func.walk([&](hecate::earth::HEScaleOpInterface sop) {
      auto ID = smu.getID(sop->getResult(0));
      if (ID == -1) {
        return;
      }

      auto dat = smuLS.find(ID);
      if (dat != smuLS.end()) {
        auto &&record = dat->second;
        std::pair<int64_t, int64_t> data = {sop.getRescaleLevel(),
                                            sop.getScale()};
        if (record != data) {
          llvm::errs() << record.first << " " << record.second << " "
                       << data.first << " " << data.second << "\n";
          sop.dump();
          success = false;
        }
      } else {
        smuLS[ID] = {sop.getRescaleLevel(), sop.getScale()};
      }
    });
    if (!success) {
      func.walk([&](mlir::Block *block) {
        for (auto &&arg : block->getArguments()) {
          llvm::errs() << smu.getID(arg) << ":";
          llvm::errs() << " => ";
          for (auto &&user : arg.getUsers()) {
            llvm::errs() << smu.getID(user->getResult(0)) << " ";
          }
          llvm::errs() << " \n";
          arg.dump();
        }
      });
      func.walk([&](hecate::earth::HEScaleOpInterface sop) {
        llvm::errs() << smu.getID(sop->getResult(0)) << ":";
        for (auto &&arg : sop->getOperands()) {
          llvm::errs() << smu.getID(arg) << " ";
        }
        llvm::errs() << " => ";
        for (auto &&arg : sop->getUsers()) {
          llvm::errs() << smu.getID(arg->getResult(0)) << " ";
        }
        llvm::errs() << " \n";
        sop->getLoc()->dump();
        sop->dump();
      });

      if (!success) {
        assert(0 && "SMU is not correct");
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
