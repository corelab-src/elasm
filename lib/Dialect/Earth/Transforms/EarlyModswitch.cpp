
//===- Bufferize.cpp - Bufferization for Arith ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_EARLYMODSWITCH
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_em"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct EarlyModswitchPass
    : public hecate::earth::impl::EarlyModswitchBase<EarlyModswitchPass> {
  EarlyModswitchPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    SmallVector<mlir::Type, 4> inputTypes;

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
      if (auto op = dyn_cast<hecate::earth::HEScaleOpInterface>(*iter)) {
        // Gather the users and finds the minimum "downFactor"

        uint64_t minModFactor = -1;
        for (auto &&oper : op->getResult(0).getUses()) {
          if (auto oop =
                  dyn_cast<hecate::earth::ModswitchOp>(oper.getOwner())) {
            minModFactor = std::min(minModFactor, oop.getDownFactor());
          } else {
            minModFactor = 0;
          }
        }

        // Check that every user needs the "downFactor"ed level
        if (!minModFactor) {
          continue; // Go to next operation
        }

        // Move the modswitch
        if (auto oop =
                dyn_cast<hecate::earth::ModswitchOp>(op.getOperation())) {
          // Modswitch movement can be absorbed into modswitch
          oop.setDownFactor(oop.getDownFactor() + minModFactor);
          oop.getResult().setType(oop.getScaleType().switchLevel(
              oop.getRescaleLevel() + minModFactor));
        } else {
          // Modswitch is moved to the opreands
          for (int i = 0; i < op->getNumOperands(); i++) {
            auto oper = op->getOperand(i);
            builder.setInsertionPoint(op);
            auto newOper = builder.create<hecate::earth::ModswitchOp>(
                op->getLoc(), oper, minModFactor);
            op->setOperand(i, newOper);
          }
          op->getResult(0).setType(op.getScaleType().switchLevel(
              op.getRescaleLevel() + minModFactor));
        }

        // Change the user modswitch downFactors
        for (auto &&oper : op->getResult(0).getUsers()) {
          if (auto oop = dyn_cast<hecate::earth::ModswitchOp>(oper)) {
            oop.setDownFactor(oop.getDownFactor() - minModFactor);
          }
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
