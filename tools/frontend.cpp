
#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <memory>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace mlir;

void handler(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

namespace hecate {

using valueID = size_t;
using funcID = size_t;

struct Context {
  Context();
  mlir::MLIRContext ctxt;
  mlir::OwningOpRef<mlir::ModuleOp> mod;
  std::unique_ptr<mlir::OpBuilder> builder;
  llvm::SmallVector<mlir::Value, 32> valueMap;
  llvm::SmallVector<mlir::func::FuncOp, 32> funcMap;
};

Context::Context() : ctxt(), mod(), builder() {

  ctxt.getOrLoadDialect<mlir::func::FuncDialect>();
  auto ed = ctxt.getOrLoadDialect<hecate::earth::EarthDialect>();

  auto tmp = std::make_unique<mlir::OpBuilder>(&ctxt);
  builder.swap(tmp);

  mod = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder->getUnknownLoc()));
}

extern "C" {

valueID createConstant(Context *ctxt, double *data, int64_t len, char *filename,
                       size_t line) {
  auto &&builder = *ctxt->builder;
  auto cons = builder.create<hecate::earth::ConstantOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      llvm::ArrayRef(data, len));

  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}
funcID createFunc(Context *ctxt, char *name, int *inputTys, size_t len,
                  char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&funcMap = ctxt->funcMap;
  llvm::SmallVector<mlir::Type, 4> arg_types(len);
  std::transform(inputTys, inputTys + len, arg_types.begin(), [&](auto a) {
    return mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t, 1>{1},
        builder.getType<hecate::earth::CipherType>(0, 0));
  });
  auto funcType = builder.getFunctionType(
      arg_types, mlir::RankedTensorType::get(
                     llvm::SmallVector<int64_t, 1>{1},
                     builder.getType<hecate::earth::CipherType>(0, 0)));
  auto funcOp = mlir::func::FuncOp::create(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      std::string("_hecate_") + name, funcType);
  funcMap.push_back(funcOp);
  ctxt->mod->push_back(funcOp);
  return funcMap.size() - 1;
}

void initFunc(Context *ctxt, funcID fun, valueID *args, size_t len) {
  auto &&funcOp = ctxt->funcMap[fun];
  auto &&valueMap = ctxt->valueMap;
  auto entryBlock = funcOp.addEntryBlock();
  auto funcInput = entryBlock->getArguments();
  ctxt->builder->setInsertionPointToStart(entryBlock);
  {
    int i = 0;
    for (auto a : funcInput) {
      valueMap.push_back(a);
      args[i++] = valueMap.size() - 1;
    }
  }
}

char *save(Context *c, char *const_name, char *mlir_name) {
  c->mod->getOperation()->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                                  c->builder->getStringAttr(mlir_name));
  std::string s_const_name(const_name);
  mlir::PassManager pm(&c->ctxt);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      earth::createElideConstant({s_const_name + "/"}));
  pm.addNestedPass<func::FuncOp>(earth::createPrivatizeConstant());
  pm.addPass(createCanonicalizerPass());

  auto ret = pm.run(*c->mod);

  std::error_code EC;
  llvm::raw_fd_ostream outputFile(mlir_name, EC);
  c->mod->print(outputFile, mlir::OpPrintingFlags()
                                .printGenericOpForm()
                                .enableDebugInfo()
                                .useLocalScope());
  c->valueMap.clear();
  c->funcMap.clear();
  c->mod.release();
  return mlir_name;
}

/* Unary Operation */
valueID createUnary(Context *ctxt, size_t opcode, valueID lhs, char *filename,
                    size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  auto &&source = valueMap[lhs];
  switch (opcode) {
  case 0: {
    break;
  }
  case 13: {
    auto res = builder.create<hecate::earth::NegateOp>(location, source);
    valueMap.push_back(res);
    break;
  }

  default:
    assert(0 && "Unary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

/* Binary Operation */
valueID createBinary(Context *ctxt, size_t opcode, valueID lhs, valueID rhs,
                     char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);

  auto &&srcl = valueMap[lhs];
  auto &&srcr = valueMap[rhs];

  switch (opcode) {
  case 6: {
    auto res = builder.create<hecate::earth::AddOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }
  case 7: {
    auto neg = builder.create<hecate::earth::NegateOp>(location, srcr);
    valueMap.push_back(neg);
    auto res = builder.create<hecate::earth::AddOp>(location, srcl, neg);
    valueMap.push_back(res);
    break;
  }

  case 8: {
    auto res = builder.create<hecate::earth::MulOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }

  default:
    assert(0 && "Binary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

valueID createRotation(Context *ctxt, size_t valueID, int offset,
                       char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&srcl = ctxt->valueMap[valueID];
  auto cons = builder.create<hecate::earth::RotateOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0), srcl,
      offset);
  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}

void setOutput(Context *ctxt, funcID fun, valueID *ret, size_t len) {
  llvm::SmallVector<mlir::Value, 2> rets;
  llvm::SmallVector<mlir::Type, 1> types;
  for (int i = 0; i < len; i++) {
    rets.push_back(ctxt->valueMap[ret[i]]);
    types.push_back(ctxt->valueMap[ret[i]].getType());
  }
  auto func = ctxt->funcMap[fun];
  ctxt->builder->create<mlir::func::ReturnOp>(func.getLoc(), rets);
  auto retType = func.getFunctionType();
  func.setFunctionType(
      ctxt->builder->getFunctionType(retType.getInputs(), types));
}

Context *init() {
  /* signal(SIGSEGV, handler); */
  return new ::hecate::Context();
}
void finalize(Context *ctxt) { delete ctxt; }
} // namespace hecate
} // namespace hecate

/* int main() {} */
