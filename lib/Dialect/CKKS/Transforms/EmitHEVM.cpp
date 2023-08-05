
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <filesystem>
#include <fstream>

#include "hecate/Support/HEVMHeader.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITHEVM
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct EmitHEVMPass : public hecate::ckks::impl::EmitHEVMBase<EmitHEVMPass> {
  EmitHEVMPass() {}
  EmitHEVMPass(hecate::ckks::EmitHEVMOptions ops) { this->prefix = ops.prefix; }

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto &&func = getOperation();
    HEVMHeader header;
    ConfigBody config_body;
    header.hevm_header_size = sizeof(HEVMHeader);
    header.config_header.arg_length = func.getNumArguments();
    header.config_header.res_length = func.getNumResults();

    config_body.config_body_length = sizeof(ConfigBody);
    SmallVector<uint64_t, 4> config_body_ints;
    SmallVector<HEVMOperation, 4> insts;

    int64_t cipher_registers = 0;
    int64_t plain_registers = 0;
    llvm::DenseMap<mlir::Value, int64_t> cipher_register_file;
    llvm::DenseMap<mlir::Value, int64_t> plain_register_file;
    for (auto &&arg : func.getArguments()) {
      auto tt = arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
      if (tt.getNumPoly() == 1) {
        plain_register_file.insert({arg, plain_registers++});
      } else {
        cipher_register_file.insert({arg, cipher_registers++});
      }
    }
    func.walk([&](mlir::Operation *op) {
      if (auto alloc = dyn_cast<mlir::tensor::EmptyOp>(op)) {
        auto tt = alloc.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
        HEVMOperation heops;
        heops.opcode = -1;
        insts.push_back(heops);

        if (tt.getNumPoly() == 1) {
          plain_register_file.insert({alloc, plain_registers++});
        } else {
          cipher_register_file.insert({alloc, cipher_registers++});
        }
      } else if (auto ops = dyn_cast<hecate::ckks::HEVMOpInterface>(op)) {
        HEVMOperation heops =
            ops.getHEVMOperation(plain_register_file, cipher_register_file);
        insts.push_back(heops);

        if (heops.opcode > 0) {
          cipher_register_file.insert({op->getResult(0), heops.dst});
        } else {
          plain_register_file.insert({op->getResult(0), heops.dst});
        }
      }
    });

    SmallVector<int64_t> ret_dst;
    auto retOp =
        dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
    for (auto arg : retOp.getOperands()) {
      ret_dst.push_back(cipher_register_file[arg]);
    }

    auto arg_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_scale").asArrayRef();
    auto arg_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_level").asArrayRef();
    auto res_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_scale").asArrayRef();
    auto res_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_level").asArrayRef();

    config_body.num_operations = insts.size();
    config_body.num_ctxt_buffer = cipher_registers;
    config_body.num_ptxt_buffer = plain_registers;
    config_body.init_level =
        func->getAttrOfType<IntegerAttr>("init_level").getInt();

    config_body_ints.append(arg_scale_array.begin(), arg_scale_array.end());
    config_body_ints.append(arg_level_array.begin(), arg_level_array.end());
    config_body_ints.append(res_scale_array.begin(), res_scale_array.end());
    config_body_ints.append(res_level_array.begin(), res_level_array.end());
    config_body_ints.append(ret_dst.begin(), ret_dst.end());

    config_body.config_body_length +=
        config_body_ints.size() * sizeof(uint64_t);

    std::filesystem::path printpath(prefix.getValue());

    printpath = std::string(printpath) + "." + func.getName().str() + ".hevm";

    std::ofstream of(printpath, std::ios::binary);
    of.write((char *)&header, sizeof(HEVMHeader));
    of.write((char *)&config_body, sizeof(ConfigBody));
    of.write((char *)config_body_ints.data(),
             config_body_ints.size() * sizeof(uint64_t));
    of.write((char *)insts.data(), insts.size() * sizeof(HEVMOperation));
    of.close();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
  }
};
} // namespace
