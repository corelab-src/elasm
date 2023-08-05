
#ifndef HECATE_SUPPORT_SUPPORT
#define HECATE_SUPPORT_SUPPORT

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

namespace hecate {

inline llvm::SmallVector<int, 4> naf(int value) {
  llvm::SmallVector<int, 4> res;

  // Record the sign of the original value and compute abs
  bool sign = value < 0;
  value = std::abs(value);

  // Transform to non-adjacent form (NAF)
  for (int i = 0; value; i++) {
    int zi = (value & int(0x1)) ? 2 - (value & int(0x3)) : 0;
    value = (value - zi) >> 1;
    if (zi) {
      res.push_back((sign ? -zi : zi) * (1 << i));
    }
  }

  return res;
}

inline void setIntegerAttr(llvm::StringRef name, mlir::Value v, int64_t data) {
  unsigned argnum = 0;
  mlir::Operation *op = nullptr;
  if (auto ba = v.dyn_cast<mlir::BlockArgument>()) {
    argnum = ba.getArgNumber();
    op = ba.getOwner()->getParentOp();
  } else if (auto opr = v.dyn_cast<mlir::OpResult>()) {
    argnum = opr.getResultNumber();
    op = opr.getOwner();
  } else {
    assert(0 && "Value should be either block argument or op result");
  }
  auto builder = mlir::OpBuilder(op);
  op->setAttr(std::string(name) + std::to_string(argnum),
              builder.getI64IntegerAttr(data));
}

inline int64_t getIntegerAttr(llvm::StringRef name, mlir::Value v) {
  unsigned argnum = 0;
  mlir::Operation *op = nullptr;
  if (auto ba = v.dyn_cast<mlir::BlockArgument>()) {
    argnum = ba.getArgNumber();
    op = ba.getOwner()->getParentOp();
  } else if (auto opr = v.dyn_cast<mlir::OpResult>()) {
    argnum = opr.getResultNumber();
    op = opr.getOwner();
  } else {
    assert(0 && "Value should be either block argument or op result");
  }
  if (auto attr = op->getAttr(std::string(name) + std::to_string(argnum))) {
    return attr.dyn_cast<mlir::IntegerAttr>().getInt();
  } else {
    return -1;
  }
}

} // namespace hecate

#endif
