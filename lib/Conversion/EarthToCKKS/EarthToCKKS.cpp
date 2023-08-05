

#include "hecate/Conversion/EarthToCKKS/EarthToCKKS.h"
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace hecate {
#define GEN_PASS_DEF_EARTHTOCKKSCONVERSION
#include "hecate/Conversion/Passes.h.inc"
} // namespace hecate

using namespace mlir;
using namespace hecate;

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

/// Directly lower to LLVM op.
struct ConstantOpLowering
    : public OpConversionPattern<hecate::earth::ConstantOp> {
  using OpConversionPattern<hecate::earth::ConstantOp>::ConversionPattern;
  ConstantOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt,
                     int64_t init_level)
      : OpConversionPattern<hecate::earth::ConstantOp>(converter, ctxt),
        init_level(init_level) {}

  LogicalResult
  matchAndRewrite(hecate::earth::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
  int64_t init_level;
};

struct MulOpLowering : public OpConversionPattern<hecate::earth::MulOp> {
  using OpConversionPattern<hecate::earth::MulOp>::ConversionPattern;
  MulOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::MulOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AddOpLowering : public OpConversionPattern<hecate::earth::AddOp> {
  using OpConversionPattern<hecate::earth::AddOp>::ConversionPattern;
  AddOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::AddOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct NegateOpLowering : public OpConversionPattern<hecate::earth::NegateOp> {
  using OpConversionPattern<hecate::earth::NegateOp>::ConversionPattern;
  NegateOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::NegateOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RotateOpLowering : public OpConversionPattern<hecate::earth::RotateOp> {
  using OpConversionPattern<hecate::earth::RotateOp>::ConversionPattern;
  RotateOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::RotateOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::RotateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct UpscaleOpLowering
    : public OpConversionPattern<hecate::earth::UpscaleOp> {
  using OpConversionPattern<hecate::earth::UpscaleOp>::ConversionPattern;
  UpscaleOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::UpscaleOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::UpscaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RescaleOpLowering
    : public OpConversionPattern<hecate::earth::RescaleOp> {
  using OpConversionPattern<hecate::earth::RescaleOp>::ConversionPattern;
  RescaleOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::RescaleOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::RescaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ModswitchOpLowering
    : public OpConversionPattern<hecate::earth::ModswitchOp> {
  using OpConversionPattern<hecate::earth::ModswitchOp>::ConversionPattern;
  ModswitchOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<hecate::earth::ModswitchOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::earth::ModswitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReturnOpLowering : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern<mlir::func::ReturnOp>::ConversionPattern;
  ReturnOpLowering(mlir::TypeConverter &converter, MLIRContext *ctxt)
      : OpConversionPattern<mlir::func::ReturnOp>(converter, ctxt) {}

  LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(hecate::earth::ConstantOp op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  auto tt = op.getType()
                .getElementType()
                .dyn_cast<hecate::earth::HEScaleTypeInterface>();

  rewriter.replaceOpWithNewOp<ckks::EncodeOp>(
      op, dst, adaptor.getValue().dyn_cast<IntegerAttr>().getInt(),
      tt.getScale(), init_level - tt.getLevel());

  return success();
}

//===----------------------------------------------------------------------===//
// MulOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
MulOpLowering::matchAndRewrite(hecate::earth::MulOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  if (ckks::getPolyType(adaptor.getLhs()).getNumPoly() > 1 &&
      ckks::getPolyType(adaptor.getRhs()).getNumPoly() > 1) {
    rewriter.replaceOpWithNewOp<ckks::MulCCOp>(op, dst, adaptor.getLhs(),
                                               adaptor.getRhs());
  } else if (ckks::getPolyType(adaptor.getLhs()).getNumPoly() == 1) {
    rewriter.replaceOpWithNewOp<ckks::MulCPOp>(op, dst, adaptor.getRhs(),
                                               adaptor.getLhs());
  } else {
    rewriter.replaceOpWithNewOp<ckks::MulCPOp>(op, dst, adaptor.getLhs(),
                                               adaptor.getRhs());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AddOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
AddOpLowering::matchAndRewrite(hecate::earth::AddOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  if (ckks::getPolyType(adaptor.getLhs()).getNumPoly() > 1 &&
      ckks::getPolyType(adaptor.getRhs()).getNumPoly() > 1) {
    rewriter.replaceOpWithNewOp<ckks::AddCCOp>(op, dst, adaptor.getLhs(),
                                               adaptor.getRhs());
  } else if (ckks::getPolyType(adaptor.getLhs()).getNumPoly() == 1) {
    rewriter.replaceOpWithNewOp<ckks::AddCPOp>(op, dst, adaptor.getRhs(),
                                               adaptor.getLhs());
  } else {
    rewriter.replaceOpWithNewOp<ckks::AddCPOp>(op, dst, adaptor.getLhs(),
                                               adaptor.getRhs());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NegateOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
NegateOpLowering::matchAndRewrite(hecate::earth::NegateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  rewriter.replaceOpWithNewOp<ckks::NegateCOp>(op, dst, adaptor.getValue());
  return success();
}

//===----------------------------------------------------------------------===//
// RotateOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
RotateOpLowering::matchAndRewrite(hecate::earth::RotateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  rewriter.replaceOpWithNewOp<ckks::RotateCOp>(op, dst, adaptor.getValue(),
                                               adaptor.getOffset());
  return success();
}

//===----------------------------------------------------------------------===//
// UpscaleOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
UpscaleOpLowering::matchAndRewrite(hecate::earth::UpscaleOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(),
      getTypeConverter()->convertType(op.getType().getElementType()));

  rewriter.replaceOpWithNewOp<ckks::UpscaleCOp>(op, dst, adaptor.getValue(),
                                                adaptor.getUpFactor());
  return success();
}

//===----------------------------------------------------------------------===//
// RescaleOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
RescaleOpLowering::matchAndRewrite(hecate::earth::RescaleOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto elemType = getTypeConverter()
                      ->convertType(op.getType().getElementType())
                      .dyn_cast<ckks::PolyTypeInterface>();
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(), elemType);

  rewriter.replaceOpWithNewOp<ckks::RescaleCOp>(op, dst, adaptor.getValue());
  return success();
}

//===----------------------------------------------------------------------===//
// ModswitchOpLowering
//===----------------------------------------------------------------------===//

LogicalResult ModswitchOpLowering::matchAndRewrite(
    hecate::earth::ModswitchOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto elemType = getTypeConverter()
                      ->convertType(op.getType().getElementType())
                      .dyn_cast<ckks::PolyTypeInterface>();
  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(), elemType);

  rewriter.replaceOpWithNewOp<ckks::ModswitchCOp>(op, dst, adaptor.getValue(),
                                                  adaptor.getDownFactor());
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpLowering::matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {

  rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct EarthToCKKSConversion
    : public hecate::impl::EarthToCKKSConversionBase<EarthToCKKSConversion> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    auto func = getOperation();

    mlir::RewritePatternSet patterns(&getContext());

    auto level_attr = func->getAttrOfType<IntegerAttr>("init_level");
    int64_t base_level = level_attr ? level_attr.getInt() : 13;

    hecate::PolyTypeConverter converter(base_level);
    target.addLegalDialect<hecate::ckks::CKKSDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp fop) {
      return converter.isSignatureLegal(fop.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp rop) {
      return llvm::all_of(rop.getOperands(), [&](auto &&v) {
        return converter.isLegal(v.getType());
      });
    });
    target.addIllegalDialect<hecate::earth::EarthDialect>();

    hecate::earth::populateEarthToCKKSConversionPatterns(
        &getContext(), converter, patterns, base_level);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
//
void hecate::earth::populateEarthToCKKSConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns, int64_t init_level) {
  // clang-format off
  patterns.add<ConstantOpLowering> (converter, ctxt, init_level);
  patterns.add<
    MulOpLowering,
    AddOpLowering,
    NegateOpLowering,
    RotateOpLowering,
    RescaleOpLowering,
    UpscaleOpLowering,
    ModswitchOpLowering,
    ReturnOpLowering
  >(converter, ctxt);

  // clang-format on
}

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
hecate::earth::createEarthToCKKSConversionPass() {
  return std::make_unique<EarthToCKKSConversion>();
}
