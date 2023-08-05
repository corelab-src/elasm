
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"

#include "hecate/Dialect/CKKS/IR/PolyTypeInterfaceTypes.cpp.inc"

#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOps.cpp.inc"

#include "hecate/Dialect/CKKS/IR/CKKSOpsDialect.cpp.inc"

/* #include "hecate/Dialect/CKKS/IR/EarthCanonicalizerPattern.inc" */

struct PolyTypeTensorModel
    : public hecate::ckks::PolyTypeInterface::ExternalModel<
          PolyTypeTensorModel, mlir::RankedTensorType> {
  unsigned getNumPoly(Type t) const {
    if (auto polyType = t.dyn_cast<mlir::RankedTensorType>()
                            .getElementType()
                            .dyn_cast<hecate::ckks::PolyTypeInterface>()) {
      return polyType.getNumPoly();
    } else {
      return 0;
    }
  }
  unsigned getLevel(Type t) const {
    if (auto polyType = t.dyn_cast<mlir::RankedTensorType>()
                            .getElementType()
                            .dyn_cast<hecate::ckks::PolyTypeInterface>()) {
      return polyType.getLevel();
    } else {
      return 0;
    }
  }

  hecate::ckks::PolyTypeInterface switchLevel(Type t, unsigned level) const {
    return mlir::RankedTensorType::get(
        t.dyn_cast<mlir::RankedTensorType>().getShape(),
        t.dyn_cast<mlir::RankedTensorType>()
            .getElementType()
            .dyn_cast<hecate::ckks::PolyTypeInterface>()
            .switchLevel(level));
  }
  hecate::ckks::PolyTypeInterface switchNumPoly(Type t,
                                                unsigned num_poly) const {
    return mlir::RankedTensorType::get(
        t.dyn_cast<mlir::RankedTensorType>().getShape(),
        t.dyn_cast<mlir::RankedTensorType>()
            .getElementType()
            .dyn_cast<hecate::ckks::PolyTypeInterface>()
            .switchNumPoly(num_poly));
  }
};

void hecate::ckks::CKKSDialect::initialize() {
  // Registers all the Types into the EVADialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "hecate/Dialect/CKKS/IR/CKKSOpsTypes.cpp.inc"
      >();

  // Registers all the Operations into the EVADialect class
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/CKKS/IR/CKKSOps.cpp.inc"
      >();
  mlir::RankedTensorType::attachInterface<PolyTypeTensorModel>(*getContext());
}

::mlir::LogicalResult hecate::ckks::EncodeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {

  auto op = EncodeOpAdaptor(operands, attributes, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  if (dPoly.getNumPoly() == 1 &&
      (dPoly.getLevel() == op.getLevel() || dPoly.getLevel() == 0)) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::RescaleCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RescaleCOpAdaptor(operands, attributes, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getNumPoly() == lPoly.getNumPoly() &&
      (dPoly.getLevel() == lPoly.getLevel() - 1 || dPoly.getLevel() == 0)) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::ModswitchCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ModswitchCOpAdaptor(operands, attributes, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getNumPoly() == lPoly.getNumPoly() &&
      (dPoly.getLevel() == lPoly.getLevel() - op.getDownFactor() ||
       dPoly.getLevel() == 0)) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::AddCPOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddCPOpAdaptor(operands, attributes, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getLhs());
  auto rPoly = ckks::getPolyType(op.getRhs());
  if (std::min(lPoly.getNumPoly(), rPoly.getNumPoly()) == 1 &&
      dPoly.getNumPoly() == std::max(rPoly.getNumPoly(), lPoly.getNumPoly()) &&
      lPoly.getLevel() == rPoly.getLevel() &&
      dPoly.getLevel() == lPoly.getLevel()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::MulCPOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddCPOpAdaptor(operands, attributes, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getLhs());
  auto rPoly = ckks::getPolyType(op.getRhs());

  if (std::min(lPoly.getNumPoly(), rPoly.getNumPoly()) == 1 &&
      dPoly.getNumPoly() == std::max(lPoly.getNumPoly(), rPoly.getNumPoly()) &&
      lPoly.getLevel() == rPoly.getLevel() &&
      lPoly.getLevel() == dPoly.getLevel()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}
hecate::ckks::PolyTypeInterface
hecate::ckks::PolyType::switchLevel(unsigned level) const {
  return get(getContext(), getNumPoly(), level);
}
hecate::ckks::PolyTypeInterface
hecate::ckks::PolyType::switchNumPoly(unsigned num_poly) const {
  return get(getContext(), num_poly, getLevel());
}

mlir::RankedTensorType hecate::ckks::getTensorType(mlir::Value v) {
  return v.getType().dyn_cast<mlir::RankedTensorType>();
}
hecate::ckks::PolyTypeInterface hecate::ckks::getPolyType(mlir::Value v) {
  return v.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
  /* .dyn_cast<mlir::TensorType>() */
  /* .getElementType() */
  /* .dyn_cast<hecate::earth::HEScaleTypeInterface>(); */
}
