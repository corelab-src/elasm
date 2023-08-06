
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"

using namespace hecate;

namespace {

struct SMUBuilder {
  SMUBuilder(mlir::Operation *op) : idMax(0), _op(op) { build(); }

  int64_t getID(mlir::Value v) {
    if (smuIds.find(v) != smuIds.end())
      return smuIds[v];
    else
      return -1;
  }
  int64_t getID(mlir::Operation *op);
  std::set<int64_t> idSet;

  void build();
  void splitForward();
  void defineGeneral(mlir::Value v, bool isConsume, bool isForward);
  void lookGeneral(mlir::Value v, bool isConsume, bool isForward);
  void splitBackward();

  int64_t idMax;
  std::map<std::int64_t, std::int64_t> originID;
  llvm::SmallVector<std::pair<mlir::Value, bool>, 4> valueSet;

  llvm::DenseMap<int64_t, bool> consumeDefinition;
  llvm::DenseMap<int64_t, std::set<int64_t>> definition;

  std::map<std::pair<int64_t, std::set<int64_t>>, int64_t> forwardMap;
  std::map<std::pair<int64_t, std::set<int64_t>>, int64_t> backwardMap;
  std::map<std::pair<int64_t, std::set<int64_t>>, int64_t> forwardConsumeMap;
  std::map<std::pair<int64_t, std::set<int64_t>>, int64_t> backwardConsumeMap;

  llvm::DenseMap<mlir::Value, int64_t> smuIds;
  mlir::Operation *_op;
};

} // namespace

void SMUBuilder::build() {
  // Prepare the analysis
  llvm::SmallVector<hecate::earth::HEScaleOpInterface, 4> hopSet;
  // Initialize
  idMax = 1;
  valueSet.resize(0);

  _op->walk([&](mlir::Block *block) {
    for (auto &&arg : block->getArguments()) {
      smuIds[arg] = 0;
      valueSet.push_back({arg, false});
    }
  });

  _op->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      assert(0 && "Currently not supported");
    }
    for (auto &&val : sop.getOperation()->getResults()) {
      valueSet.push_back({val, sop.isConsume()});
      smuIds[val] = 0;
    }
  });

  smuIds.reserve(valueSet.size());

  idSet.insert(0);

  uint64_t id_count = 0;

  std::map<int64_t, int64_t> idToIdMap;
  std::map<int64_t, std::vector<mlir::Value>> revID;

  while (id_count != idSet.size()) {
    // Current implementation does not reflect associativity
    id_count = idSet.size();

    originID.clear();
    definition.clear();
    splitForward();

    originID.clear();
    definition.clear();
    splitBackward();

    idSet.clear();
    for (auto &&id : smuIds) {
      idSet.insert(id.second);
    }
  }
}

void SMUBuilder::splitForward() {

  for (auto &&v : valueSet) {
    defineGeneral(v.first, v.second, true);
  }
  for (auto &&v : valueSet) {
    lookGeneral(v.first, v.second, true);
  }
}
void SMUBuilder::splitBackward() {
  for (auto &&it = valueSet.rbegin(); it != valueSet.rend(); ++it) {
    defineGeneral(it->first, it->second, false);
  }
  for (auto &&it = valueSet.rbegin(); it != valueSet.rend(); ++it) {
    lookGeneral(it->first, it->second, false);
  }
}

void SMUBuilder::defineGeneral(mlir::Value v, bool isConsume, bool isForward) {

  auto nid = getID(v);
  auto &&correctMap = isForward
                          ? (isConsume ? forwardConsumeMap : forwardMap)
                          : (isConsume ? backwardConsumeMap : backwardMap);
  auto &&defIter = definition.find(nid);
  if (defIter == definition.end()) {
    std::set<int64_t> def;
    if (isForward && v.isa<mlir::OpResult>()) {
      for (auto &&oper : v.getDefiningOp()->getOpOperands()) {
        def.insert(getID(oper.get()));
      }
    } else if (!isForward) {
      for (auto &&oper : v.getUses()) {
        def.insert(getID(oper.getOwner()));
      }
    }
    consumeDefinition[nid] = isConsume;
    definition[nid] = def;
    correctMap[{nid, def}] = nid;
  }
}

void SMUBuilder::lookGeneral(mlir::Value v, bool isConsume, bool isForward) {
  // Build key
  auto nid = getID(v);
  std::set<int64_t> def = {};
  auto &&correctMap = isForward
                          ? (isConsume ? forwardConsumeMap : forwardMap)
                          : (isConsume ? backwardConsumeMap : backwardMap);

  if (isForward && v.isa<mlir::OpResult>()) {
    for (auto &&oper : v.getDefiningOp()->getOpOperands()) {
      def.insert(getID(oper.get()));
    }
  } else if (!isForward) {
    for (auto &&oper : v.getUses()) {
      def.insert(getID(oper.getOwner()));
    }
  }

  // Perform Sub Key Expansion
  for (auto &&defi : def) {
    if ((defi == nid || (originID.count(defi) && originID[defi] == nid)) &&
        !consumeDefinition[defi]) {
      std::set<int64_t> subkey = def;
      subkey.erase(defi);
      auto defidefi = definition[defi];
      if (std::includes(defidefi.begin(), defidefi.end(), subkey.begin(),
                        subkey.end())) {
        def = defidefi;
        break;
      }
    }
  }

  // Look key
  auto res = correctMap.try_emplace({nid, def}, idMax);
  // Add defintion if new group is created
  if (res.second) {
    definition[idMax] = def;
    consumeDefinition[idMax] = isConsume;
    originID[idMax] = nid;
    ++idMax;
  }
  // Update SMU
  smuIds[v] = res.first->second;
}

int64_t SMUBuilder::getID(mlir::Operation *op) {
  return op->getNumResults() > 0 ? smuIds[op->getResult(0)] : -1;
}

void setIntegerAttr(llvm::StringRef name, mlir::Value v, int64_t data) {
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

int64_t getIntegerAttr(llvm::StringRef name, mlir::Value v) {
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

hecate::ScaleManagementUnit::ScaleManagementUnit(mlir::Operation *op)
    : idMax(0), smuIds(), smuEdges(), idToValue(), edgeToOper(), _op(op) {

  int64_t idNum = 0;
  llvm::SmallVector<mlir::Value, 4> values;
  _op->walk([&](mlir::Block *block) {
    for (auto &&arg : block->getArguments()) {
      values.push_back(arg);
    }
  });
  _op->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      assert(0 && "Currently not supported");
    }
    for (auto &&val : sop.getOperation()->getResults()) {
      values.push_back(val);
    }
  });
  if (op->hasAttr("smu_attached") &&
      op->getAttrOfType<mlir::BoolAttr>("smu_attached").getValue()) {
    for (auto &&value : values) {
      auto id = getIntegerAttr("smu", value);
      smuIds[value] = id;
      idNum = std::max(id + 1, idNum);
    }
  } else {
    auto &&builder = std::make_unique<SMUBuilder>(op);
    // Build ID to ID reordering map
    llvm::DenseMap<int64_t, int64_t> idToIdMap;
    for (auto &&val : values) {
      idToIdMap.try_emplace(builder->getID(val), idToIdMap.size());
    }
    // Build sorted smuIds;
    for (auto &&kv : builder->smuIds) {
      smuIds[kv.first] = idToIdMap[kv.second];
    }
    idNum = idToIdMap.size();
  }

  noisyMap.resize(idNum);
  for (auto &&kv : smuIds) {
    if (auto &&res = kv.first.dyn_cast<mlir::OpResult>()) {
      if (auto &&sop =
              dyn_cast<earth::HEScaleOpInterface>(res.getDefiningOp())) {
        if (sop.isNoisy()) {
          noisyMap[kv.second] = true;
        }
      }
    }
  }

  idToValue.resize(idNum);

  // Build ID to Value mapping
  for (auto &&kv : smuIds) {
    idToValue[kv.second].push_back(kv.first);
  }

  // Build Edge datas
  std::map<std::pair<int64_t, int64_t>, int64_t> edgeMap;
  for (auto &&val : values) {
    for (auto &&oper : val.getUses()) {
      if (getID(oper.get()) == getID(oper.getOwner())) {
        // Self-edge should be removed
        continue;
      }
      auto add = edgeMap.try_emplace(
          {getID(oper.get()), getID(oper.getOwner())}, edgeMap.size());
      smuEdges[&oper] = add.first->second;
    }
  }

  edgeToOper.resize(edgeMap.size());
  for (auto &&kv : smuEdges) {
    edgeToOper[kv.second].push_back(kv.first);
  }
}

int64_t hecate::ScaleManagementUnit::getID(mlir::Value v) const {
  auto &&id = smuIds.find(v);
  if (id != smuIds.end())
    return id->second;
  else
    return -1;
}
int64_t hecate::ScaleManagementUnit::getEdge(mlir::OpOperand *op) const {
  auto &&id = smuEdges.find(op);
  if (id != smuEdges.end())
    return id->second;
  else
    return -1;
}
mlir::SmallVector<mlir::OpOperand *, 4>
hecate::ScaleManagementUnit::getEdgeSet(int64_t edge) const {
  return edgeToOper[edge];
}
mlir::SmallVector<mlir::Value, 4>
hecate::ScaleManagementUnit::getValueSet(int64_t id) const {
  return idToValue[id];
}
int64_t hecate::ScaleManagementUnit::getNumEdges() const {
  return edgeToOper.size();
}
int64_t hecate::ScaleManagementUnit::getNumSMUs() const {
  return idToValue.size();
}
int64_t hecate::ScaleManagementUnit::getID(mlir::Operation *op) const {
  return op->getNumResults() > 0 ? getID(op->getResult(0)) : -1;
}
bool hecate::ScaleManagementUnit::inNoisyGroup(mlir::Operation *op) const {
  return noisyMap[getID(op)];
}
bool hecate::ScaleManagementUnit::inNoisyGroup(mlir::Value v) const {
  return noisyMap[getID(v)];
}

bool hecate::ScaleManagementUnit::isInvalidated(
    const mlir::AnalysisManager::PreservedAnalyses &) {
  // Before fix SMU to handle allow scale management op,
  // We need to invalidate smu always.
  detach();
  return true;
}

void hecate::ScaleManagementUnit::attach() {
  mlir::OpBuilder builder(_op);

  _op->setAttr("smu_attached", builder.getBoolAttr(true));

  for (int i = 0; i < getNumSMUs(); i++) {
    for (auto &&vv : getValueSet(i)) {
      hecate::setIntegerAttr("smu", vv, i);
    }
  }
}
void hecate::ScaleManagementUnit::detach() {
  mlir::OpBuilder builder(_op);
  _op->setAttr("smu_attached", builder.getBoolAttr(false));
}

bool hecate::ScaleManagementUnit::verify() const {

  // Invariant  : SMUs need to have an unique definition
  // Invariant  : Definitions need to have an unique SMUs
  // Definition : Set of in- and out- IDs and its consumeness
  // Limitation : Cannot check the minimality of the unit generation
  // Assumption : Non-consuming SMU can have self-edge

  llvm::SmallVector<mlir::Value, 4> values;
  _op->walk([&](mlir::Block *block) {
    for (auto &&arg : block->getArguments()) {
      values.push_back(arg);
    }
  });
  bool checker = true;
  _op->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      /* assert(0 && "Currently not supported"); */
      checker = false;
    }
    for (auto &&val : sop.getOperation()->getResults()) {
      values.push_back(val);
    }
  });
  if (!checker)
    return false;
  std::map<int64_t, bool> consumeness;
  std::map<int64_t, std::set<int64_t>> forwardSMU;
  std::map<int64_t, std::set<int64_t>> backwardSMU;
  std::map<std::tuple<std::set<int64_t>, std::set<int64_t>, bool>, int64_t>
      definition;

  // Check consumess
  for (auto &&val : values) {
    if (auto res = val.dyn_cast<mlir::OpResult>()) {
      if (auto hop =
              llvm::dyn_cast<earth::HEScaleOpInterface>(val.getDefiningOp())) {
        auto &&it = consumeness.try_emplace(getID(val), hop.isConsume());
        if (it.first->second != hop.isConsume()) {
          val.dump();
          assert(0 && "Consumeness Violation for HEScaleOp");
        }
      } else {
        auto &&it = consumeness.try_emplace(getID(val), false);
        if (it.first->second != false) {
          val.dump();
          assert(0 && "Consumeness Violation for non-HEScaleOp");
        }
      }

    } else {
      auto &&it = consumeness.try_emplace(getID(val), false);
      if (it.first->second != false) {
        val.dump();
        assert(0 && "Consumeness Violation for input argument");
      }
    }
  }

  // Build Forward and check uniqueness of definition
  for (auto &&val : values) {
    // block argument does not have forward Def
    std::set<int64_t> def;
    if (auto res = val.dyn_cast<mlir::OpResult>()) {
      for (auto &&oper : res.getDefiningOp()->getOpOperands()) {
        if (getID(oper.get()) == getID(oper.getOwner())) {
          if (consumeness[getID(oper.getOwner())]) {
            oper.get().dump();
            assert(0 && "Consuming Op cannot has self edge");
          };
          auto &&canonDef = forwardSMU.find(getID(oper.get()));
          if (canonDef == forwardSMU.end()) {
            oper.get().dump();
            assert(0 && "Used before definition");
          }
          for (auto &&defi : canonDef->second) {
            // Self-edge should be replaced
            def.insert(defi);
          }
        } else {
          def.insert(getID(oper.get()));
        }
      }
    }
    auto add = forwardSMU.try_emplace(getID(val), def);
    if (add.first->second != def) {
      val.dump();
      assert(0 && "Definition mismatch ");
    }
  }

  // Build Backward and check uniqueness of definition
  for (auto &&vali = values.rbegin(); vali != values.rend(); ++vali) {
    // block argument does not have forward Def
    auto val = *vali;
    std::set<int64_t> def;
    for (auto &&oper : val.getUses()) {
      if (getID(oper.get()) == getID(oper.getOwner())) {
        if (consumeness[getID(oper.get())]) {
          oper.get().dump();
          assert(0 && "Consuming Op cannot has self edge");
        };
        auto &&canonDef = backwardSMU.find(getID(oper.getOwner()));
        if (canonDef == backwardSMU.end()) {
          oper.get().dump();
          assert(0 && "Used before definition");
        }
        for (auto &&defi : canonDef->second) {
          // Self-edge should be expanded
          def.insert(defi);
        }
      } else {
        def.insert(getID(oper.getOwner()));
      }
    }
    auto add = backwardSMU.try_emplace(getID(val), def);
    if (add.first->second != def) {
      val.dump();
      assert(0 && "Definition mismatch ");
    }
  }

  // Build Backward and check uniqueness of definition
  // Check Uniqueness of SMU
  for (uint64_t i = 0; i < forwardSMU.size(); i++) {
    auto add = definition.try_emplace(
        {forwardSMU[i], backwardSMU[i], consumeness[i]}, i);
    if (!add.second) {
      assert(0 && "Duplicated defintion");
    }
  }
  return true;
}
