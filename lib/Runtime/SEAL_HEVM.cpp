#include <cassert>
#include <fstream>
#include <iostream>

#include <map>
#include <seal/encryptionparams.h>
#include <seal/galoiskeys.h>
#include <seal/seal.h>

#include <type_traits>
#include <vector>

#include "hecate/Support/HEVMHeader.h"

struct SEAL_HEVM {
  std::vector<std::vector<double>> buffer;
  HEVMHeader header;
  ConfigBody config;
  /* std::vector<uint64_t> config_dats; */
  std::vector<HEVMOperation> ops;
  std::vector<uint64_t> arg_scale;
  std::vector<uint64_t> arg_level;
  std::vector<uint64_t> res_scale;
  std::vector<uint64_t> res_level;
  std::vector<uint64_t> res_dst;

  std::vector<seal::Ciphertext> ciphers;
  std::vector<seal::Plaintext> plains;
  std::map<uint64_t, seal::Plaintext> upscale_const;

  seal::EncryptionParameters parms;
  std::unique_ptr<seal::GaloisKeys> gal_key;
  std::unique_ptr<seal::RelinKeys> relin_key;
  std::unique_ptr<seal::Encryptor> encryptor;
  std::unique_ptr<seal::Evaluator> evaluator;
  std::unique_ptr<seal::Decryptor> decryptor;
  std::unique_ptr<seal::CKKSEncoder> encoder;

  static const int N = 15;
  static const int L = 14;

  bool debug = false;

  static void create_context(char *dir) {

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    auto strdir = std::string(dir);
    parms.set_poly_modulus_degree(1LL << N);
    std::vector<int> coeffs;
    for (int i = 0; i < L; i++) {
      coeffs.push_back(60);
    }
    parms.set_coeff_modulus(seal::CoeffModulus::Create(1LL << N, coeffs));
    {
      std::ofstream f(strdir + "/parm.seal", std::ios::out | std::ios::binary);
      parms.save(f);
      f.close();
    }
    auto context = seal::SEALContext(parms);
    seal::KeyGenerator keygen(context);
    seal::PublicKey pubkey;
    {
      std::ofstream f(strdir + "/pub.seal", std::ios::out | std::ios::binary);
      keygen.create_public_key(pubkey);
      pubkey.save(f);
      f.close();
    }
    seal::SecretKey seckey = keygen.secret_key();
    {

      std::ofstream f(strdir + "/sec.seal", std::ios::out | std::ios::binary);
      seckey.save(f);
      f.close();
    }
    seal::RelinKeys relkey;
    {
      std::ofstream f(strdir + "/relin.seal", std::ios::out | std::ios::binary);
      keygen.create_relin_keys(relkey);
      relkey.save(f);
      f.close();
    }
    seal::GaloisKeys galkey;
    keygen.create_galois_keys(galkey);
    {
      std::ofstream f(strdir + "/gal.seal", std::ios::out | std::ios::binary);
      galkey.save(f);
      f.close();
    }
  }

  void loadSEAL(char *dir) {
    auto strdir = std::string(dir);
    parms = seal::EncryptionParameters(seal::scheme_type::ckks);
    {
      std::ifstream f(strdir + "/parm.seal", std::ios::in | std::ios::binary);
      parms.load(f);
      f.close();
    }
    auto context = seal::SEALContext(parms);
    seal::PublicKey public_key;
    {
      std::ifstream f(strdir + "/pub.seal", std::ios::in | std::ios::binary);
      public_key.load(context, f);
      f.close();
    }
    seal::SecretKey secret_key;
    {
      std::ifstream f(strdir + "/sec.seal", std::ios::in | std::ios::binary);
      secret_key.load(context, f);
      f.close();
    }
    relin_key = std::make_unique<seal::RelinKeys>();
    {
      std::ifstream f(strdir + "/relin.seal", std::ios::in | std::ios::binary);
      relin_key->load(context, f);
      f.close();
    }
    gal_key = std::make_unique<seal::GaloisKeys>();
    {
      std::ifstream f(strdir + "/gal.seal", std::ios::in | std::ios::binary);
      gal_key->load(context, f);
      f.close();
    }

    encryptor = std::make_unique<seal::Encryptor>(context, public_key);
    decryptor = std::make_unique<seal::Decryptor>(context, secret_key);
    encoder = std::make_unique<seal::CKKSEncoder>(context);
    evaluator = std::make_unique<seal::Evaluator>(context);
  }

  void loadClient(char *dir) {
    auto strdir = std::string(dir);
    parms = seal::EncryptionParameters(seal::scheme_type::ckks);
    {
      std::ifstream f(strdir + "/parm.seal", std::ios::in | std::ios::binary);
      parms.load(f);
      f.close();
    }
    auto context = seal::SEALContext(parms);
    seal::PublicKey public_key;
    {
      std::ifstream f(strdir + "/pub.seal", std::ios::in | std::ios::binary);
      public_key.load(context, f);
      f.close();
    }
    seal::SecretKey secret_key;
    {
      std::ifstream f(strdir + "/sec.seal", std::ios::in | std::ios::binary);
      secret_key.load(context, f);
      f.close();
    }
    encryptor = std::make_unique<seal::Encryptor>(context, public_key);
    decryptor = std::make_unique<seal::Decryptor>(context, secret_key);
    encoder = std::make_unique<seal::CKKSEncoder>(context);
  }
  void loadServer(char *dir) {
    auto strdir = std::string(dir);
    parms = seal::EncryptionParameters(seal::scheme_type::ckks);
    {
      std::ifstream f(strdir + "parm.seal", std::ios::in | std::ios::binary);
      parms.load(f);
      f.close();
    }
    auto context = seal::SEALContext(parms);
    relin_key = std::make_unique<seal::RelinKeys>();
    {
      std::ifstream f(strdir + "/relin.seal", std::ios::in | std::ios::binary);
      relin_key->load(context, f);
      f.close();
    }
    gal_key = std::make_unique<seal::GaloisKeys>();
    {
      std::ifstream f(strdir + "/gal.seal", std::ios::in | std::ios::binary);
      gal_key->load(context, f);
      f.close();
    }

    encoder = std::make_unique<seal::CKKSEncoder>(context);
    evaluator = std::make_unique<seal::Evaluator>(context);
  }

  void loadConstants(char *name) {
    std::string sname(name);

    std::ifstream iff(sname, std::ios::binary);
    int64_t len;
    iff.read((char *)&len, sizeof(int64_t));
    buffer.resize(len);

    for (size_t i = 0; i < len; i++) {
      int64_t veclen;
      iff.read((char *)&veclen, sizeof(int64_t));
      std::vector<double> tmp;
      tmp.resize(veclen);
      iff.read((char *)tmp.data(), veclen * sizeof(double));
      buffer[i] = tmp;
    }
    iff.close();
    /* std::cerr << "Constant Loaded From" << sname << std::endl; */
  }

  void loadHEVM(char *name) {
    std::string sname(name);

    std::ifstream iff(sname, std::ios::binary);

    loadHeader(iff);

    ops.resize(config.num_operations);
    iff.read((char *)ops.data(), ops.size() * sizeof(HEVMOperation));

    ciphers.resize(config.num_ctxt_buffer);
    plains.resize(config.num_ptxt_buffer);
  }

  void loadHeader(std::istream &iff) {

    iff.read((char *)&header, sizeof(HEVMHeader));
    iff.read((char *)&config, sizeof(ConfigBody));

    arg_scale.resize(header.config_header.arg_length);
    arg_level.resize(header.config_header.arg_length);
    res_scale.resize(header.config_header.res_length);
    res_level.resize(header.config_header.res_length);
    res_dst.resize(header.config_header.res_length);
    iff.read((char *)arg_scale.data(), arg_scale.size() * sizeof(uint64_t));
    iff.read((char *)arg_level.data(), arg_level.size() * sizeof(uint64_t));
    iff.read((char *)res_scale.data(), res_scale.size() * sizeof(uint64_t));
    iff.read((char *)res_level.data(), res_level.size() * sizeof(uint64_t));
    iff.read((char *)res_dst.data(), res_dst.size() * sizeof(uint64_t));

    ciphers.resize(header.config_header.arg_length +
                   header.config_header.res_length);
  }

  void resetResDst() {
    for (int i = 0; i < header.config_header.res_length; i++) {
      res_dst[i] = i + header.config_header.arg_length;
    }
  }

  void preprocess() {

    std::vector<double> datas(1LL << (N - 1), 0.0);
    std::vector<double> identity(1LL << (N - 1), 1.0);
    for (HEVMOperation &op : ops) {
      if (op.opcode == 0) {
        encode_internal(plains[op.dst],
                        op.lhs == ((unsigned short)-1) ? identity
                                                       : buffer[op.lhs],
                        op.rhs >> 8, op.rhs & 0xFF);
      }
    }
  }

  void encode_internal(seal::Plaintext &dst, std::vector<double> src,
                       int8_t level, int8_t scale) {
    std::vector<double> datas(1LL << (N - 1), 0.0);
    for (int i = 0; i < datas.size(); i++) {
      datas[i] = src[i % src.size()];
    }
    encoder->encode(datas, std::pow(2.0, scale), dst);
    for (int i = L - 1; i > level; i--) {
      evaluator->mod_switch_to_next_inplace(dst);
    }
    return;
  }
  void encode(int16_t dst, int16_t src, int8_t level, int8_t scale) { return; }
  void rotate(int16_t dst, int16_t src, int16_t offset) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;

    evaluator->rotate_vector(ciphers[src], offset, *gal_key, ciphers[dst]);
  }
  void negate(int16_t dst, int16_t src) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    evaluator->negate(ciphers[src], ciphers[dst]);
  }
  void rescale(int16_t dst, int16_t src) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    evaluator->rescale_to_next(ciphers[src], ciphers[dst]);
  }
  void modswitch(int16_t dst, int16_t src, int16_t downFactor) {
    if (debug)
      std::cout << std::log2(ciphers[src].scale()) << std::endl;
    if (downFactor > 0)
      evaluator->mod_switch_to_next(ciphers[src], ciphers[dst]);
    for (int i = 1; i < downFactor; i++) {
      evaluator->mod_switch_to_next_inplace(ciphers[dst]);
    }
  }
  void upscale(int16_t dst, int16_t src, int16_t upFactor) {
    assert(0 && "This VM does not support native upscale op");
  }
  void addcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(ciphers[rhs].scale()) << std::endl;
    ciphers[lhs].scale() = ciphers[rhs].scale();
    evaluator->add(ciphers[lhs], ciphers[rhs], ciphers[dst]);
  }
  void addcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(plains[rhs].scale()) << std::endl;
    ciphers[lhs].scale() = plains[rhs].scale();
    evaluator->add_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
  }
  void mulcc(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(ciphers[rhs].scale()) << std::endl;
    evaluator->multiply(ciphers[lhs], ciphers[rhs], ciphers[dst]);
    evaluator->relinearize_inplace(ciphers[dst], *relin_key);
  }
  void mulcp(int16_t dst, int16_t lhs, int16_t rhs) {
    if (debug)
      std::cout << std::log2(ciphers[lhs].scale())
                << std::log2(plains[rhs].scale()) << std::endl;
    evaluator->multiply_plain(ciphers[lhs], plains[rhs], ciphers[dst]);
  }

  void run() {
    int i = (header.hevm_header_size + config.config_body_length) / 8;
    int j = 0;
    for (HEVMOperation &op : ops) {
      if (debug) {
        std::cout << std::oct << i++ << " " << std::dec << j++ << std::endl;
        std::cout << op.opcode << " " << op.dst << " " << op.lhs << " "
                  << op.rhs << std::endl;
      }
      switch (op.opcode) {
      case 0: { // Encode
        encode(op.dst, op.lhs, op.rhs >> 8, op.rhs & 0xFF);
        break;
      }
      case 1: { // RotateC
        rotate(op.dst, op.lhs, op.rhs);
        break;
      }
      case 2: { // NegateC
        negate(op.dst, op.lhs);
        break;
      }
      case 3: { // RescaleC
        rescale(op.dst, op.lhs);
        break;
      }
      case 4: { // ModswtichC
        modswitch(op.dst, op.lhs, op.rhs);
        break;
      }
      case 5: { // UpscaleC
        upscale(op.dst, op.lhs, op.rhs);
        break;
      }
      case 6: { // AddCC
        addcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 7: { // AddCP
        addcp(op.dst, op.lhs, op.rhs);
        break;
      }
      case 8: { // MulCC
        mulcc(op.dst, op.lhs, op.rhs);
        break;
      }
      case 9: { // MulCP
        mulcp(op.dst, op.lhs, op.rhs);
        break;
      }
      default: {
        break;
      }
      }
    }
  }
};

extern "C" {
void *initFullVM(char *dir) {
  auto vm = new SEAL_HEVM();
  vm->loadSEAL(dir);
  return (void *)vm;
}
void *initClientVM(char *dir) {
  auto vm = new SEAL_HEVM();
  vm->loadClient(dir);
  return (void *)vm;
}
void *initServerVM(char *dir) {
  auto vm = new SEAL_HEVM();
  vm->loadServer(dir);
  return (void *)vm;
}

void create_context(char *dir) { SEAL_HEVM::create_context(dir); }

// Loader for server
void load(void *vm, char *constant, char *vmfile) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  hevm->loadConstants(constant);
  hevm->loadHEVM(vmfile);
}

// Loader for client
void loadClient(void *vm, void *is) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  std::istream &iss = *static_cast<std::istream *>(is);
  hevm->loadHeader(iss);
  hevm->resetResDst();
}

// encryption and decryption uses internal buffer id
void encrypt(void *vm, int64_t i, double *dat, int len) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  seal::Plaintext ptxt;
  std::vector<double> dats(dat, dat + len);
  hevm->encode_internal(ptxt, dats, hevm->arg_level[i], hevm->arg_scale[i]);
  hevm->encryptor->encrypt(ptxt, hevm->ciphers[i]);
}
void decrypt(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  seal::Plaintext ptxt;
  hevm->decryptor->decrypt(hevm->ciphers[i], ptxt);
  std::vector<double> dats;
  hevm->encoder->decode(ptxt, dats);
  for (int i = 0; i < (1LL << (SEAL_HEVM::N - 1)); i++) {
    std::copy(dats.begin(), dats.end(), dat);
  }
}
// simple wrapper to elide getResIdx call
// use res_idx for i
void decrypt_result(void *vm, int64_t i, double *dat) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  decrypt(vm, hevm->res_dst[i], dat);
}

// We need this for communication code to access the proper buffer id
int64_t getResIdx(void *vm, int64_t i) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  return hevm->res_dst[i];
}

// use this to implement communication
void *getCtxt(void *vm, int64_t id) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  return &(hevm->ciphers[id]);
}

void preprocess(void *vm) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  hevm->preprocess();
}
void run(void *vm) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  hevm->run();
}
int64_t getArgLen(void *vm) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  return hevm->header.config_header.arg_length;
}
int64_t getResLen(void *vm) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  return hevm->header.config_header.res_length;
}
void setDebug(void *vm, bool enable) {
  auto hevm = static_cast<SEAL_HEVM *>(vm);
  hevm->debug = enable;
}
};
