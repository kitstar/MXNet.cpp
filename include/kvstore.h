/*!
*  Copyright (c) 2016 by Contributors
* \file kvstore.h
* \brief definition of kvstore
* \author Chuntao Hong, Cheng CHEN
*/

#ifndef MXNETCPP_KVSTORE_H
#define MXNETCPP_KVSTORE_H

#include <numeric>
#include <string>
#include <vector>
#include "ndarray.h"
#include "optimizer.h"

namespace mxnet {
namespace cpp {

class KVStore {
 public:
  explicit KVStore(const std::string& name = "local");
  KVStore(const KVStore &) = delete;  
  KVStore(KVStore &&);
  void RunServer();
  inline void Init(int key, const NDArray& val);
  void Init(const std::vector<int>& keys, const std::vector<NDArray>& vals);
  void Push(int key, const NDArray& val, int priority = 0);
  void Push(const std::vector<int>& keys,
      const std::vector<NDArray>& vals, int priority = 0);  
  void Pull(int key, NDArray* out, int priority = 0);
  void Pull(const std::vector<int>& keys, std::vector<NDArray>* outs, int priority = 0);
  void AllReduce(std::vector<NDArray>* vals);
  // TODO(lx): put lr in optimizer or not?
  inline void SetOptimizer(std::unique_ptr<Optimizer> optimizer, bool is_local = false);
  inline std::string GetType() const;
  int GetRank() const;
  int GetNumWorkers() const;
  inline std::string GetRole() const;
  ~KVStore() { MXKVStoreFree(handle_); }

  void Barrier() const;

 private:
  KVStoreHandle handle_;
  std::unique_ptr<Optimizer> optimizer_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_KVSTORE_H