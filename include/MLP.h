/*!
*  Copyright (c) 2016 by Contributors
* \file Mlp.h
* \brief base class of MLP
* \author Cheng CHEN
*/
# pragma once

# include <unordered_map>
# include "mxnet-cpp/MxNetCpp.h"
# include "chana/chana_ps.h"
# include "data.h"

enum class sync_mode_t
{
    Invalid = 0,
    Local,
    Sync,
    Async
};

enum class RunningMethods
{
    kInvalid = 0,
    kTrain,
    kPredict,
    kValidate
};


class Mlp
{
public:
    Mlp();

    virtual void Run(mxnet::cpp::KVStore *kv_store)
    {
        LG << "Not implement!";
        exit(-1);
    }

    virtual void Train(mxnet::cpp::KVStore *kv_store)
    {
        LG << "Not implement!";
        exit(-1);
    }

    virtual void Predict(mxnet::cpp::KVStore *kv_store, bool is_validation)
    {
        LG << "Not implement!";
        exit(-1);
    }

    virtual size_t train(std::string file_name, std::string kvstore_args)
    {
        LG << "Not implement!";
        exit(-1);
    }

    virtual size_t predict(std::string file_name, std::string model_name, std::string kvstore_args)
    {
        LG << "Not implement!";
        exit(-1);
    }

    static std::string generate_kvstore_args(sync_mode_t mode, std::string machine_list, std::string ps_per_machine);

    static mxnet::cpp::KVStore * InitializeKvstore(sync_mode_t mode, std::string machine_list, std::string ps_per_machine);

    static mx_float GetSoftmaxResult(const mx_float *prediction, int cat_num);

    static size_t BuildVocabulary(const std::string &vocab_file, const std::string &corpus_file, std::unordered_map<std::string, uint32_t> &vocab_map, const std::string &unknown_token);

protected:
    virtual mxnet::cpp::Symbol BuildNetwork()
    {
        LG << "Not implement [BuildNetwork]!";
        exit(-1);
    }

    virtual bool SaveModel(const std::string &model_name, std::vector<mxnet::cpp::NDArray> &parameters);

    virtual bool LoadModel(const std::string &model_name, std::vector<mxnet::cpp::NDArray> &parameters);

    virtual double Accuracy(const mxnet::cpp::NDArray& result, const mxnet::cpp::NDArray& labels);


public:
    std::string     output_file;
    sync_mode_t     running_mode_;
    RunningMethods  running_op_;

protected:
    const std::string mxnet_section = "apps.MXNET";

    mxnet::cpp::Context ctx_cpu;
    mxnet::cpp::Context ctx_dev;

    std::map<std::string, mxnet::cpp::NDArray> args_map;
    std::map<std::string, mxnet::cpp::OpReqType> grad_type_map;

    int batch_size_;
    int sample_size;
    int epoch_count_;
};