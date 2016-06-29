/*!
*  Copyright (c) 2016 by Contributors
* \file Mlp.h
* \brief base class of MLP
* \author Cheng CHEN
*/
# pragma once

# include "MxNetCpp.h"
# include "chana_ps.h"
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


void get_file_stream(const std::string &file_name, dmlc::Stream *&stream, size_t *file_size, const char *op);

DataReader * get_file_reader(const std::string &file_name, int buffer_sample_count, int sample_size, int my_rank, int total_rank);

void merge_files(std::string final_name, int count);

class Mlp
{
public:
    Mlp() : ctx_cpu(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
        ctx_dev(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
        running_mode_(sync_mode_t::Local), running_op_(RunningMethods::kInvalid)
    {
        auto mode = chana_config_get_value_string(mxnet_section.c_str(), "running_mode", "local", "");
        if (strcmp(mode, "async") == 0) running_mode_ = sync_mode_t::Async;
        else if (strcmp(mode, "sync") == 0) running_mode_ = sync_mode_t::Sync;

        auto op = chana_config_get_value_string(mxnet_section.c_str(), "operation", "", "");
        if (strcmp(op, "train") == 0) running_op_ = RunningMethods::kTrain;
        else if (strcmp(op, "predict") == 0) running_op_ = RunningMethods::kPredict;
        else if (strcmp(op, "validation") == 0) running_op_ = RunningMethods::kValidate;        
    }


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

protected:    
    virtual void load_model(std::string model_name) = 0;

    virtual void output_model(std::string model_name) = 0;

    virtual void build_network()
    {
        LG << "Not implement [build_network]!";
        exit(-1);
    }

    virtual mxnet::cpp::Symbol BuildNetwork()
    {
        LG << "Not implement [BuildNetwork]!";
        exit(-1);
    }

    virtual double ValAccuracy(mxnet::cpp::Symbol mlp,
        const mxnet::cpp::NDArray& samples,
        const mxnet::cpp::NDArray& labels);

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

    int batch_size_;
    int sample_size;
    int epoch_count_;
};