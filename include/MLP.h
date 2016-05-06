/*!
*  Copyright (c) 2016 by Contributors
* \file Mlp.h
* \brief base class of MLP
* \author Chuntao Hong, Cheng CHEN
*/

# pragma once
# include "ndarray.h"
# include "symbol.h"

enum class sync_mode_t
{
    Invalid = 0,
    Sync,
    Async
};

class Mlp
{
public:
    Mlp() : ctx_cpu(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
        ctx_dev(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0))        
    { }
    
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

protected:    
    virtual void load_model(std::string model_name) = 0;

    virtual void output_model(std::string model_name) = 0;
    
    virtual double ValAccuracy(mxnet::cpp::Symbol mlp,
        const mxnet::cpp::NDArray& samples,
        const mxnet::cpp::NDArray& labels);

    virtual double Accuracy(const mxnet::cpp::NDArray& result, const mxnet::cpp::NDArray& labels);  


public:    
    std::string output_file;


protected:
    mxnet::cpp::Context ctx_cpu;
    mxnet::cpp::Context ctx_dev;

    int batchSize;
    int sampleSize;
    int epochCount;
};