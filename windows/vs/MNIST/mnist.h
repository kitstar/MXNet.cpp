# pragma once

# include "MLP.h"

class Mnist : public Mlp
{
public:
    Mnist();
    
    virtual void Run(mxnet::cpp::KVStore *kv_store)
    {
        switch (running_op_)
        {
        case (RunningMethods::kTrain) :
            Train(kv_store);
            break;
        }
    }

    virtual void Train(mxnet::cpp::KVStore *kv_store);

protected:
    virtual double Accuracy(const mxnet::cpp::NDArray &result, const mxnet::cpp::NDArray &labels);

    virtual mxnet::cpp::Symbol BuildNetwork();
    
    virtual void load_model(std::string model_name)
    { }

    virtual void output_model(std::string model_name)
    { }


protected:
    std::vector<mxnet::cpp::NDArray> network_args;
    std::vector<mxnet::cpp::NDArray> arg_grad_store;
    std::vector<mxnet::cpp::OpReqType> grad_req_type;

    std::map<std::string, mxnet::cpp::NDArray> args_map;

    float learning_rate_;
    float weight_decay_;
    int image_size_;    
};