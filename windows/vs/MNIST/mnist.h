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

        case (RunningMethods::kPredict) :
            Predict(kv_store, false);
            break;

        case (RunningMethods::kValidate) :
            Predict(kv_store, true);
            break;

        default:
            LG << "Invalid Running Operation.";
        }
    }

    virtual void Train(mxnet::cpp::KVStore *kv_store);

    virtual void Predict(mxnet::cpp::KVStore *kv_store, bool is_validation);

    virtual void PrintResult(dmlc::Stream *stream, const mxnet::cpp::NDArray &result, size_t sample_count);

protected:
    virtual size_t CorrectCount(const mxnet::cpp::NDArray &result, const mxnet::cpp::NDArray &labels, size_t sample_count);

    virtual mxnet::cpp::Symbol BuildNetwork();
    
    virtual void load_model(std::string model_name)
    { }

    virtual void output_model(std::string model_name)
    { }


protected:    
    float learning_rate_;
    float weight_decay_;
    int image_size_;    
};