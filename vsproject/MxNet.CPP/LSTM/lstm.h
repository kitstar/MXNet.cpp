# pragma once

# include "MLP.h"


class LSTM : public Mlp
{
public:
    LSTM();

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

    virtual void Predict(mxnet::cpp::KVStore *kv_store, bool is_validation)
    {
    }

    virtual void PrintResult(dmlc::Stream *stream, const mxnet::cpp::NDArray &result, size_t sample_count)
    {

    }

protected:
    virtual mxnet::cpp::Symbol BuildNetwork();


protected:
    std::unordered_map<std::string, uint32_t> vocab_map;
    
    double  learning_rate_;
    double  weight_decay_;
    int     sentence_length_;

    int     embedding_size_;
    int     num_lstm_layer_;
    int     num_hidden_;
};