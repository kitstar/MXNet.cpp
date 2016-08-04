# pragma once

# include "MLP.h"


class LSTM : public Mlp
{
protected:
    struct LstmState
    {
        LstmState(std::string h_name, std::string c_name) : h(mxnet::cpp::Symbol::Variable(h_name)), c(mxnet::cpp::Symbol::Variable(c_name)) 
        { }

        mxnet::cpp::Symbol h;
        mxnet::cpp::Symbol c;
    };

    
    struct LstmParam
    {
        LstmParam(std::string name) : 
            i2h_weight(mxnet::cpp::Symbol::Variable("i2h weight" + name)),
            i2h_bias(mxnet::cpp::Symbol::Variable("i2h bias" + name)),
            h2h_weight(mxnet::cpp::Symbol::Variable("h2h weight" + name)),
            h2h_bias(mxnet::cpp::Symbol::Variable("h2h bias" + name))
        { }

        mxnet::cpp::Symbol i2h_weight;
        mxnet::cpp::Symbol i2h_bias;
        mxnet::cpp::Symbol h2h_weight;
        mxnet::cpp::Symbol h2h_bias;
    };


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

    LstmState Cell(
        int num_hidden, const mxnet::cpp::Symbol &indata, 
        const LstmState &prev_state, const LstmParam &param, int seq_idx,
        int layer_idx, double dropout = 0);    


protected:
    std::unordered_map<std::string, uint32_t> vocab_map;

    std::string unknown_token_;
    
    double  learning_rate_;
    double  weight_decay_;
    double  momentum_;

    double  drop_out_;
    int     vocab_size_;
    int     embedding_size_;
    int     num_steps_;
    int     num_lstm_layer_;
    int     hidden_size_;
};