# pragma once

# include <mutex>
# include <string>
# include "MLP.h"


class TrainAds : public Mlp 
{
public:
    TrainAds();
   
    virtual size_t train(std::string file_name, std::string kvstore_args);

    virtual size_t predict(std::string file_name, std::string model_name, std::string kvstore_args);    

protected:
    virtual void build_network();
    
    virtual void output_model(std::string model_name);

    virtual void load_model(std::string model_name);

    
private:
	std::vector<mxnet::cpp::NDArray> in_args;
	std::vector<mxnet::cpp::NDArray> arg_grad_store;
	std::vector<mxnet::cpp::OpReqType> grad_req_type;

    mxnet::cpp::Symbol mlp;
        
    double learning_rate = 0.01;
    double weight_decay = 1e-5;    
};